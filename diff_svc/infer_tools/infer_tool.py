import json
import os
import time
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

import diff_svc.utils as utils
from diff_svc.infer_tools.f0_static import compare_pitch, static_f0_time
from diff_svc.modules.diff.diffusion import GaussianDiffusion
from diff_svc.modules.diff.net import DiffNet
from diff_svc.modules.vocoders.nsf_hifigan import NsfHifiGAN
from diff_svc.preprocessing.hubertinfer import HubertEncoder
from diff_svc.preprocessing.process_pipeline import File2Batch, get_pitch_parselmouth
from diff_svc.utils.hparams import hparams, set_hparams
from diff_svc.utils.pitch_utils import denorm_f0, norm_interp_f0


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None,res_type='polyphase')
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


class Svc:
    def __init__(self, project_name, config_name, hubert_gpu, model_path, onnx=False):
        self.project_name = project_name
        self.DIFF_DECODERS = {
            'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
        }

        self.model_path = model_path
        self.dev = torch.device("cpu")

        self._ = set_hparams(config=config_name, exp_name=self.project_name, infer=True,
                             reset=True, hparams_str='', print_hparams=False)

        hparams['hubert_gpu'] = hubert_gpu
        self.hubert = HubertEncoder(hparams['hubert_path'], onnx=onnx)
        self.model = GaussianDiffusion(
            phone_encoder=self.hubert,
            out_dims=hparams['audio_num_mel_bins'],
            denoise_fn=self.DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        utils.load_ckpt(self.model, self.model_path, 'model', force=True, strict=True)
        self.model.cpu()
        self.vocoder = NsfHifiGAN()

    def infer(self, in_path, key, acc, spk_id=0, use_crepe=True, singer=False):
        batch = self.pre(in_path, acc, spk_id, use_crepe)
        batch['f0'] = batch['f0'] + (key / 12)
        batch['f0'][batch['f0'] > np.log2(hparams['f0_max'])] = 0

        @timeit
        def diff_infer():
            spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
            energy = batch.get('energy').cpu() if batch.get('energy') else None
            if spk_embed is None:
                spk_embed = torch.LongTensor([0])
            diff_outputs = self.model(
                hubert=batch['hubert'].cpu(), spk_embed_id=spk_embed.cpu(), mel2ph=batch['mel2ph'].cpu(),
                f0=batch['f0'].cpu(), energy=energy, ref_mels=batch["mels"].cpu(), infer=True)
            return diff_outputs

        outputs = diff_infer()
        batch['outputs'] = outputs['mel_out']
        batch['mel2ph_pred'] = outputs['mel2ph']
        batch['f0_gt'] = denorm_f0(batch['f0'], batch['uv'], hparams)
        batch['f0_pred'] = outputs.get('f0_denorm')
        return self.after_infer(batch, singer, in_path)

    @timeit
    def after_infer(self, prediction, singer, in_path):
        for k, v in prediction.items():
            if type(v) is torch.Tensor:
                prediction[k] = v.cpu().numpy()

        # remove paddings
        mel_gt = prediction["mels"]
        mel_gt_mask = np.abs(mel_gt).sum(-1) > 0

        mel_pred = prediction["outputs"]
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

        f0_gt = prediction.get("f0_gt")
        f0_pred = prediction.get("f0_pred")
        if f0_pred is not None:
            f0_gt = f0_gt[mel_gt_mask]
        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]
        torch.cuda.is_available() and torch.cuda.empty_cache()

        if singer:
            data_path = in_path.replace("batch", "singer_data")
            mel_path = data_path[:-4] + "_mel.npy"
            f0_path = data_path[:-4] + "_f0.npy"
            np.save(mel_path, mel_pred)
            np.save(f0_path, f0_pred)
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        return f0_gt, f0_pred, wav_pred

    def pre(self, wav_fn, accelerate, spk_id=0, use_crepe=True):
        if isinstance(wav_fn, BytesIO):
            item_name = self.project_name
        else:
            song_info = wav_fn.split('/')
            item_name = song_info[-1].split('.')[-2]
        temp_dict = {'wav_fn': wav_fn, 'spk_id': spk_id, 'id': 0}

        temp_dict = File2Batch.temporary_dict2processed_input(item_name, temp_dict, self.hubert, infer=True,
                                                              use_crepe=use_crepe)
        hparams['pndm_speedup'] = accelerate
        batch = File2Batch.processed_input2batch([getitem(temp_dict)])
        return batch

    def evaluate_key(self, wav_path, key, auto_key):
        if "f0_static" in hparams.keys():
            f0_static = json.loads(hparams['f0_static'])
            wav, mel = self.vocoder.wav2spec(wav_path)
            input_f0 = get_pitch_parselmouth(wav, mel, hparams)[0]
            pitch_time_temp = static_f0_time(input_f0)
            eval_dict = {}
            for trans_key in range(-12, 12):
                eval_dict[trans_key] = compare_pitch(f0_static, pitch_time_temp, trans_key=trans_key)
            sort_key = sorted(eval_dict, key=eval_dict.get, reverse=True)[:5]
            print(f"推荐移调:{sort_key}")
            if auto_key:
                print(f"自动变调已启用，您的输入key被{sort_key[0]}key覆盖，控制参数为auto_key")
                return sort_key[0]
        else:
            print("config缺少f0_staic，无法使用自动变调，可通过infer_tools/data_static添加")
        return key


def getitem(item):
    max_frames = hparams['max_frames']
    spec = torch.Tensor(item['mel'])[:max_frames]
    mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
    f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
    hubert = torch.Tensor(item['hubert'][:hparams['max_input_tokens']])
    pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
    sample = {
        "id": item['id'],
        "spk_id": item['spk_id'],
        "item_name": item['item_name'],
        "hubert": hubert,
        "mel": spec,
        "pitch": pitch,
        "f0": f0,
        "uv": uv,
        "mel2ph": mel2ph,
        "mel_nonpadding": spec.abs().sum(-1) > 0,
    }
    if hparams['use_energy_embed']:
        sample['energy'] = item['energy']
    return sample
