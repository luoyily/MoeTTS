import hashlib
import json
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import parselmouth
import resampy
import torch
import torchcrepe
import librosa

import diff_svc.utils as utils
from diff_svc.modules.vocoders.nsf_hifigan import nsf_hifigan
from diff_svc.utils.hparams import hparams
from diff_svc.utils.pitch_utils import f0_to_coarse

warnings.filterwarnings("ignore")


class BinarizationError(Exception):
    pass


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


f0_dict = read_temp("./diff_svc/infer_tools/f0_temp.json")


def get_pitch_parselmouth(wav_data, mel, hparams):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    time_step = hparams['hop_size'] / hparams['audio_sample_rate']
    f0_min = hparams['f0_min']
    f0_max = hparams['f0_max']

    f0 = parselmouth.Sound(wav_data, hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size = (int(len(wav_data) // hparams['hop_size']) - len(f0) + 1) // 2
    f0 = np.pad(f0, [[pad_size, len(mel) - len(f0) - pad_size]], mode='constant')
    pitch_coarse = f0_to_coarse(f0, hparams)
    return f0, pitch_coarse


def get_pitch_crepe(wav_data, mel, hparams, threshold=0.05):
    device = torch.device("cpu")
    # device = torch.device("cuda")
    # crepe只支持16khz采样率，需要重采样
    # librosa.resample(wav, sr, 16000,res_type='polyphase')
    # wav16k = resampy.resample(wav_data, hparams['audio_sample_rate'], 16000)
    wav16k = librosa.resample(wav_data, hparams['audio_sample_rate'], 16000, res_type='polyphase')
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)

    # 频率范围
    f0_min = hparams['f0_min']
    f0_max = hparams['f0_max']

    # 重采样后按照hopsize=80,也就是5ms一帧分析f0
    model = 'tiny' if hparams['use_crepe_tiny'] else 'full'
    f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, f0_min, f0_max, pad=True, model=model, batch_size=1024,
                                device=device, return_periodicity=True)

    # 滤波，去掉静音，设置uv阈值，参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    # 将nan频率（uv部分）转换为0频率
    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    # 去掉0频率，并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(len(mel)) * hparams['hop_size'] / hparams['audio_sample_rate']
    if f0.shape[0] == 0:
        f0 = torch.FloatTensor(time_frame.shape[0]).fill_(0)
        print('f0 all zero!')
    else:
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    pitch_coarse = f0_to_coarse(f0, hparams)
    return f0, pitch_coarse


class File2Batch:
    '''
        pipeline: file -> temporary_dict -> processed_input -> batch
    '''

    @staticmethod
    def file2temporary_dict(raw_data_dir, ds_id):
        '''
            read from file, store data in temporary dicts
        '''
        raw_data_dir = Path(raw_data_dir)
        utterance_labels = []
        utterance_labels.extend(list(raw_data_dir.rglob(f"*.wav")))
        utterance_labels.extend(list(raw_data_dir.rglob(f"*.ogg")))

        all_temp_dict = {}
        for utterance_label in utterance_labels:
            item_name = str(utterance_label)
            temp_dict = {'wav_fn': str(utterance_label), 'spk_id': ds_id}
            all_temp_dict[item_name] = temp_dict
        return all_temp_dict

    @staticmethod
    def temporary_dict2processed_input(item_name, temp_dict, encoder, infer=False, **kwargs):
        '''
            process data in temporary_dicts
        '''

        def get_pitch(wav, mel):
            # get ground truth f0 by self.get_pitch_algorithm
            global f0_dict
            use_crepe = hparams['use_crepe'] if not infer else kwargs['use_crepe']
            if use_crepe:
                md5 = get_md5(wav)
                if infer and md5 in f0_dict.keys():
                    print("load temp crepe f0")
                    gt_f0 = np.array(f0_dict[md5]["f0"])
                    coarse_f0 = np.array(f0_dict[md5]["coarse"])
                else:
                    torch.cuda.is_available() and torch.cuda.empty_cache()
                    gt_f0, coarse_f0 = get_pitch_crepe(wav, mel, hparams, threshold=0.05)
                if infer:
                    f0_dict[md5] = {"f0": gt_f0.tolist(), "coarse": coarse_f0.tolist(), "time": int(time.time())}
                    write_temp("./diff_svc/infer_tools/f0_temp.json", f0_dict)
            else:
                gt_f0, coarse_f0 = get_pitch_parselmouth(wav, mel, hparams)
            if sum(gt_f0) == 0:
                raise BinarizationError("Empty **gt** f0")
            processed_input['f0'] = gt_f0
            processed_input['pitch'] = coarse_f0

        def get_align(mel, phone_encoded):
            mel2ph = np.zeros([mel.shape[0]], int)
            start_frame = 0
            ph_durs = mel.shape[0] / phone_encoded.shape[0]
            for i_ph in range(phone_encoded.shape[0]):
                end_frame = int(i_ph * ph_durs + ph_durs + 0.5)
                mel2ph[start_frame:end_frame + 1] = i_ph + 1
                start_frame = end_frame + 1

            processed_input['mel2ph'] = mel2ph

        wav, mel = nsf_hifigan.wav2spec(temp_dict['wav_fn'])
        processed_input = {
            'item_name': item_name, 'mel': mel,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]
        }
        processed_input = {**temp_dict, **processed_input,
                           'spec_min': np.min(mel, axis=0),
                           'spec_max': np.max(mel, axis=0)}  # merge two dicts
        try:
            get_pitch(wav, mel)
            try:
                hubert_encoded = processed_input['hubert'] = encoder.encode(temp_dict['wav_fn'])
            except:
                traceback.print_exc()
                raise Exception(f"hubert encode error")
            get_align(mel, hubert_encoded)
        except Exception as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {temp_dict['wav_fn']}")
            return None
        if hparams['use_energy_embed']:
            max_frames = hparams['max_frames']
            spec = torch.Tensor(processed_input['mel'])[:max_frames]
            processed_input['energy'] = (spec.exp() ** 2).sum(-1).sqrt()
        return processed_input

    @staticmethod
    def processed_input2batch(samples):
        '''
            Args:
                samples: one batch of processed_input
            NOTE:
                the batch size is controlled by hparams['max_sentences']
        '''
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        hubert = utils.collate_2d([s['hubert'] for s in samples], 0.0)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples])
        uv = utils.collate_1d([s['uv'] for s in samples])
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'hubert': hubert,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }
        if hparams['use_energy_embed']:
            batch['energy'] = utils.collate_1d([s['energy'] for s in samples], 0.0)
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch
