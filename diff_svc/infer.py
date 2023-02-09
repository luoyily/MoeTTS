import io
from pathlib import Path

import numpy as np
import soundfile

from diff_svc.infer_tools import infer_tool
from diff_svc.infer_tools import slicer
from diff_svc.infer_tools.infer_tool import Svc
from diff_svc.utils.hparams import hparams


def run_clip(raw_audio_path, svc_model, key, acc, use_crepe, spk_id=0, auto_key=False, out_path=None, slice_db=-40,
             use_crepe_tiny=False,**kwargs):
    print(f'code version:2023-01-22')
    hparams['use_crepe_tiny'] = use_crepe_tiny
    infer_tool.format_wav(raw_audio_path)
    wav_path = str(Path(raw_audio_path).with_suffix('.wav'))
    
    key = svc_model.evaluate_key(wav_path, key, auto_key)
    chunks = slicer.cut(wav_path, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

    count = 0
    f0_tst, f0_pred, audio = [], [], []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        length = int(np.ceil(len(data) / audio_sr * hparams['audio_sample_rate']))
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        raw_path.seek(0)
        if slice_tag:
            print('jump empty segment')
            _f0_tst, _f0_pred, _audio = (
                np.zeros(int(np.ceil(length / hparams['hop_size']))),
                np.zeros(int(np.ceil(length / hparams['hop_size']))),
                np.zeros(length))
        else:
            _f0_tst, _f0_pred, _audio = svc_model.infer(raw_path, spk_id=spk_id, key=key, acc=acc, use_crepe=use_crepe)
        fix_audio = np.zeros(length)
        fix_audio[:] = np.mean(_audio)
        fix_audio[:len(_audio)] = _audio[0 if len(_audio) < len(fix_audio) else len(_audio) - len(fix_audio):]
        f0_tst.extend(_f0_tst)
        f0_pred.extend(_f0_pred)
        audio.extend(list(fix_audio))
        count += 1
    # if out_path is None:
    #     out_path = f'./output/123.{kwargs["format"]}'
    soundfile.write(out_path, audio, hparams["audio_sample_rate"], 'PCM_16', format=out_path.split('.')[-1])
    print(f'File Saved: {out_path}')
    return np.array(f0_tst), np.array(f0_pred), audio


if __name__ == '__main__':
    # 工程文件夹名，训练时用的那个
    project_name = "mieru441"
    model_path = f'./diff_svc/checkpoints/{project_name}/model_ckpt_steps_42000.ckpt'
    config_path = f'./diff_svc/checkpoints/{project_name}/config.yaml'

    # 支持多个wav/ogg文件，放在raw文件夹下，带扩展名
    # file_names = ["逍遥仙"]
    spk_id = 0
    # 自适应变调（仅支持单人模型）
    auto_key = False
    tran = 0  # 音高调整，支持正负（半音），数量与上一行对应，不足的自动按第一个移调参数补齐
    # 加速倍数
    accelerate = 50
    hubert_gpu = True
    wav_format = 'flac'
    step = int(model_path.split("_")[-1].split(".")[0])

    # 下面不动
    # infer_tool.mkdir(["./raw", "./results"])
    # infer_tool.fill_a_to_b(trans, file_names)

    model = Svc(project_name, config_path, hubert_gpu, model_path, onnx=False)
    # for f_name, tran in zip(file_names, trans):
    #     if "." not in f_name:
    #         f_name += ".wav"
    #     audio_path = f"./raw/{f_name}"
    run_clip(raw_audio_path='raw/想要变得可爱-dfn.wav', svc_model=model, key=tran, acc=accelerate, use_crepe=False,
                spk_id=spk_id, auto_key=auto_key, project_name=project_name, format=wav_format)
