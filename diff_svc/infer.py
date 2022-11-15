import io
import time
from pathlib import Path

import numpy as np
import soundfile
import librosa

from infer_tools import infer_tool
from infer_tools import slicer
from infer_tools.infer_tool import Svc
from utils.hparams import hparams


chunks_dict = infer_tool.read_temp("./diff_svc/infer_tools/chunks_temp.json")


def run_clip(svc_model, key, acc, use_pe, use_crepe, thre, use_gt_mel, add_noise_step,f_name=None,out_path=None,audio_rate=24000):
    # 推理前准备（加载缓存，处理音频等）
    use_pe = use_pe if hparams['audio_sample_rate'] == 24000 else False
    # 转格式
    infer_tool.format_wav(f_name)
    wav_path = Path(f_name).with_suffix('.wav')
    # 检查缓存
    global chunks_dict
    audio, sr = librosa.load(wav_path, mono=True)
    wav_hash = infer_tool.get_md5(audio)
    if wav_hash in chunks_dict.keys():
        print("load chunks from temp")
        chunks = chunks_dict[wav_hash]["chunks"]
    else:
        chunks = slicer.cut(wav_path)
    chunks_dict[wav_hash] = {"chunks": chunks, "time": int(time.time())}
    infer_tool.write_temp("./diff_svc/infer_tools/chunks_temp.json", chunks_dict)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
    # audio_data, audio_sr = slicer.cut(Path(f_name).with_suffix('.wav'))
    
    count = 0
    f0_tst = []
    f0_pred = []
    audio = []
    epsilon = 0.0002
    for data in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        length = int(np.ceil(len(data) / audio_sr * hparams['audio_sample_rate']))
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        if hparams['debug']:
            print(np.mean(data), np.var(data))
        raw_path.seek(0)
        if np.var(data) < epsilon:
            print('jump empty segment')
            _f0_tst, _f0_pred, _audio = (
                np.zeros(int(np.ceil(length / hparams['hop_size']))), np.zeros(int(np.ceil(length / hparams['hop_size']))),
                np.zeros(length))
        else:
            _f0_tst, _f0_pred, _audio = svc_model.infer(raw_path, key=key, acc=acc, use_pe=use_pe, use_crepe=use_crepe,
                                                        thre=thre, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step)
        fix_audio = np.zeros(length)
        fix_audio[:] = np.mean(_audio)
        fix_audio[:len(_audio)] = _audio
        f0_tst.extend(_f0_tst)
        f0_pred.extend(_f0_pred)
        audio.extend(list(fix_audio))
        count += 1

    soundfile.write(out_path, audio, audio_rate, 'PCM_16')
    print(f'File Saved: {out_path}')
    return np.array(f0_tst), np.array(f0_pred), audio


if __name__ == '__main__':
    # 工程文件夹名，训练时用的那个
    project_name = "sena"
    model_path = f'./ckpt/{project_name}/model_ckpt_steps_80000.ckpt'
    config_path = f'./ckpt/{project_name}/config.yaml'

    # 支持多个wav/ogg文件，放在raw文件夹下，带扩展名
    file_names = ["4.wav"]
    trans = [0]  # 音高调整，支持正负（半音），数量与上一行对应，不足的自动按第一个移调参数补齐
    # 加速倍数
    accelerate = 20
    hubert_gpu = False
    cut_time = 30

    # 
    infer_tool.mkdir(["./raw", "./results"])
    infer_tool.fill_a_to_b(trans, file_names)

    model = Svc(project_name, config_path, hubert_gpu, model_path)
    for f_name, tran in zip(file_names, trans):
        run_clip(model, key=tran, acc=accelerate, use_crepe=True, thre=0.05, use_pe=True, use_gt_mel=False,
                 add_noise_step=500, f_name=f_name, project_name=project_name)
