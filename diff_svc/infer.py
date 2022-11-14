import io
import os
from pathlib import Path

import numpy as np
import soundfile

from infer_tools import infer_tool
from infer_tools import slicer
from infer_tools.infer_tool import Svc


def run_clip(svc_model, key, acc, use_pe, use_crepe, thre, use_gt_mel, add_noise_step,f_name=None,out_path=None,audio_rate=24000):
    infer_tool.format_wav(f_name)
    audio_data, audio_sr = slicer.cut(Path(f_name).with_suffix('.wav'))
    count = 0
    f0_tst = []
    f0_pred = []
    audio = []
    for data in audio_data:
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        raw_path.seek(0)
        _f0_tst, _f0_pred, _audio = svc_model.infer(raw_path, key=key, acc=acc, use_pe=use_pe, use_crepe=use_crepe,
                                                    thre=thre, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step)
        f0_tst.extend(_f0_tst)
        f0_pred.extend(_f0_pred)
        audio.extend(list(_audio))
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
