import os
from functools import reduce

import numpy as np

from diff_svc.modules.vocoders.nsf_hifigan import NsfHifiGAN
from diff_svc.preprocessing.process_pipeline import get_pitch_parselmouth, get_pitch_crepe
from diff_svc.utils.hparams import hparams

head_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def compare_pitch(f0_static_dict, pitch_time_temp, trans_key=0):
    return sum({k: v * f0_static_dict[str(k + trans_key)] for k, v in pitch_time_temp.items() if
                str(k + trans_key) in f0_static_dict}.values())


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return round(f0_pitch, 0)


def pitch_to_name(pitch):
    return f"{head_list[int(pitch % 12)]}{int(pitch / 12) - 1}"


def get_f0(audio_path, crepe=False):
    wav, mel = NsfHifiGAN.wav2spec(audio_path)
    if crepe:
        f0, pitch_coarse = get_pitch_crepe(wav, mel, hparams)
    else:
        f0, pitch_coarse = get_pitch_parselmouth(wav, mel, hparams)
    return f0


def merge_f0_dict(dict_list):
    def sum_dict(a, b):
        temp = dict()
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp

    return reduce(sum_dict, dict_list)


def collect_f0(f0):
    pitch_num = {}
    pitch_list = [f0_to_pitch(x) for x in f0[f0 > 0]]
    for key in pitch_list:
        pitch_num[key] = pitch_num.get(key, 0) + 1
    return pitch_num


def static_f0_time(f0):
    if isinstance(f0, dict):
        pitch_num = merge_f0_dict({k: collect_f0(v) for k, v in f0.items()}.values())
    else:
        pitch_num = collect_f0(f0)
    static_pitch_time = {}
    sort_key = sorted(pitch_num.keys())
    for key in sort_key:
        static_pitch_time[key] = round(pitch_num[key] * hparams['hop_size'] / hparams['audio_sample_rate'], 2)
    return static_pitch_time


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists

