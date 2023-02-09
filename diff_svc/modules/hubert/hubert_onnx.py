import time

import torch
import torchaudio


def get_onnx_units(hbt_soft, raw_wav_path):
    source, sr = torchaudio.load(raw_wav_path)
    source = torchaudio.functional.resample(source, sr, 16000)
    if len(source.shape) == 2 and source.shape[1] >= 2:
        source = torch.mean(source, dim=0).unsqueeze(0)
    source = source.unsqueeze(0)
    # 使用ONNX Runtime进行推理
    start = time.time()
    units = hbt_soft.run(output_names=["units"],
                         input_feed={"wav": source.numpy()})[0]
    use_time = time.time() - start
    print("hubert_onnx_session.run time:{}".format(use_time))
    return units
