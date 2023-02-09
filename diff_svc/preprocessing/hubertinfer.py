import os.path
from io import BytesIO
from pathlib import Path

import numpy as np
# import onnxruntime as ort
import torch

from diff_svc.modules.hubert.cn_hubert import load_cn_model, get_cn_hubert_units
from diff_svc.modules.hubert.hubert_model import hubert_soft, get_units
from diff_svc.modules.hubert.hubert_onnx import get_onnx_units
from diff_svc.utils.hparams import hparams


class HubertEncoder:
    def __init__(self, pt_path='diff_svc/checkpoints/hubert/hubert_soft.pt', hubert_mode='', onnx=False):
        self.hubert_mode = hubert_mode
        self.onnx = onnx
        if 'use_cn_hubert' not in hparams.keys():
            hparams['use_cn_hubert'] = False
        if hparams['use_cn_hubert'] or self.hubert_mode == 'cn_hubert':
            pt_path = "checkpoints/cn_hubert/chinese-hubert-base-fairseq-ckpt.pt"
            self.dev = torch.device("cpu")
            self.hbt_model = load_cn_model(pt_path)
        else:
            # if onnx:
            #     self.hbt_model = ort.InferenceSession("onnx/hubert_soft.onnx",
            #                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider', ])
            # else:
                pt_path = list(Path(pt_path).parent.rglob('*.pt'))[0]
                if 'hubert_gpu' in hparams.keys():
                    self.use_gpu = hparams['hubert_gpu']
                else:
                    self.use_gpu = True
                self.dev = torch.device("cpu")
                self.hbt_model = hubert_soft(str(pt_path)).to(self.dev)
        print(f"| load 'model' from '{pt_path}'")

    def encode(self, wav_path):
        if isinstance(wav_path, BytesIO):
            npy_path = ""
            wav_path.seek(0)
        else:
            npy_path = Path(wav_path).with_suffix('.npy')
        if os.path.exists(npy_path):
            units = np.load(str(npy_path))
        elif self.onnx:
            units = get_onnx_units(self.hbt_model, wav_path).squeeze(0)
        elif hparams['use_cn_hubert'] or self.hubert_mode == 'cn_hubert':
            units = get_cn_hubert_units(self.hbt_model, wav_path, self.dev).cpu().numpy()[0]
        else:
            units = get_units(self.hbt_model, wav_path, self.dev).cpu().numpy()[0]
        return units  # [T,256]
