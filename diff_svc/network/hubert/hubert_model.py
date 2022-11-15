import copy
import os
import random
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as t_func
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from utils import hparams


class Hubert(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        self.proj = nn.Linear(768, 256)

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
            self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode(x)
        x = self.proj(x)
        logits = self.logits(x)
        return logits, mask


class HubertSoft(Hubert):
    def __init__(self):
        super().__init__()

    # @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)

    def forward(self, wav: torch.Tensor):
        return self.units(wav)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = t_func.gelu(self.norm0(self.conv0(x)))
        x = t_func.gelu(self.conv1(x))
        x = t_func.gelu(self.conv2(x))
        x = t_func.gelu(self.conv3(x))
        x = t_func.gelu(self.conv4(x))
        x = t_func.gelu(self.conv5(x))
        x = t_func.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = t_func.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(
            self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
            self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


def _compute_mask(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        device: torch.device,
        min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask


def hubert_soft(
        path: str
) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hubert = HubertSoft()
    checkpoint = torch.load(path)
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hubert.load_state_dict(checkpoint)
    hubert.eval().to(dev)
    return hubert


def get_units(hbt_soft, raw_wav_path, dev=torch.device('cuda')):
    wav, sr = librosa.load(raw_wav_path, sr=None)
    assert (sr >= 16000)
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav)
    if sr != 16000:
        wav16 = librosa.resample(wav, sr, 16000)
    else:
        wav16 = wav
    dev = torch.device("cuda" if (dev == torch.device('cuda') and torch.cuda.is_available()) else "cpu")
    torch.cuda.is_available() and torch.cuda.empty_cache()
    with torch.inference_mode():
        units = hbt_soft.units(torch.FloatTensor(wav16.astype(float)).unsqueeze(0).unsqueeze(0).to(dev))
        return units


def get_end_file(dir_path, end):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_list.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_list


if __name__ == '__main__':
    from pathlib import Path

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hubert的模型路径
    hbt_model = hubert_soft(str(list(Path(hparams['hubert_path']).home().rglob('*.pt'))[0]))
    # 这个不用改，自动在根目录下所有wav的同文件夹生成其对应的npy
    file_lists = list(Path(hparams['raw_data_dir']).rglob('*.wav'))
    nums = len(file_lists)
    count = 0
    for wav_path in file_lists:
        npy_path = wav_path.with_suffix(".npy")
        npy_content = get_units(hbt_model, wav_path).cpu().numpy()[0]
        np.save(str(npy_path), npy_content)
        count += 1
        print(f"hubert process：{round(count * 100 / nums, 2)}%")
