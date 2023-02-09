import librosa
import torch
import torch.nn as nn


def load_cn_model(ch_hubert_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from fairseq import checkpoint_utils
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ch_hubert_path],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    model.eval()
    return model


def get_cn_hubert_units(con_model, audio_path, dev):
    audio, sampling_rate = librosa.load(audio_path,res_type='polyphase')
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000,res_type='polyphase')

    feats = torch.from_numpy(audio).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.to(dev),
        "padding_mask": padding_mask.to(dev),
        "output_layer": 9,  # layer 9
    }
    with torch.no_grad():
        logits = con_model.extract_features(**inputs)
        feats = con_model.final_proj(logits[0])
    return feats
