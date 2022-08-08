from typing import Any, Tuple
from scipy.io.wavfile import write
from hifi_gan_model import Generator
from hparams import create_hparams
import torch
from distributed import apply_gradient_allreduce
from tacotron2_model import Tacotron2
from numpy import finfo
import numpy as np
import json
import os
from text import text_to_sequence
from argparse import ArgumentParser
import cutlet


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_tacotron2(hparams: Any, device: str) -> Any:
    model = Tacotron2(hparams).cpu(
    ) if device == "cpu" else Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    return model


def load_hifi_gan(filepath: str, device: str) -> Any:
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def initialization(tacotron2_checkpoint: str, hifi_gan_checkpoint: str, hifi_gan_config: str, device: str) -> Tuple[Any, Any]:
    print("load tacotron2 model")
    # load tacotron2 model
    tacotron2_model = load_tacotron2(create_hparams(), device=device)
    tacotron2_model.load_state_dict(torch.load(tacotron2_checkpoint, map_location=torch.device('cpu'))['state_dict'])
    if device == "cpu":
        tacotron2_model.cpu().eval()
    else:
        tacotron2_model.cuda().eval()

    # load hifi gan config
    with open(hifi_gan_config) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    # load hifi gan model
    print("load hifi-gan gen")
    hifi_gan_generator = Generator(h).to(device)
    hifi_gan_model = load_hifi_gan(hifi_gan_checkpoint, device)
    hifi_gan_generator.load_state_dict(hifi_gan_model['generator'])
    hifi_gan_generator.eval()
    hifi_gan_generator.remove_weight_norm()
    print("All mods loaded")
    return tacotron2_model, hifi_gan_generator


def text_to_speech(tacotron2_model: Any, hifi_gan_generator: Any, input_text: str, output: str, cleaners: str, device: str) -> str:
    cut = cutlet.Cutlet("kunrei")
    with torch.no_grad():
        for idx, text in enumerate(input_text, start=1):
            romaji = cut.romaji(text).replace(" "."")
            romaji = romaji if romaji.endswith(".") else romaji+"."
            sequence = np.array(text_to_sequence(romaji, [cleaners]))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long() if device == "cpu" else torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

            mel_outputs, *_ = tacotron2_model.inference(sequence)
            
            raw_audio = hifi_gan_generator(mel_outputs.float())
            audio = raw_audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16') if device == "cpu" else audio.cuda().numpy().astype('int16')
            output_file = os.path.join(output,f"output_{idx}.wav")
            write(output_file, h.sampling_rate, audio)
            print(f"Generated voice -> {output_file}")
    return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_cuda", type=bool,
                        default=False, help="是否使用gpu加速,默认为cpu")
    parser.add_argument("-tt2ck","--tacotron2_checkpoint",
                        type=str, help="tacotron2模型存放路径")
    parser.add_argument("-hgck","--hifi_gan_checkpoint",
                        type=str, help="hifi_gan模型存放路径")
    parser.add_argument("-hgc","--hifi_gan_config", type=str, help="hifi_gan配置文件存放路径")
    parser.add_argument("-i","--input_text", type=str, default=None, help="合成的文本")
    parser.add_argument("-f","--input_file", type=str, default=None, help="合成的文本文件")
    parser.add_argument("-o","--output", type=str, help="语音输出的路径")
    parser.add_argument("-p","--cleaners", choices=["no_cleaners", "basic_cleaners",
                        "transliteration_cleaners", "english_cleaners"], default=["no_cleaners"], help="文本的预处理模式")
    args = parser.parse_args()
    
    
    if not args.input_text  and  not  args.input_file:
        raise Exception("缺少输入")
    input_text = [args.input_text] if args.input_text is not None else  []
    if args.input_file is not None:
        with open(args.input_file) as f:
            input_text += f.read().split("\n")
    device = "cuda" if args.use_cuda else "cpu"
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    tacotron2_model, hifi_gan_gen = initialization(
        args.tacotron2_checkpoint, args.hifi_gan_checkpoint, args.hifi_gan_config, device)
    output = text_to_speech(
        tacotron2_model=tacotron2_model,
        hifi_gan_generator=hifi_gan_gen,
        input_text= input_text,
        output=args.output,
        cleaners=args.cleaners,
        device=device
    )
    print("wav file has been saved in", output)
