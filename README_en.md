# MoeTTS
Speech synthesis model repo for galgame characters based on Tacotron2 , Hifigan and VITS. The repo is also used to publish precompiled GUI.

**Notice ：** This project is only for AI study and hobby, because of the like of the characters to develop this project, no malicious purpose. If there is infringement, please submit issues, we will immediately remove the associated model.

[简体中文](README.md)   [English](README_en.md)

## About

Speech synthesis model repo for galgame characters based on Tacotron2, Hifigan and VITS. The repo is also used to publish precompiled GUI.

(Beta version is also used to try other features)

## Latest Updates

1.2.4：

>1. Add recent use history.
>2. Add infer finish notice
>3. Update diff-svc (code version:12-04)
>4. Packed Openvpi's nsf_hifigan weights. (https://openvpi.github.io/vocoders/)

1.2.2-beta:

> 1. fix: Some devices are missing dlls and cannot run.
> 2. update: support diff-svc.

GPU version is coming soon.(code only)

## User Agreement

On the basis of the open source agreement please also comply with the following rules.

1. (Important) Do not use this software, the pre-trained models provided by this repository or the speech synthesis results for direct commercial purposes (e.g. QQ bots with paid features, direct sales, commercial games, etc.) derivative work is not included.
2. Please comply with the user agreement of the original work and do not create content that will adversely affect the original work.
3. The pre-trained models and datasets provided by this repository are partially from the community, and all consequences caused by their use are borne by the users, not the authors and contributors of this repository.
4. The use of any content of this repository (including but not limited to code, compiled exe, pre-trained models) for original game development is prohibited.

Attitude of this project:

1. This project only encourages derivative work of the original work within reasonable limits, and is against any inappropriate behavior such as insulting the original work and related industries.

## Tutorial

### File directory format

1. If the model does not have a configuration file, you can place it in any directory, if the model has a configuration file, rename it to `config.json` and place it in the same directory as the model.

2. (Notes on TTS model) Since version 1.2.0, the text module and cleaners from the original project have been deprecated. So, you need to write the symbols used by your model to the moetts configuration file. (If you don't know how to do this step, you can refer to the pre-trained models given)

   Example (moetts.json):

   ```json
   {
   	"symbols":["_", ",", ".", "!", "?", "-", "A", "E", "I", "N", "O", "Q", "U", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "m", "n", "o", "p", "r", "s", "t", "u", "v", "w", "y", "z", "\u0283", "\u02a7", "\u2193", "\u2191", " "]
   }
   
   ```

   

### Inputs

Mostly you need to input phoneme (In Japanese, it stands for the Roman Tones), but the author of the models can decide other input methods. For example, the ATRI model(Tacotron2 version) supports Roman Tones without spaces only and the commas and the periods only.

### GUI

![User Interface](assets/tacotron2.png)

After opened the software, please choose your model's path and output path. At last, type the text need to be speaked, click on the "合成语音" button, wait for a while, then the software will output the audio to the /{output_path}/outpus.wav

Notes:

 1. Because of the model loading, when generate voice for the first time, it may take a long time. However, the same model won't be reloaded when generate for the next time and will directly generate the voice.
 2. If you change the model, the software will reload it when generate voice next time.
 3. If you have modified cleaners and symbols, you will need to restart the software for them to take effect.
 4. The software supports amd64 only, instead of i386.

About VITS

![vits](assets/vits.png)

1. VITS-Single and VITS-Multi are single speaker model and multi speaker model respectively.
2. the `原角色ID` in VITS-Multi is the ID of the speaker to be synthesized, you need to provide a number, the `目标角色ID` is used as the target speaker id for `voice conversion`.
3. Using `语音迁移`(voice conversion) requires a wav file with a sample rate of 22050.

## V 1.2.0 Update

1. ToolBox Update

   1. Add Chinese g2p tool.
   2. Change pyopenjtalk to build-in and fixed Unicode error.

   ![1.2.0 tool](assets/1.2.0_tool.png)

2. Settings Update

   1. Add batch processing mode
   2. Support use custom filename
   3. Support change VITS length-scale

   ![1.2.0 tool](assets/1.2.0_settings.png)

## Beta Version

This version may be unstable.

### diff-svc

![diff-svc gui](./assets/diff_svc.png)

Setting Instructions:

1. 升降半音(Transposition): int (Unit: semitone)
2. 启用Crepe(Enable Crepe)：Improved audio quality when enabled, but will take longer.
3. 加速倍率(Acceleration ratio)：Default is 20, higher values will infer faster, but may affect quality.
4. 待转换音频(Input audio)：wav or ogg file with vocals only.

## Online Demo

Integrated into [Huggingface Spaces ](https://huggingface.co/spaces) using  [Gradio](https://github.com/gradio-app/gradio). Try it out [![Hugging  Face  Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/skytnt/moe-tts)

## Model Downloads

### TTS models:


| Name&Download                                                | Type              | Info                                                         | Input format                                                 |
| ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ATRI](https://pan.baidu.com/s/1itDhrhzw6uZYxB2238BzTQ?pwd=0z3u) | Tacotron2+Hifigan | ATRI - Character's model of *My dear moments*                | Roman tone without spaces. For example: `tozendesu.koseinodesukara.`(You can also use the "jp g2p" tool in the toolbox to help convert text) |
| [ATRI-VITS](https://pan.baidu.com/s/1_vhOx50OE5R4bE02ZMe9GA?pwd=9jo4) | VITS Single       | ATRI - Character's model of *My dear moments*                | Japanese text converted by "jp g2p - 调形标注+替换ts" in the toolbox |
| [13 Galgame characters](https://pan.baidu.com/s/1anZ3eusmG8BVhrQQrvlkug?pwd=0i9r) | VITS Multi        | Speakers: 0:Takakura Anri, 1:Takakura Anzu, 2:Apeiria, 3:Kurashina Asuka, 4:ATRI, 5:ISLA, 6:Shindou Ayane, 7:Himeno Sena, 8:Komari Yui, 9:Miyohashi Koori, 10:Arisaka Mashiro, 11:Sirosaki Mieru, 12:Nikaidou Shinku | Japanese text converted by "jp g2p - 调形标注+替换ts" in the toolbox |
| [Mori](https://pan.baidu.com/s/1ekfczPolJRNjqVDNgjTJSA?pwd=rm1v) | VITS Single       | Mori - Character's model of *Fox Hime Zero*                  | Japanese text converted by "jp g2p - 普通转换" in the toolbox |

### Diff-svc models:

| Name&Download                                                | Notes                           | Trainer                               |
| ------------------------------------------------------------ | ------------------------------- | ------------------------------------- |
| [Himeno Sena](https://pan.baidu.com/s/1vc7lLpyAjUDCKI_PO5CR6w?pwd=wad5) | 24000Hz(infer only)             | luoyily                               |
| [Komari Yui](https://pan.baidu.com/s/1WwluFplMLjVD9ZeF6qdAxQ?pwd=i4yc) | 24000Hz(infer only)             | luoyily                               |
| [ATRI](https://pan.baidu.com/s/1-jc9DSQp_fOv-kdc_4bkyQ?pwd=3jm3) | 24000Hz(infer only)             | [RiceCake](https://github.com/gak123) |
| [Takakura Anzu](https://pan.baidu.com/s/1aBL3geIXJmb7dfEzi4AF5g?pwd=86aq) | 24000Hz(infer only)             | luoyily                               |
| [Yune](https://pan.baidu.com/s/18cG-DX38V8LrnFqy83Mzaw?pwd=riwm) | 24000Hz(infer only)             | luoyily                               |
| [Nishiki Asumi](https://pan.baidu.com/s/1ZNMn0hRu2MhNeLwQ9_VZ3Q?pwd=xe57) | 24000Hz(infer only)             | luoyily                               |
| [Sirosaki mieru](https://pan.baidu.com/s/1lIXXxiAKShxoLkaXOJCdsw?pwd=7c1v) | 24000Hz(infer only)             | luoyily                               |
| [Isla(Plastic Memories)](https://pan.baidu.com/s/14mQfBOAIllqqanP03c5w0g?pwd=z1im) | 24000Hz(infer only)             | luoyily                               |
| [Illya](https://pan.baidu.com/s/14v-XkNtp8pDqXXVjljYFSw?pwd=mvh1) | 24000Hz(infer only)             | luoyily                               |
| [ATRI_44100](https://pan.baidu.com/s/1KV_SiWUdBZjayyUioPhF3w?pwd=q515) | 44100Hz(infer only)             | [RiceCake](https://github.com/gak123) |
| [Niimi Sora](https://pan.baidu.com/s/1OYBMiHFPOm6fp-EV7JJWbA?pwd=iavq) | 44100Hz(infer only)             | [RiceCake](https://github.com/gak123) |
| Mori [BaiduNetdisk](https://pan.baidu.com/s/15FsM-NM6XXhGCb8MAzp05w?pwd=5mcg) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/mori) | 44100Hz(infer only) | luoyily                               |
| Kurashina Asuka [BaiduNetdisk](https://pan.baidu.com/s/1saGS166ty3JvcwPa_xwKNw?pwd=hrqx) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/asuka441) | 44100Hz(infer only+full weight) | luoyily                               |
| [Arihara Nanami](https://pan.baidu.com/s/1sgshGvHc9Y8IGmSrgWTJHg?pwd=zr76) | 44100Hz(infer only) | [RiceCake](https://github.com/gak123) |
| [Nikaidou Shinku](https://pan.baidu.com/s/1uBiAEr-YB4pi-W_cn3U3gg?pwd=yuf8) | 44100Hz(infer only) | [RiceCake](https://github.com/gak123) |
| Himeno Sena [BaiduNetdisk](https://pan.baidu.com/s/1DZlPNTErn91Si4FPkplbEQ?pwd=k603) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/sena441) | 44100Hz(infer only) | luoyily                               |
| Sirosaki Mieru [BaiduNetdisk](https://pan.baidu.com/s/1WJE5JY-rTBdPaId-nnY1Gg?pwd=kv72) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/mieru441) | 44100Hz(infer only) | luoyily |


Note : Most of the above models include only the necessary inference weights, and you may not be able to continue training them directly.

## FAQ

1. Q: Can this GUI process non-official Tacotron2 models?

   A：If the structure of the model and thinking method didn't changed, and the differences between the official and non-official is only the method when processing data, then it seems like to be mostly fine.

2. Q: Is there a command line version or HttpApi?

   A: Considering that CLI and HttpApi are prone to abuse, development has been temporarily postponed. (In version 1.0.0, **[ShiroDoMain](https://github.com/ShiroDoMain)** developed the CLI version, you can get it from the cli branch if you need it)

## Share models&Join the development!

Welcome if you want to share your models! It's not recommend to storage your models here in the Github because they may be very large. You can Pull Request with your own download address.

If you have any suggestions or discovered and bugs, please submit an issue.

## Credits

1. **[ShiroDoMain](https://github.com/ShiroDoMain)**: Developed the cli for version 1.0.0
2. **[menproject](https://github.com/menproject)**: Translation of the English README for version 1.0.0
3. **[CjangCjengh](https://github.com/CjangCjengh/)**: Provides compiled g2p tools and symbol files suitable for Japanese tonal annotation.
4. **[skytnt](https://huggingface.co/skytnt)**: hugging face online demo

## Reference

hifi-gan: https://github.com/jik876/hifi-gan

Tacotron2: https://github.com/NVIDIA/tacotron2

VITS:https://github.com/jaywalnut310/vits

diff-svc:https://github.com/prophesier/diff-svc

DiffSinger:https://github.com/MoonInTheRiver/DiffSinger

DiffSinger(openvpi):https://github.com/openvpi/DiffSinger

DiffSinger Community Vocoder Project: https://openvpi.github.io/vocoders/
