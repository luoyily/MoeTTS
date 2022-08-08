# MoeTTS
Speech synthesis model repo for galgame characters based on Tacotron2 and Hifigan. The repo is also used to publish precompiled GUI.

[简体中文](README.md)   [English](README_en.md)

## About

Speech synthesis model repo for galgame characters based on Tacotron2 and Hifigan. The repo is also used to publish precompiled GUI.

hifi-gan: https://github.com/jik876/hifi-gan

Tacotron2: https://github.com/NVIDIA/tacotron2

## Tutorial

![User Interface](assets/start.png)

After opened the software, please choose your model's path and output path. If the model's author issued one, you need to choose Cleaners. At last, type the text need to be speaked, click on the "合成语音" button, wait for a while, then the software will output the audio to the /{output_path}/outpus.wav

Notes:

 1. Because of the model loading, when generate voice for the first time, it may take a long time. However, the same model won't be reloaded when generate for the next time and will directly generate the voice.
 2. If you change the model, the software will reload it when generate voice next time.
 3. Hifigan's config.json need to be put in the same path of the model.
 4. The software supports amd64 only, instead of i386.

Example:

![Example](assets/example.png)

## Model Downloads

Submit models in this model:

x. Model's name

Description: Model of the character xxx

Input method: 

Download address:

Hifigan's Model download address: Optional

More info: Optional



1. ATRI

   Description:  ATRI - Character's model of *My dear moments* 

   Input method: Roman tone without spaces. Only English comma or period accepted. Choose basic_cleaners for the Cleaners option. For example: `tozendesu.koseinodesukara.`

   Download Address: https://pan.baidu.com/s/1hJIbIX0r1UpI3UEtsp-6EA?pwd=jdi4 Passcode: jdi4
   
   Hifigan's Model download address:https://pan.baidu.com/s/1PGU8XEs5wy4ppJL6GjTgMQ?pwd=24g8 Passcode: 24g8
   
   More info: Trained with 1300 in game voices. About 600 Epoch.

## FAQ

1. Q: Can this GUI process non-official Tacotron2 models?

   A：If the structure of the model and thinking method didn't changed, and the differences between the official and non-official is only the method when processing data, then it seems like to be mostly fine.

2. Q: What should I input for the text?

   A: Mostly you need to input phoneme (In Japanese, it stands for the Roman Tones), but the author of the models can decide other input methods. For exmaple, the ATRI model supports Roman Tones without spaces only and the commas and the periods only.

   Text will be directly sent to the basic_cleaners and be converted into a list for the machine inference. If your model's author trained the model with outher prepared data, please input those prepared data in the same way.

   ```python
   # A example for the data preparing
   sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
   sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()
   ```

   

## Share models&Join the development!

Welcome if you want to share your models! It's not recommend to storage your models here in the Github because they may be very large. You can Pull Request with your own download address.

If you have any suggestions or discovered and bugs, please submit an issue.
