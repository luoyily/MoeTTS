import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog

import os
import ctypes
import json

from scipy.io.wavfile import write

import g2p.jp.japanese_g2p as jp_g2p
import g2p.zh.chinese_g2p as zh_g2p

import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# win 10 notice
try:
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    is_notice_avaliable = True
except:
    is_notice_avaliable = False

# functions

is_init_model = False
# Settings
notice_at_fin = False
enable_batch_out = False
enable_custom_filename = False
vits_length_scale = 1


def notice():
    if notice_at_fin and is_notice_avaliable:
        try:
            toaster.show_toast(
                "MoeTTS", "您的推理任务已完成", icon_path="favicon.ico", duration=3, threaded=True)
        except:
            pass


def set_notice_at_fin():
    global notice_at_fin
    notice_at_fin = ~ notice_at_fin


def file_locate(entry_box, reset_model=False, file_types=None, locate_files=False):
    global is_init_model
    file = ''
    if file_types:
        if locate_files:
            file = '|'.join(filedialog.askopenfilenames(
                initialdir=os.getcwd(), filetypes=file_types))
        else:
            file = filedialog.askopenfilename(
                initialdir=os.getcwd(), filetypes=file_types)
    else:
        file = filedialog.askopenfilename(initialdir=os.getcwd())
    entry_box.delete(0, 'end')
    entry_box.insert(0, file)
    if reset_model:
        # 重新选择模型后，重置初始化状态
        is_init_model = False
    print(file)


def directory_locate(entry_box):
    dire = filedialog.askdirectory(initialdir=os.getcwd())
    entry_box.delete(0, 'end')
    entry_box.insert(0, dire)
    print(dire)


def save_file_locate(entry_box):
    file = filedialog.asksaveasfilename(
        initialdir=os.getcwd(), filetypes=[('wav 音频文件', '.wav')])
    entry_box.delete(0, 'end')
    entry_box.insert(0, file)
    print(file)


def save_locate(entry_box):
    if enable_custom_filename and not enable_batch_out:
        save_file_locate(entry_box)
    else:
        directory_locate(entry_box)


def set_vits_length_scale(value):
    global vits_length_scale

    vits_length_scale = float(value)
    print('设定语速：', vits_length_scale)


def load_moetts_cfg(filepath):
    """
    加载GUI所需的额外配置文件,如TTS模型符号表、说话人表等
    Returns:
        dict: config dict
    """
    try:
        print('Trying to load MoeTTS config...')
        moetts_cfg = json.load(
            open(('/'.join(filepath.split('/')[:-1]))+'/moetts.json', encoding='utf-8'))
        return moetts_cfg
    except:
        print('Failed to load MoeTTS config, use default configuration.')
        return False


# text to sequence
taco_default_symbols = ["_", "-", "!", "'", "(", ")", ",", ".", ":", ";", "?", " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "@AA", "@AA0", "@AA1", "@AA2", "@AE", "@AE0", "@AE1", "@AE2", "@AH", "@AH0", "@AH1", "@AH2", "@AO", "@AO0", "@AO1", "@AO2", "@AW", "@AW0",
                        "@AW1", "@AW2", "@AY", "@AY0", "@AY1", "@AY2", "@B", "@CH", "@D", "@DH", "@EH", "@EH0", "@EH1", "@EH2", "@ER", "@ER0", "@ER1", "@ER2", "@EY", "@EY0", "@EY1", "@EY2", "@F", "@G", "@HH", "@IH", "@IH0", "@IH1", "@IH2", "@IY", "@IY0", "@IY1", "@IY2", "@JH", "@K", "@L", "@M", "@N", "@NG", "@OW", "@OW0", "@OW1", "@OW2", "@OY", "@OY0", "@OY1", "@OY2", "@P", "@R", "@S", "@SH", "@T", "@TH", "@UH", "@UH0", "@UH1", "@UH2", "@UW", "@UW0", "@UW1", "@UW2", "@V", "@W", "@Y", "@Z", "@ZH"]
vits_default_symbols = ['_', ',', '.', '!', '?', '-', 'A', 'E', 'I', 'N', 'O', 'Q', 'U', 'a', 'b', 'd', 'e', 'f',
                        'g', 'h', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'ʃ', 'ʧ', '↓', '↑', ' ']


def text_to_sequence(text, symbols=None):
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    seq = [_symbol_to_id[c] for c in text if c in symbols]
    return seq


def inference_taco(taco_model_path, hifigan_model_path, target_text, output_path):
    def init_taco_model():
        global is_init_model, taco_model, hifigan_model, symbols, torch
        if not is_init_model:
            import torch
            from tacotron2.hparams import create_hparams
            from tacotron2.model import Tacotron2
            from hifigan.models import Generator
            # load symbols
            symbols = taco_default_symbols
            moetts_cfg = load_moetts_cfg(taco_model_path)
            if moetts_cfg:
                symbols = moetts_cfg['symbols']

            # create hparams and load tacotron2 model
            hparams = create_hparams()
            hparams.n_symbols = len(symbols)
            taco_model = Tacotron2(hparams).cpu()
            taco_model.load_state_dict(torch.load(
                taco_model_path, map_location='cpu')['state_dict'])
            taco_model.eval()

            # load hifigan model
            class AttrDict(dict):
                def __init__(self, *args, **kwargs):
                    super(AttrDict, self).__init__(*args, **kwargs)
                    self.__dict__ = self

            hifigan_cfg = '/'.join(hifigan_model_path.split('/')
                                   [:-1]) + '/config.json'
            h = AttrDict(json.load(open(hifigan_cfg)))
            torch.manual_seed(h.seed)

            hifigan_model = Generator(h).to('cpu')
            state_dict_g = torch.load(hifigan_model_path, map_location='cpu')
            hifigan_model.load_state_dict(state_dict_g['generator'])
            hifigan_model.eval()
            hifigan_model.remove_weight_norm()
            print('Tacotron2 & hifigan loaded.')
            is_init_model = True

    def synthesis():
        text_list = [target_text]
        if enable_batch_out:
            text_list = target_text.split('|')

        for n, text in enumerate(text_list):
            sequence = text_to_sequence(text, symbols)
            sequence = torch.autograd.Variable(
                torch.tensor(sequence).unsqueeze(0)).cpu().long()

            mel_outputs, mel_outputs_postnet, _, alignments = taco_model.inference(
                sequence)

            with torch.no_grad():
                raw_audio = hifigan_model(mel_outputs.float())
                audio = raw_audio.squeeze() * 32768.0
                audio = audio.cpu().numpy().astype('int16')
                sampling_rate = 22050
                if enable_custom_filename:
                    write(output_path+'.wav', sampling_rate, audio)
                else:
                    if enable_batch_out:
                        write(os.path.join(output_path,
                              f'output_{n}.wav'), sampling_rate, audio)
                    else:
                        write(os.path.join(output_path, 'output.wav'),
                              sampling_rate, audio)
                print(f'File saved to {output_path}')

    init_taco_model()
    synthesis()
    notice()
# vits


def get_text(text, hps, symbols):
    text_norm = text_to_sequence(text, symbols)
    if hps.data.add_blank:
        text_norm = vits_commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def inference_vits(vits_model_path, target_text, output, speaker_id, mode='synthesis', target_speaker_id=0, src_audio=None):
    """
    :params
    mode: synthesis, convert
    """
    def init_vits_model():
        global is_init_model, vits_hps, vits_model, symbols, vits_commons, torch
        global spectrogram_torch, load_wav_to_torch
        if not is_init_model:
            import torch
            import vits.commons as vits_commons
            import vits.utils
            from vits.models import SynthesizerTrn
            from vits.mel_processing import spectrogram_torch
            from vits.utils import load_wav_to_torch

            symbols = vits_default_symbols
            moetts_cfg = load_moetts_cfg(vits_model_path)
            if moetts_cfg:
                symbols = moetts_cfg['symbols']

            # load vits model
            vits_cfg = '/'.join(vits_model_path.split('/')[:-1])+'/config.json'
            vits_hps = vits.utils.get_hparams_from_file(vits_cfg)
            vits_model = SynthesizerTrn(
                len(symbols),
                vits_hps.data.filter_length // 2 + 1,
                vits_hps.train.segment_size // vits_hps.data.hop_length,
                n_speakers=vits_hps.data.n_speakers,
                **vits_hps.model).cpu()
            vits_model.eval()
            vits.utils.load_checkpoint(vits_model_path, vits_model, None)

            is_init_model = True

    def convert():
        src_speaker_id = torch.LongTensor([int(speaker_id)]).cpu()
        tgt_speaker_id = torch.LongTensor([int(target_speaker_id)]).cpu()
        audio, sampling_rate = load_wav_to_torch(src_audio)

        if sampling_rate != 22050:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(sampling_rate, 22050))
        # wav to spec
        audio_norm = audio / 32768
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm, 1024, 22050, 256, 1024, center=False).cpu()
        # spec = torch.squeeze(spec, 0)
        spec_length = torch.LongTensor([spec.data.shape[2]]).cpu()
        audio = vits_model.voice_conversion(spec, spec_length,
                                            sid_src=src_speaker_id, sid_tgt=tgt_speaker_id)[0][0, 0].data.cpu().float().numpy()

        audio = (audio * 32768.0).squeeze().astype('int16')
        if enable_custom_filename:
            write(output+'.wav', 22050, audio)
        else:
            write(os.path.join(output, 'output_convert.wav'), 22050, audio)

    def synthesis():
        text_list = [target_text]
        if enable_batch_out:
            text_list = target_text.split('|')
        for n, tar_text in enumerate(text_list):
            stn_tst = get_text(tar_text, vits_hps, symbols)
            with torch.no_grad():
                x_tst = stn_tst.cpu().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
                sid = 0 if vits_hps.data.n_speakers == 0 else torch.LongTensor(
                    [int(speaker_id)]).cpu()
                # :
                #     sid = 0
                audio = vits_model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                         length_scale=vits_length_scale)[0][0, 0].data.cpu().float().numpy()
                audio = (audio * 32768.0).squeeze().astype('int16')
                if enable_custom_filename:
                    write(output+'.wav', 22050, audio)
                else:
                    if enable_batch_out:
                        write(os.path.join(
                            output, f'output_vits_{n}.wav'), 22050, audio)
                    else:
                        write(os.path.join(output, 'output_vits.wav'), 22050, audio)
    init_vits_model()
    if mode == 'synthesis':
        synthesis()
    elif mode == 'convert':
        convert()
    print(f'File saved to {output}.')
    notice()


# Main window
root = tk.Tk()
root.geometry("760x450")
root.title('MoeTTS-CPU')
root.iconbitmap('./favicon.ico')
style = ttk.Style("minty")
# DPI adapt

ctypes.windll.shcore.SetProcessDpiAwareness(1)
ScaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)
root.tk.call('tk', 'scaling', ScaleFactor/80)
root.geometry(
    f"{int(760+760*0.5*(ScaleFactor/100-1))}x{int(450+450*0.5*(ScaleFactor/100-1))}")

nb = ttk.Notebook(root)
nb.pack(side=LEFT, padx=0, pady=(10, 0),  expand=YES, fill=BOTH)

# Tacotron2
taco_tab = ttk.Frame(nb)
taco_entry_ipadx = 175
# Tacotron2 model path select
taco_mp_label = ttk.Label(taco_tab, text='Tacotron2 模型：')
taco_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15, 5))


taco_mp = ttk.Entry(taco_tab)
taco_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=taco_entry_ipadx)

taco_mp_btn = ttk.Button(taco_tab,  text="浏览文件", bootstyle=(INFO, OUTLINE),
                         command=lambda: file_locate(taco_mp, reset_model=True))
taco_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# hifigan model path select
hg_mp_label = ttk.Label(taco_tab, text='HifiGAN 模型：')
hg_mp_label.grid(row=2, column=1, padx=(20, 5), pady=5)


hg_mp = ttk.Entry(taco_tab)
hg_mp.grid(row=2, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

hg_mp_btn = ttk.Button(taco_tab, text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(hg_mp, reset_model=True))
hg_mp_btn.grid(row=2, column=3, padx=5, pady=5)

# output path select
taco_out_label = ttk.Label(taco_tab, text='输出目录：')
taco_out_label.grid(row=3, column=1, padx=(20, 5), pady=5)


taco_out = ttk.Entry(taco_tab)
taco_out.grid(row=3, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

taco_out_btn = ttk.Button(taco_tab, text="浏览目录", bootstyle=(
    INFO, OUTLINE), command=lambda: save_locate(taco_out))
taco_out_btn.grid(row=3, column=3, padx=5, pady=5)

# text input
taco_text_label = ttk.Label(taco_tab, text='待合成文本：')
taco_text_label.grid(row=4, column=1, padx=(20, 5), pady=5)


taco_target_text = ttk.Entry(taco_tab)
taco_target_text.grid(row=4, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)
taco_text_btn = ttk.Button(taco_tab, text="合成语音", bootstyle=(SECONDARY, OUTLINE),
                           command=lambda: inference_taco(taco_mp.get(), hg_mp.get(), taco_target_text.get(), taco_out.get()))
taco_text_btn.grid(row=4, column=3, padx=5, pady=5)

# VITS
vits_tab = ttk.Frame(nb)
# VITS model path select
vits_mp_label = ttk.Label(vits_tab, text='VITS 模型：')
vits_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15, 5))

vits_mp = ttk.Entry(vits_tab)
vits_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=taco_entry_ipadx)

vits_mp_btn = ttk.Button(vits_tab,  text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(vits_mp, reset_model=True))
vits_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# output path select
vits_out_label = ttk.Label(vits_tab, text='输出目录：')
vits_out_label.grid(row=2, column=1, padx=(20, 5), pady=5)


vits_out = ttk.Entry(vits_tab)
vits_out.grid(row=2, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vits_out_btn = ttk.Button(vits_tab, text="浏览目录", bootstyle=(
    INFO, OUTLINE), command=lambda: save_locate(vits_out))
vits_out_btn.grid(row=2, column=3, padx=5, pady=5)
# speaker select
src_speaker_label = ttk.Label(vits_tab, text='原角色ID：')
src_speaker_label.grid(row=3, column=1, padx=(20, 5), pady=5)

src_speaker = ttk.Entry(vits_tab)
src_speaker.grid(row=3, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

# text input
vits_text_label = ttk.Label(vits_tab, text='待合成文本：')
vits_text_label.grid(row=4, column=1, padx=(20, 5), pady=5)


vits_target_text = ttk.Entry(vits_tab)
vits_target_text.grid(row=4, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vits_text_btn = ttk.Button(vits_tab, text="合成语音", bootstyle=(SECONDARY, OUTLINE),
                           command=lambda: inference_vits(vits_mp.get(), vits_target_text.get(), vits_out.get(), src_speaker.get()))
vits_text_btn.grid(row=4, column=3, padx=5, pady=5)

# audio source
vits_src_audio_label = ttk.Label(vits_tab, text='待迁移音频：')
vits_src_audio_label.grid(row=5, column=1, padx=(20, 5), pady=5)


vits_src_audio = ttk.Entry(vits_tab)
vits_src_audio.grid(row=5, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vits_src_audio_btn = ttk.Button(vits_tab, text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(vits_src_audio))
vits_src_audio_btn.grid(row=5, column=3, padx=5, pady=5)

# voice conversion
vits_tgt_speaker_label = ttk.Label(vits_tab, text=' 目标角色ID：')
vits_tgt_speaker_label.grid(row=6, column=1, padx=(20, 5), pady=5)


vits_tgt_speaker = ttk.Entry(vits_tab)
vits_tgt_speaker.grid(row=6, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vits_tgt_audio_btn = ttk.Button(vits_tab, text="语音迁移", bootstyle=(SECONDARY, OUTLINE),
                                command=lambda: inference_vits(vits_mp.get(), vits_target_text.get(), vits_out.get(), src_speaker.get(),
                                                               mode='convert', target_speaker_id=vits_tgt_speaker.get(), src_audio=vits_src_audio.get()))
vits_tgt_audio_btn.grid(row=6, column=3, padx=5, pady=5)

jtalk_mode = 'rsa'
replace_ts = False

# tool box


def jtalk_g2p(text, entry_box):
    entry_box.delete(0, 'end')
    root.update()
    res = ''
    if jtalk_mode == 'r':
        res = jp_g2p.get_romaji(text)
    elif jtalk_mode == 'rs':
        res = jp_g2p.get_romaji_with_space(text)
    elif jtalk_mode == 'rsa':
        res = jp_g2p.get_romaji_with_space_and_accent(text)
    if replace_ts:
        res = res.replace('ʦ', 'ts')
    entry_box.delete(0, 'end')
    entry_box.insert(0, res)


def set_jtalk_mode(mode):
    global jtalk_mode
    jtalk_mode = mode


def set_replace_ts():
    global replace_ts
    replace_ts = ~replace_ts


tool_tab = ttk.Frame(nb)
# Japanese g2p
jtalk_group = ttk.Labelframe(
    master=tool_tab, text="OpenJtalk g2p (base on CjangCjengh's jp_g2p tool):", padding=15)
jtalk_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)
jtext_label = ttk.Label(jtalk_group, text='日语文本：')
jtext_label.grid(row=1, column=1, padx=(20, 5), pady=5)

jtext = ttk.Entry(jtalk_group)
jtext.grid(row=1, column=2, padx=20, pady=5, ipadx=170)

jtext_btn = ttk.Button(jtalk_group,  text="转换(g2p)", bootstyle=(SECONDARY, OUTLINE),
                       command=lambda: jtalk_g2p(jtext.get(), jtext))
jtext_btn.grid(row=1, column=3, padx=5, pady=5)

text_label = ttk.Label(jtalk_group, text='转换模式:')
text_label.place(x=20, y=50)
j_radio_var = ttk.IntVar()
radio1 = ttk.Radiobutton(jtalk_group, text="普通转换", variable=j_radio_var,
                         value=1, command=lambda: set_jtalk_mode('r'))
radio1.place(x=150, y=50)
radio2 = ttk.Radiobutton(jtalk_group, text="空格分词", variable=j_radio_var,
                         value=2, command=lambda: set_jtalk_mode('rs'))
radio2.place(x=260, y=50)
radio3 = ttk.Radiobutton(jtalk_group, text="分词+调形", variable=j_radio_var,
                         value=3, command=lambda: set_jtalk_mode('rsa'))
radio3.place(x=370, y=50)
jtalk_check1 = ttk.Checkbutton(
    jtalk_group, text='替换ʦ到ts', command=lambda: set_replace_ts())
jtalk_check1.place(x=480, y=50)

# Chinese g2p
pinyin_mode = 'normal'
pinyin_space = False


def py_g2p(text, entry_box):
    entry_box.delete(0, 'end')
    root.update()
    res = zh_g2p.g2p(text, pinyin_mode, pinyin_space)
    entry_box.delete(0, 'end')
    entry_box.insert(0, res)


def set_pinyin_mode(mode):
    global pinyin_mode
    pinyin_mode = mode


def set_pinyin_space():
    global pinyin_space
    pinyin_space = ~pinyin_space


pinyin_group = ttk.Labelframe(
    master=tool_tab, text="pypinyin g2p:", padding=15)
pinyin_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)
pinyin_label = ttk.Label(pinyin_group, text='中文文本：')
pinyin_label.grid(row=1, column=1, padx=(20, 5), pady=5)


pinyin_text = ttk.Entry(pinyin_group)
pinyin_text.grid(row=1, column=2, padx=20, pady=5, ipadx=170)

pinyin_text_btn = ttk.Button(pinyin_group,  text="转换(g2p)", bootstyle=(SECONDARY, OUTLINE),
                             command=lambda: py_g2p(pinyin_text.get(), pinyin_text))
pinyin_text_btn.grid(row=1, column=3, padx=5, pady=5)

pinyin_text_label = ttk.Label(pinyin_group, text='转换模式:')
pinyin_text_label.place(x=20, y=50)
pinyin_radio_var = ttk.IntVar()
pinyin_radio1 = ttk.Radiobutton(pinyin_group, text="普通转换", variable=pinyin_radio_var,
                                value=4, command=lambda: set_pinyin_mode('normal'))
pinyin_radio1.place(x=150, y=50)
pinyin_radio2 = ttk.Radiobutton(pinyin_group, text="数字声调", variable=pinyin_radio_var,
                                value=5, command=lambda: set_pinyin_mode('tone3'))
pinyin_radio2.place(x=260, y=50)
pinyin_radio3 = ttk.Radiobutton(pinyin_group, text="注音符号", variable=pinyin_radio_var,
                                value=6, command=lambda: set_pinyin_mode('bopomofo'))
pinyin_radio3.place(x=370, y=50)
pinyin_check1 = ttk.Checkbutton(
    pinyin_group, text='是否空格', command=lambda: set_pinyin_space())
pinyin_check1.place(x=480, y=50)

# Audio tools


def audio_convert(audio_path, out_path):
    os.system(f'ffmpeg -i {audio_path} -ac 1 -ar 22050 {out_path}')


audio_group = ttk.Labelframe(
    master=tool_tab, text="音频转换(到22050 采样率,单声道wav)(依赖ffmepg):", padding=15)
audio_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)

audio_label = ttk.Label(audio_group, text='待转换音频：')
audio_label.grid(row=1, column=1, padx=(20, 5), pady=5)


audio_path = ttk.Entry(audio_group)
audio_path.grid(row=1, column=2, padx=20, pady=5, ipadx=170)

audio_sel_btn = ttk.Button(audio_group,  text="浏览文件", bootstyle=(INFO, OUTLINE),
                           command=lambda: file_locate(audio_path))
audio_sel_btn.grid(row=1, column=3, padx=5, pady=5)

audio_label2 = ttk.Label(audio_group, text='输出位置：')
audio_label2.grid(row=2, column=1, padx=(20, 5), pady=5)


audio_out = ttk.Entry(audio_group)
audio_out.grid(row=2, column=2, padx=20, pady=5, ipadx=170)

audio_out_btn = ttk.Button(audio_group,  text="浏览目录", bootstyle=(INFO, OUTLINE),
                           command=lambda: save_file_locate(audio_out))
audio_out_btn.grid(row=2, column=3, padx=5, pady=5)

audio_out_btn2 = ttk.Button(audio_group,  text="   转 换   ", bootstyle=(SECONDARY, OUTLINE),
                            command=lambda: audio_convert(audio_path.get(), str(audio_out.get())+'.wav'))
audio_out_btn2.grid(row=3, column=3, padx=5, pady=5)

# settings tab


def set_batch_mode():
    global enable_batch_out
    enable_batch_out = ~enable_batch_out


def set_custom_filename():
    global enable_custom_filename
    enable_custom_filename = ~enable_custom_filename


setting_tab = ttk.Frame(nb)
file_setting_group = ttk.Labelframe(
    master=setting_tab, text="文件设置:", padding=15)
file_setting_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=40)

file_set_check1 = ttk.Checkbutton(
    file_setting_group, text='启用批量合成模式', command=lambda: set_batch_mode())
file_set_check1.grid(row=1, column=1, padx=5, pady=5)
file_set_check2 = ttk.Checkbutton(
    file_setting_group, text='启用自定义文件名', command=lambda: set_custom_filename())
file_set_check2.grid(row=1, column=2, padx=5, pady=5)
file_set_help = ttk.Label(file_setting_group, text='说明：启用批量合成模式后，文本框中以"|"分隔句子后，会批量转换每句到文件夹下以单独文件存放。\
    \n批量转换模式无法自定义文件名。')
file_set_help.place(x=20, y=50)
# Notice setting
notice_set_check = ttk.Checkbutton(
    file_setting_group, text='启用完成通知', command=lambda: set_notice_at_fin())
notice_set_check.grid(row=1, column=3, padx=5, pady=5)

# VITS inference settings
vits_setting_group = ttk.Labelframe(
    master=setting_tab, text="VITS 语速设定(浮点数, 越大越慢):", padding=15)
vits_setting_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)

vits_ls_label = ttk.Label(vits_setting_group, text='语速 (默认1.0)：')
vits_ls_label.grid(row=1, column=1, padx=(20, 5), pady=5)
# vits length scale
vits_ls = ttk.Entry(vits_setting_group)
vits_ls.grid(row=1, column=2, padx=(20, 5), pady=5)
vits_ls_btn = ttk.Button(vits_setting_group,  text="设定", bootstyle=(INFO, OUTLINE),
                         command=lambda: set_vits_length_scale(vits_ls.get()))
vits_ls_btn.grid(row=1, column=3, padx=5, pady=5)

# ttk Theme
theme_group = ttk.Labelframe(master=setting_tab, text="外观设定:", padding=15)
theme_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)

style = ttk.Style()
style.configure('Horizontal.TScrollbar')
theme_names = style.theme_names()
theme_lb = ttk.Label(theme_group, text="选择主题:")
theme_lb.pack(padx=(20, 5), side=LEFT)
theme_cbo = ttk.Combobox(
    master=theme_group, text=style.theme.name, values=theme_names)
theme_cbo.pack(padx=(20, 5), side=LEFT)
theme_cbo.current(theme_names.index(style.theme.name))


def change_theme(e):
    style.theme_use(theme_cbo.get())
    theme_cbo.selection_clear()


theme_cbo.bind("<<ComboboxSelected>>", change_theme)

# Diff svc
# diff svc functions
use_crepe = False
use_pe = False
use_crepe_tiny = False

def set_crepe():
    global use_crepe
    use_crepe = ~use_crepe

def set_crepe_tiny():
    global use_crepe_tiny
    use_crepe_tiny = ~use_crepe_tiny


def set_pe():
    global use_pe
    use_pe = ~use_pe

def diff_svc_infer(model_path, input_files: list, out_path, tran=0, acc=20):
    global is_init_model, diff_svc_model,run_clip
    diff_svc_cfg = '/'.join(model_path.split('/')[:-1])+'/config.yaml'

    if not is_init_model:
        from diff_svc.infer_tools.infer_tool import Svc
        from diff_svc.infer import run_clip
        diff_svc_model = Svc('', diff_svc_cfg, False, model_path)
        is_init_model = True

    for i, input_file in enumerate(input_files):
        if not enable_custom_filename:
            final_out_path = os.path.join(
                out_path, f'{input_file.split("/")[-1][:-4]}_key_{tran}_{i}.wav')
        else:
            final_out_path = out_path
        # 是否使用 crepe tiny模型，通过run_clip传入hparams,最后get pitch crepe读取配置
        run_clip(diff_svc_model, key=tran, acc=acc, use_crepe=use_crepe, thre=0.05, use_pe=use_pe, use_gt_mel=False,
                 add_noise_step=500, f_name=input_file, out_path=final_out_path,use_crepe_tiny=use_crepe_tiny)
    notice()


# diff svc gui
diff_svc_tab = ttk.Frame(nb)

dsvc_mp_label = ttk.Label(diff_svc_tab, text='Diff-SVC模型：')
dsvc_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15, 5))


dsvc_mp = ttk.Entry(diff_svc_tab)
dsvc_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=taco_entry_ipadx)

dsvc_mp_btn = ttk.Button(diff_svc_tab,  text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(dsvc_mp, reset_model=True))
dsvc_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# output path select
dsvc_out_label = ttk.Label(diff_svc_tab, text='输出目录：')
dsvc_out_label.grid(row=2, column=1, padx=(20, 5), pady=5)


dsvc_out = ttk.Entry(diff_svc_tab)
dsvc_out.grid(row=2, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

dsvc_out_btn = ttk.Button(diff_svc_tab, text="浏览目录", bootstyle=(
    INFO, OUTLINE), command=lambda: save_locate(dsvc_out))
dsvc_out_btn.grid(row=2, column=3, padx=5, pady=5)

# raw audio select
dsvc_input_label = ttk.Label(diff_svc_tab, text='待转换音频：')
dsvc_input_label.grid(row=3, column=1, padx=(20, 5), pady=5)

dsvc_input = ttk.Entry(diff_svc_tab)
dsvc_input.grid(row=3, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

# 点炒饭禁止！
dsvc_input_btn = ttk.Button(diff_svc_tab, text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda: file_locate(
    dsvc_input, file_types=(("wav files", "*.wav"), ("ogg files", "*.ogg"),), locate_files=enable_batch_out))
dsvc_input_btn.grid(row=3, column=3, padx=5, pady=5)

# 变调，Crepe，PE

dsvc_pitch_label = ttk.Label(diff_svc_tab, text='升降半音|支持正负整数:')
dsvc_pitch_label.grid(row=4, column=1, padx=(5, 5), pady=5)


dsvc_pitch = ttk.Entry(diff_svc_tab, width=10)
dsvc_pitch.grid(row=4, column=2, padx=20, pady=5, ipadx=0, sticky='w')
dsvc_pitch.insert(index=0, string="0")

dsvc_crepe_check = ttk.Checkbutton(
    diff_svc_tab, text='启用Crepe[耗时较长|1:8]', command=lambda: set_crepe())
dsvc_crepe_check.place(x=350, y=155)

dsvc_pe_check = ttk.Checkbutton(
    diff_svc_tab, text='启用pe', command=lambda: set_pe())
dsvc_pe_check.place(x=570, y=155)

# 加速
dsvc_acc_label = ttk.Label(diff_svc_tab, text='加速倍率[推荐20]：')
dsvc_acc_label.grid(row=5, column=1, padx=(5, 5), pady=5)

dsvc_acc = ttk.Entry(diff_svc_tab, width=10)
dsvc_acc.grid(row=5, column=2, padx=20, pady=5, ipadx=0, sticky='w')
dsvc_acc.insert(0, '20')

# Crepe Tiny
dsvc_crepe_tiny_check = ttk.Checkbutton(
    diff_svc_tab, text='Crepe轻量模式[耗时更短|4:1]', command=lambda: set_crepe_tiny())
dsvc_crepe_tiny_check.place(x=350, y=200)

dsvc_input_btn = ttk.Button(diff_svc_tab, text="转换音频", bootstyle=(SECONDARY, OUTLINE),
                            command=lambda: diff_svc_infer(model_path=dsvc_mp.get(), input_files=dsvc_input.get().split("|"),
                                                           tran=int(dsvc_pitch.get()), out_path=dsvc_out.get(), acc=int(dsvc_acc.get())))
dsvc_input_btn.grid(row=4, column=3, padx=5, pady=5)

nb.add(taco_tab, text="Tacotron2", sticky=NW)
nb.add(vits_tab, text="VITS", sticky=NW)
nb.add(diff_svc_tab, text="Diff-svc", sticky=NW)
nb.add(tool_tab, text="ToolBox", sticky=NW)
nb.add(setting_tab, text="Settings", sticky=NW)

# save recent use


def save_recent_use():
    try:
        f = open('recent.json', mode='w', encoding='utf-8')
        entry_boxes = {
            "taco_mp": taco_mp.get(),
            "hg_mp": hg_mp.get(),
            "taco_out": taco_out.get(),
            "vits_mp": vits_mp.get(),
            "vits_out": vits_out.get(),
            "dsvc_mp": dsvc_mp.get(),
            "dsvc_out": dsvc_out.get()
        }
        json.dump(entry_boxes, f)
    except:
        pass
    root.destroy()


def load_recent_use():
    f = open('recent.json', mode='r', encoding='utf-8')
    entry_boxes = json.load(f)
    taco_mp.insert(0, entry_boxes['taco_mp'])
    hg_mp.insert(0, entry_boxes['hg_mp'])
    taco_out.insert(0, entry_boxes['taco_out'])
    vits_mp.insert(0, entry_boxes['vits_mp'])
    vits_out.insert(0, entry_boxes['vits_out'])
    dsvc_mp.insert(0, entry_boxes['dsvc_mp'])
    dsvc_out.insert(0, entry_boxes['dsvc_out'])


try:
    load_recent_use()
except:
    pass
root.resizable(0, 0)
root.protocol('WM_DELETE_WINDOW', save_recent_use)
root.mainloop()
