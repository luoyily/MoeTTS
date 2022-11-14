import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog

import os
import sys

import ctypes
import json

from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.io.wavfile import write

import g2p.jp.japanese_g2p as jp_g2p
import g2p.zh.chinese_g2p as zh_g2p



import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# functions

is_init_model= False
# Settings
enable_batch_out = False
enable_custom_filename = False
vits_length_scale = 1

def file_locate(entry_box, reset_model=False,file_types=None):
    global is_init_model
    file = ''
    if file_types:
        file = filedialog.askopenfilename(initialdir=os.getcwd(),filetypes=file_types)
    else:
        file = filedialog.askopenfilename(initialdir=os.getcwd())
    entry_box.delete(0, 'end')
    entry_box.insert(0, file)
    if reset_model:
        # 重新选择模型后，重置初始化状态
        is_init_model= False
    print(file)

def directory_locate(entry_box):
    dire = filedialog.askdirectory(initialdir=os.getcwd())
    entry_box.delete(0, 'end')
    entry_box.insert(0, dire)
    print(dire)

def save_file_locate(entry_box):
    file = filedialog.asksaveasfilename(initialdir=os.getcwd(), filetypes=[('wav 音频文件', '.wav')])
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

# text to sequence
taco_default_symbols = ["_", "-", "!", "'", "(", ")", ",", ".", ":", ";", "?", " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "@AA", "@AA0", "@AA1", "@AA2", "@AE", "@AE0", "@AE1", "@AE2", "@AH", "@AH0", "@AH1", "@AH2", "@AO", "@AO0", "@AO1", "@AO2", "@AW", "@AW0", "@AW1", "@AW2", "@AY", "@AY0", "@AY1", "@AY2", "@B", "@CH", "@D", "@DH", "@EH", "@EH0", "@EH1", "@EH2", "@ER", "@ER0", "@ER1", "@ER2", "@EY", "@EY0", "@EY1", "@EY2", "@F", "@G", "@HH", "@IH", "@IH0", "@IH1", "@IH2", "@IY", "@IY0", "@IY1", "@IY2", "@JH", "@K", "@L", "@M", "@N", "@NG", "@OW", "@OW0", "@OW1", "@OW2", "@OY", "@OY0", "@OY1", "@OY2", "@P", "@R", "@S", "@SH", "@T", "@TH", "@UH", "@UH0", "@UH1", "@UH2", "@UW", "@UW0", "@UW1", "@UW2", "@V", "@W", "@Y", "@Z", "@ZH"]
vits_default_symbols = ['_', ',', '.', '!', '?', '-', 'A', 'E', 'I', 'N', 'O', 'Q', 'U', 'a', 'b', 'd', 'e', 'f','g', 'h', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'ʃ', 'ʧ', '↓', '↑', ' ']


def text_to_sequence(text, symbols=None):
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    seq = [_symbol_to_id[c] for c in text if c in symbols]
    return seq

def inference_taco(tts_model, hifigan_model, target_text, output):
    global is_init_model, model, generator, np, torch, symbols
    def synthesis():
        text_list = [target_text]
        if enable_batch_out:
            text_list = target_text.split('|')
        for n, tar_text in enumerate(text_list):
            global label_img, img
            text = tar_text
            sequence = np.array(text_to_sequence(text, symbols))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()

            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
            # Draw mel out
            fig = Figure(figsize=(5, 1.8), dpi=100)
            canvas = FigureCanvasAgg(fig)
            cmap = LinearSegmentedColormap.from_list('my_cmap', [(1, 1, 1), (161/255, 234/255, 251/255)], N=200)
            # Do some plotting.
            ax = fig.add_subplot(1, 2, 1)
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_color('#F6F6F6')
            ax.spines['bottom'].set_color('#F6F6F6')
            ax.spines['left'].set_color('#F6F6F6')
            ax.spines['right'].set_color('#F6F6F6')
            ax.tick_params(axis='x',colors='#AAAAAA')
            ax.tick_params(axis='y',colors='#AAAAAA')
            ax.imshow(mel_outputs.float().data.cpu().numpy()[0], aspect='auto', origin='lower', interpolation='none', cmap=cmap)
            ax = fig.add_subplot(1, 2, 2)
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_color('#F6F6F6')
            ax.spines['bottom'].set_color('#F6F6F6')
            ax.spines['left'].set_color('#F6F6F6')
            ax.spines['right'].set_color('#F6F6F6')
            ax.tick_params(axis='x',colors='#AAAAAA')
            ax.tick_params(axis='y',colors='#AAAAAA')
            ax.imshow(alignments.float().data.cpu().numpy()[0].T, aspect='auto', origin='lower', interpolation='none',cmap=cmap)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())

            im = Image.fromarray(rgba)
            img = ImageTk.PhotoImage(im)
            label_img = ttk.Label(taco_tab, image=img)
            label_img.grid(row=5, column=2, padx=0, pady=5)

            
            with torch.no_grad():
                raw_audio = generator(mel_outputs.float())
                audio = raw_audio.squeeze()
                audio = audio * 32768.0
                audio = audio.cpu().numpy().astype('int16')
                if enable_custom_filename:
                    write(output+'.wav', h.sampling_rate, audio)
                else:
                    if enable_batch_out:
                        write(os.path.join(output,f'output_{n}.wav'), h.sampling_rate, audio)
                    else:
                        write(os.path.join(output,'output.wav'), h.sampling_rate, audio)

    if not is_init_model:
        import numpy as np
        import torch
        from tacotron2.hparams import create_hparams
        from tacotron2.train import load_model
        # load symbols
        symbols = taco_default_symbols
        try:
            print('Trying to load MoeTTS config...')
            moe_cfg_taco = json.load(open(('/'.join(tts_model.split('/')[:-1]))+'/moetts.json', encoding='utf-8'))
            symbols = moe_cfg_taco['symbols']
        except:
            print('Failed to load MoeTTS config, use default configuration.')

        hparams = create_hparams()
        hparams.n_symbols=len(symbols)
        hparams.sampling_rate = 22050

        checkpoint_path = tts_model

        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
        _ = model.cpu().eval()
       
        # sys.path.append('hifigan/')
        from hifigan.models import Generator
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self

        def load_checkpoint(filepath, device):
            assert os.path.isfile(filepath)
            print("Loading '{}'".format(filepath))
            checkpoint_dict = torch.load(filepath, map_location=device)
            print("Complete.")
            return checkpoint_dict
        hifigan_cfg = '/'.join(hifigan_model.split('/')[:-1])
        config_file = (hifigan_cfg+'/config.json')
        with open(config_file) as f:
            data = f.read()
        global h
        json_config = json.loads(data)
        h = AttrDict(json_config)

        torch.manual_seed(h.seed)
        global device
        device = torch.device('cpu')

        generator = Generator(h).to(device)
        state_dict_g = load_checkpoint(hifigan_model, device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
        is_init_model = True
        synthesis()
    else:
        synthesis()
# vits

def get_text(text, hps, symbols):
    text_norm = text_to_sequence(text, symbols)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def inference_vitss(tts_model, target_text, output):
    global hps, net_g, torch, SynthesizerTrn, symbols, text_to_sequence, commons
    global is_init_model
    def synthesis():
        text_list = [target_text]
        if enable_batch_out:
            text_list = target_text.split('|')
        for n, tar_text in enumerate(text_list):
            stn_tst = get_text(tar_text, hps, symbols)
            with torch.no_grad():
                x_tst = stn_tst.cpu().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
                audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=vits_length_scale)[0][0,0].data.cpu().float().numpy()
                audio = audio * 32768.0
                audio = audio.squeeze()
                audio = audio.astype('int16')
                if enable_custom_filename:
                    write(output+'.wav', 22050, audio)
                else:
                    if enable_batch_out:
                        write(os.path.join(output,f'output_vitss_{n}.wav'), 22050, audio)
                    else:
                        write(os.path.join(output,'output_vitss.wav'), 22050, audio)
            
    if not is_init_model:
        # init model
        import torch
        import vits.commons as commons
        import vits.utils
        from vits.models import SynthesizerTrn
        # sys.path.append('./custom/')
        # from vitstext.symbols import symbols
        # from vitstext import text_to_sequence
        # load symbols
        symbols = vits_default_symbols
        try:
            print('Trying to load MoeTTS config...')
            moe_cfg_vits = json.load(open(('/'.join(tts_model.split('/')[:-1]))+'/moetts.json', encoding='utf-8'))
            symbols = moe_cfg_vits['symbols']
        except:
            print('Failed to load MoeTTS config, use default configuration.')
        vitss_cfg = '/'.join(tts_model.split('/')[:-1])
        config_file = (vitss_cfg+'/config.json')
        hps = vits.utils.get_hparams_from_file(config_file)
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cpu()
        _ = net_g.eval()

        _ = vits.utils.load_checkpoint(tts_model, net_g, None)
        is_init_model = True
        synthesis()
    else:
        synthesis()

def inference_vitsm(tts_model, target_text, output, speaker_id, mode='synthesis', target_speaker_id=0, src_audio=''):
    """
    :params
    mode: synthesis, convert
    """
    global hps, net_g, torch, SynthesizerTrn, symbols, text_to_sequence, commons
    global is_init_model
    global spectrogram_torch, load_wav_to_torch
    def convert():
        src_speaker_id = torch.LongTensor([int(speaker_id)]).cpu()
        tgt_speaker_id = torch.LongTensor([int(target_speaker_id)]).cpu()
        audio, sampling_rate = load_wav_to_torch(src_audio)

        if sampling_rate != 22050:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, 22050))
        # wav to spec
        audio_norm = audio / 32768
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, 1024, 22050, 256, 1024, center=False).cpu()
        # spec = torch.squeeze(spec, 0)
        spec_length = torch.LongTensor([spec.data.shape[2]]).cpu()
        audio = net_g.voice_conversion(spec, spec_length, 
            sid_src=src_speaker_id, sid_tgt=tgt_speaker_id)[0][0,0].data.cpu().float().numpy()

        audio = audio * 32768.0
        audio = audio.squeeze()
        audio = audio.astype('int16')
        if enable_custom_filename:
            write(output+'.wav', 22050, audio)
        else:
            write(os.path.join(output,'output_convert.wav'), 22050, audio)
    
    def synthesis():
        text_list = [target_text]
        if enable_batch_out:
            text_list = target_text.split('|')
        for n, tar_text in enumerate(text_list):
            stn_tst = get_text(tar_text, hps, symbols)
            with torch.no_grad():
                x_tst = stn_tst.cpu().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
                sid = torch.LongTensor([int(speaker_id)]).cpu()
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=vits_length_scale)[0][0,0].data.cpu().float().numpy()
                audio = audio * 32768.0
                audio = audio.squeeze()
                audio = audio.astype('int16')
                if enable_custom_filename:
                    write(output+'.wav', 22050, audio)
                else:
                    if enable_batch_out:
                        write(os.path.join(output,f'output_vitsm_{n}.wav'), 22050, audio)
                    else:
                        write(os.path.join(output,'output_vitsm.wav'), 22050, audio)
    if not is_init_model:
        import torch
        import vits.commons as commons
        import vits.utils
        from vits.models import SynthesizerTrn
        
        from vits.mel_processing import spectrogram_torch
        from vits.utils import load_wav_to_torch

        # sys.path.append('./custom/')
        # from vitstext.symbols import symbols
        # from vitstext import text_to_sequence
        symbols = vits_default_symbols
        try:
            print('Trying to load MoeTTS config...')
            moe_cfg_vits = json.load(open(('/'.join(tts_model.split('/')[:-1]))+'/moetts.json', encoding='utf-8'))
            symbols = moe_cfg_vits['symbols']
            # speakers = moe_cfg_vits['speakers']
            # print(f'Speakers:\n{speakers}')
        except:
            print('Failed to load MoeTTS config, use default configuration.')
        
        vitss_cfg = '/'.join(tts_model.split('/')[:-1])
        config_file = (vitss_cfg+'/config.json')
        hps = vits.utils.get_hparams_from_file(config_file)
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).cpu()
        _ = net_g.eval()

        _ = vits.utils.load_checkpoint(tts_model, net_g, None)
        is_init_model = True
        if mode=='synthesis':
            synthesis()
        elif mode=='convert':
            convert()
    else:
        if mode=='synthesis':
            synthesis()
        elif mode=='convert':
            convert()

# Main window
root = tk.Tk()
root.geometry("760x450")
root.title('MoeTTS-CPU')
root.iconbitmap('./favicon.ico')
style = ttk.Style("minty")
# DPI adapt

ctypes.windll.shcore.SetProcessDpiAwareness(1)
ScaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0)
root.tk.call('tk', 'scaling', ScaleFactor/80)
root.geometry(f"{int(760+760*0.5*(ScaleFactor/100-1))}x{int(450+450*0.5*(ScaleFactor/100-1))}")

nb = ttk.Notebook(root)
nb.pack(side=LEFT, padx=0,pady=(10, 0),  expand=YES, fill=BOTH)

# Tacotron2
taco_tab = ttk.Frame(nb)
taco_entry_ipadx = 175
# Tacotron2 model path select
taco_mp_label = ttk.Label(taco_tab, text='Tacotron2 模型：')
taco_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15,5))


taco_mp = ttk.Entry(taco_tab)
taco_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=taco_entry_ipadx)

taco_mp_btn = ttk.Button(taco_tab,  text="浏览文件", bootstyle=(INFO, OUTLINE), 
    command=lambda :file_locate(taco_mp, reset_model=True))
taco_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# hifigan model path select
hg_mp_label = ttk.Label(taco_tab, text='HifiGAN 模型：')
hg_mp_label.grid(row=2, column=1, padx=(20, 5), pady=5)


hg_mp = ttk.Entry(taco_tab)
hg_mp.grid(row=2, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

hg_mp_btn = ttk.Button(taco_tab, text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(hg_mp, reset_model=True))
hg_mp_btn.grid(row=2, column=3, padx=5, pady=5)

# output path select
taco_out_label = ttk.Label(taco_tab, text='输出目录：')
taco_out_label.grid(row=3, column=1, padx=(20, 5), pady=5)


taco_out = ttk.Entry(taco_tab)
taco_out.grid(row=3, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

taco_out_btn = ttk.Button(taco_tab, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :save_locate(taco_out))
taco_out_btn.grid(row=3, column=3, padx=5, pady=5)

# text input
taco_text_label = ttk.Label(taco_tab, text='待合成文本：')
taco_text_label.grid(row=4, column=1, padx=(20, 5), pady=5)

# def test():
#     global label_img, img
#     img = ImageTk.PhotoImage(Image.open('test.png'))
#     label_img = ttk.Label(taco_tab, image=img)
#     label_img.grid(row=5, column=2, padx=0, pady=5)

taco_target_text = ttk.Entry(taco_tab)
taco_target_text.grid(row=4, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)
taco_text_btn = ttk.Button(taco_tab, text="合成语音", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :inference_taco(taco_mp.get(), hg_mp.get(), taco_target_text.get(), taco_out.get()))
    #lambda :inference(tts_mp.get(), hg_mp.get(), target_text.get(), out.get())
taco_text_btn.grid(row=4, column=3, padx=5, pady=5)

# Draw result
taco_res_label = ttk.Label(taco_tab, text='Mel out:')
taco_res_label.grid(row=5, column=1, padx=5, pady=5)


# VITS single speaker
vits_tab = ttk.Frame(nb)
# VITS model path select
vitss_mp_label = ttk.Label(vits_tab, text='VITS 单角色模型：')
vitss_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15,5))


vitss_mp = ttk.Entry(vits_tab)
vitss_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=taco_entry_ipadx)

vitss_mp_btn = ttk.Button(vits_tab,  text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(vitss_mp, reset_model=True))
vitss_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# output path select
vitss_out_label = ttk.Label(vits_tab, text='输出目录：')
vitss_out_label.grid(row=2, column=1, padx=(20, 5), pady=5)


vitss_out = ttk.Entry(vits_tab)
vitss_out.grid(row=2, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vitss_out_btn = ttk.Button(vits_tab, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :save_locate(vitss_out))
vitss_out_btn.grid(row=2, column=3, padx=5, pady=5)

# text input
vitss_text_label = ttk.Label(vits_tab, text='待合成文本：')
vitss_text_label.grid(row=3, column=1, padx=(20, 5), pady=5)


vitss_target_text = ttk.Entry(vits_tab)
vitss_target_text.grid(row=3, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)
vitss_text_btn = ttk.Button(vits_tab, text="合成语音", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :inference_vitss(vitss_mp.get(), vitss_target_text.get(), vitss_out.get()))
vitss_text_btn.grid(row=3, column=3, padx=5, pady=5)

# VITS multi speaker
vits_tab_2 = ttk.Frame(nb)
# VITS model path select
vitsm_mp_label = ttk.Label(vits_tab_2, text='VITS 多角色模型：')
vitsm_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15,5))


vitsm_mp = ttk.Entry(vits_tab_2)
vitsm_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=taco_entry_ipadx)

vitsm_mp_btn = ttk.Button(vits_tab_2,  text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(vitsm_mp, reset_model=True))
vitsm_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# output path select
vitsm_out_label = ttk.Label(vits_tab_2, text='输出目录：')
vitsm_out_label.grid(row=2, column=1, padx=(20, 5), pady=5)


vitsm_out = ttk.Entry(vits_tab_2)
vitsm_out.grid(row=2, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vitsm_out_btn = ttk.Button(vits_tab_2, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :save_locate(vitsm_out))
vitsm_out_btn.grid(row=2, column=3, padx=5, pady=5)
# speaker select
src_speaker_label = ttk.Label(vits_tab_2, text='原角色ID：')
src_speaker_label.grid(row=3, column=1, padx=(20, 5), pady=5)

src_speaker = ttk.Entry(vits_tab_2)
src_speaker.grid(row=3, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

# text input
vitsm_text_label = ttk.Label(vits_tab_2, text='待合成文本：')
vitsm_text_label.grid(row=4, column=1, padx=(20, 5), pady=5)


vitsm_target_text = ttk.Entry(vits_tab_2)
vitsm_target_text.grid(row=4, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vitsm_text_btn = ttk.Button(vits_tab_2, text="合成语音", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :inference_vitsm(vitsm_mp.get(), vitsm_target_text.get(), vitsm_out.get(), src_speaker.get()))
vitsm_text_btn.grid(row=4, column=3, padx=5, pady=5)

# audio source
vitsm_src_audio_label = ttk.Label(vits_tab_2, text='待迁移音频：')
vitsm_src_audio_label.grid(row=5, column=1, padx=(20, 5), pady=5)


vitsm_src_audio = ttk.Entry(vits_tab_2)
vitsm_src_audio.grid(row=5, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vitsm_src_audio_btn = ttk.Button(vits_tab_2, text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(vitsm_src_audio))
vitsm_src_audio_btn.grid(row=5, column=3, padx=5, pady=5)

# voice conversion
vitsm_tgt_speaker_label = ttk.Label(vits_tab_2, text=' 目标角色ID：')
vitsm_tgt_speaker_label.grid(row=6, column=1, padx=(20, 5), pady=5)


vitsm_tgt_speaker = ttk.Entry(vits_tab_2)
vitsm_tgt_speaker.grid(row=6, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

vitsm_tgt_audio_btn = ttk.Button(vits_tab_2, text="语音迁移", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :inference_vitsm(vitsm_mp.get(), vitsm_target_text.get(), vitsm_out.get(), src_speaker.get(), 
    mode='convert', target_speaker_id=vitsm_tgt_speaker.get(), src_audio=vitsm_src_audio.get()))
vitsm_tgt_audio_btn.grid(row=6, column=3, padx=5, pady=5)

jtalk_mode = 'rsa'
replace_ts = False

# tool box
def jtalk_g2p(text, entry_box):
    entry_box.delete(0, 'end')
    root.update()
    # result = os.popen(f'.\jp_g2p\japanese_g2p.exe -{jtalk_mode} {text}')
    # res = result.read()
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

# 滚动条设计尝试
# canvas=ttk.Canvas(nb, width=800, height=1000)
# canvas.place(x=0, y=0)
tool_tab = ttk.Frame(nb)
# canvas.create_window(1100, 400,anchor='center' , window=tool_tab) 
# Japanese g2p
jtalk_group = ttk.Labelframe(master=tool_tab, text="OpenJtalk g2p (base on CjangCjengh's jp_g2p tool):", padding=15)
jtalk_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)
# nb.pack(side=LEFT, padx=0,pady=(10, 0),  expand=YES, fill=BOTH)
jtext_label = ttk.Label(jtalk_group, text='日语文本：')
jtext_label.grid(row=1, column=1, padx=(20, 5), pady=5)


jtext = ttk.Entry(jtalk_group)
jtext.grid(row=1, column=2, padx=20, pady=5, ipadx=170)

jtext_btn = ttk.Button(jtalk_group,  text="转换(g2p)", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :jtalk_g2p(jtext.get(), jtext))
jtext_btn.grid(row=1, column=3, padx=5, pady=5)

text_label = ttk.Label(jtalk_group, text='转换模式:')
text_label.place(x=20, y=50)
j_radio_var = ttk.IntVar()
radio1 = ttk.Radiobutton(jtalk_group, text="普通转换",variable=j_radio_var,  value=1, command=lambda:set_jtalk_mode('r'))
radio1.place(x=150, y=50)
radio2 = ttk.Radiobutton(jtalk_group, text="空格分词",variable=j_radio_var, value=2, command=lambda:set_jtalk_mode('rs'))
radio2.place(x=260, y=50)
radio3 = ttk.Radiobutton(jtalk_group, text="分词+调形",variable=j_radio_var, value=3, command=lambda:set_jtalk_mode('rsa'))
radio3.place(x=370, y=50)
jtalk_check1 = ttk.Checkbutton(jtalk_group, text='替换ʦ到ts', command=lambda:set_replace_ts())
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

pinyin_group = ttk.Labelframe(master=tool_tab, text="pypinyin g2p:", padding=15)
pinyin_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)
pinyin_label = ttk.Label(pinyin_group, text='中文文本：')
pinyin_label.grid(row=1, column=1, padx=(20, 5), pady=5)


pinyin_text = ttk.Entry(pinyin_group)
pinyin_text.grid(row=1, column=2, padx=20, pady=5, ipadx=170)

pinyin_text_btn = ttk.Button(pinyin_group,  text="转换(g2p)", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :py_g2p(pinyin_text.get(), pinyin_text))
pinyin_text_btn.grid(row=1, column=3, padx=5, pady=5)

pinyin_text_label = ttk.Label(pinyin_group, text='转换模式:')
pinyin_text_label.place(x=20, y=50)
pinyin_radio_var = ttk.IntVar()
pinyin_radio1 = ttk.Radiobutton(pinyin_group, text="普通转换",variable=pinyin_radio_var,  value=4, command=lambda:set_pinyin_mode('normal'))
pinyin_radio1.place(x=150, y=50)
pinyin_radio2 = ttk.Radiobutton(pinyin_group, text="数字声调",variable=pinyin_radio_var,  value=5, command=lambda:set_pinyin_mode('tone3'))
pinyin_radio2.place(x=260, y=50)
pinyin_radio3 = ttk.Radiobutton(pinyin_group, text="注音符号",variable=pinyin_radio_var,  value=6, command=lambda:set_pinyin_mode('bopomofo'))
pinyin_radio3.place(x=370, y=50)
pinyin_check1 = ttk.Checkbutton(pinyin_group, text='是否空格', command=lambda:set_pinyin_space())
pinyin_check1.place(x=480, y=50)

# Audio tools
def audio_convert(audio_path, out_path):
    os.system(f'ffmpeg -i {audio_path} -ac 1 -ar 22050 {out_path}')

audio_group = ttk.Labelframe(master=tool_tab, text="音频转换(到22050 采样率,单声道wav)(依赖ffmepg):", padding=15)
audio_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)

audio_label = ttk.Label(audio_group, text='待转换音频：')
audio_label.grid(row=1, column=1, padx=(20, 5), pady=5)


audio_path = ttk.Entry(audio_group)
audio_path.grid(row=1, column=2, padx=20, pady=5, ipadx=170)

audio_sel_btn = ttk.Button(audio_group,  text="浏览文件", bootstyle=(INFO, OUTLINE), 
    command=lambda :file_locate(audio_path))
audio_sel_btn.grid(row=1, column=3, padx=5, pady=5)

audio_label2 = ttk.Label(audio_group, text='输出位置：')
audio_label2.grid(row=2, column=1, padx=(20, 5), pady=5)


audio_out = ttk.Entry(audio_group)
audio_out.grid(row=2, column=2, padx=20, pady=5, ipadx=170)

audio_out_btn = ttk.Button(audio_group,  text="浏览目录", bootstyle=(INFO, OUTLINE), 
    command=lambda :save_file_locate(audio_out))
audio_out_btn.grid(row=2, column=3, padx=5, pady=5)

audio_out_btn2 = ttk.Button(audio_group,  text="   转 换   ", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :audio_convert(audio_path.get(), str(audio_out.get())+'.wav'))
audio_out_btn2.grid(row=3, column=3, padx=5, pady=5)

# settings tab
def set_batch_mode():
    global enable_batch_out 
    enable_batch_out = ~enable_batch_out

def set_custom_filename():
    global enable_custom_filename 
    enable_custom_filename = ~enable_custom_filename 

setting_tab = ttk.Frame(nb)
file_setting_group = ttk.Labelframe(master=setting_tab, text="文件设置:", padding=15)
file_setting_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=40)

file_set_check1 = ttk.Checkbutton(file_setting_group, text='启用批量合成模式', command=lambda:set_batch_mode())
file_set_check1.grid(row=1, column=1, padx=5, pady=5)
file_set_check2 = ttk.Checkbutton(file_setting_group, text='启用自定义文件名', command=lambda:set_custom_filename())
file_set_check2.grid(row=1, column=2, padx=5, pady=5)
file_set_help = ttk.Label(file_setting_group, text='说明：启用批量合成模式后，文本框中以"|"分隔句子后，会批量转换每句到文件夹下以单独文件存放。\
    \n批量转换模式无法自定义文件名。')
file_set_help.place(x=20, y=50)

# VITS inference settings
vits_setting_group = ttk.Labelframe(master=setting_tab, text="VITS 语速设定(浮点数, 越大越慢):", padding=15)
vits_setting_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)

vits_ls_label = ttk.Label(vits_setting_group, text='语速 (默认1.0)：')
vits_ls_label.grid(row=1, column=1, padx=(20, 5), pady=5)
# vits length scale
vits_ls = ttk.Entry(vits_setting_group)
vits_ls.grid(row=1, column=2, padx=(20, 5), pady=5)
vits_ls_btn = ttk.Button(vits_setting_group,  text="设定", bootstyle=(INFO, OUTLINE), 
    command=lambda :set_vits_length_scale(vits_ls.get()))
vits_ls_btn.grid(row=1, column=3, padx=5, pady=5)

# ttk Theme 
theme_group = ttk.Labelframe(master=setting_tab, text="外观设定:", padding=15)
theme_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=20)

style = ttk.Style()
style.configure('Horizontal.TScrollbar')
theme_names = style.theme_names()
theme_lb = ttk.Label(theme_group, text="选择主题:")
theme_lb.pack(padx=(20, 5), side=LEFT)
theme_cbo = ttk.Combobox(master=theme_group,text=style.theme.name,values=theme_names)
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

def set_crepe():
    global use_crepe
    use_crepe = ~use_crepe

def set_pe():
    global use_pe
    use_pe = ~use_pe
# TODO: 规范代码，规范import
def diff_svc_infer(model_path,input_file,out_path,tran=0,acc=20):
    global is_init_model,run_clip,Svc,model
    accelerate = acc
    hubert_gpu = False
    project_name = ''
    audio_rate = 24000
    model_folder = '/'.join(model_path.split('/')[:-1])
    config_file = (model_folder+'/config.yaml')
    with open(config_file) as f:
        cfg_yaml = f.read()
        asr_index = cfg_yaml.find('audio_sample_rate: ')+19
        audio_rate = int(cfg_yaml[asr_index:asr_index+5])
        # print(cfg_yaml[asr_index:asr_index+5])
    # model = None
    # 重新选择以及首次加载模型再初始化
    if not is_init_model:
        sys.path.append('./diff_svc/')
        from diff_svc.infer_tools.infer_tool import Svc
        from diff_svc.infer import run_clip
        model = Svc(project_name, config_file, hubert_gpu, model_path)
        is_init_model = True
    if not enable_custom_filename:
        out_path = os.path.join(out_path,f'{input_file.split("/")[-1][:-4]}_key_{tran}.wav')
    run_clip(model, key=tran, acc=accelerate, use_crepe=use_crepe, thre=0.05, use_pe=use_pe, use_gt_mel=False,
                add_noise_step=500, f_name=input_file,audio_rate=audio_rate,out_path=out_path)

# diff svc gui
diff_svc_tab = ttk.Frame(nb)

dsvc_mp_label = ttk.Label(diff_svc_tab, text='Diff-SVC模型：')
dsvc_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15,5))


dsvc_mp = ttk.Entry(diff_svc_tab)
dsvc_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=taco_entry_ipadx)

dsvc_mp_btn = ttk.Button(diff_svc_tab,  text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(dsvc_mp, reset_model=True))
dsvc_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# output path select
dsvc_out_label = ttk.Label(diff_svc_tab, text='输出目录：')
dsvc_out_label.grid(row=2, column=1, padx=(20, 5), pady=5)


dsvc_out = ttk.Entry(diff_svc_tab)
dsvc_out.grid(row=2, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

dsvc_out_btn = ttk.Button(diff_svc_tab, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :save_locate(dsvc_out))
dsvc_out_btn.grid(row=2, column=3, padx=5, pady=5)

# raw audio select
dsvc_input_label = ttk.Label(diff_svc_tab, text='待转换音频：')
dsvc_input_label.grid(row=3, column=1, padx=(20, 5), pady=5)


dsvc_input = ttk.Entry(diff_svc_tab)
dsvc_input.grid(row=3, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)

# 点炒饭禁止！
dsvc_input_btn = ttk.Button(diff_svc_tab, text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(dsvc_input,file_types=(("wav files", "*.wav"),("ogg files", "*.ogg"),)))
dsvc_input_btn.grid(row=3, column=3, padx=5, pady=5)

# 变调，Crepe，PE

dsvc_pitch_label = ttk.Label(diff_svc_tab, text='升降半音|支持正负整数:')
dsvc_pitch_label.grid(row=4, column=1, padx=(5, 5), pady=5)


dsvc_pitch = ttk.Entry(diff_svc_tab,width=10)
dsvc_pitch.grid(row=4, column=2, padx=20, pady=5, ipadx=0,sticky='w')
dsvc_pitch.insert(index=0,string="0")

dsvc_crepe_check = ttk.Checkbutton(diff_svc_tab, text='启用Crepe[耗时较长|1:8]', command=lambda:set_crepe())
dsvc_crepe_check.place(x=300,y=155)

dsvc_pe_check = ttk.Checkbutton(diff_svc_tab, text='启用pe', command=lambda:set_pe())
dsvc_pe_check.place(x=520,y=155)

# 加速
dsvc_acc_label = ttk.Label(diff_svc_tab, text='加速倍率[推荐20]：')
dsvc_acc_label.grid(row=5, column=1, padx=(5, 5), pady=5)


dsvc_acc = ttk.Entry(diff_svc_tab)
dsvc_acc.grid(row=5, column=2, padx=20, pady=5, ipadx=taco_entry_ipadx)
dsvc_acc.insert(0,'20')

dsvc_input_btn = ttk.Button(diff_svc_tab, text="转换音频", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :diff_svc_infer(model_path=dsvc_mp.get(),input_file=dsvc_input.get(), 
    tran=int(dsvc_pitch.get()),out_path=dsvc_out.get(),acc=int(dsvc_acc.get())))
dsvc_input_btn.grid(row=4, column=3, padx=5, pady=5)

# dsvc_f0_label = ttk.Label(diff_svc_tab, text='F0 OUT：')
# dsvc_f0_label.grid(row=5, column=1, padx=(20, 5), pady=5)

nb.add(taco_tab, text="Tacotron2", sticky=NW)
nb.add(vits_tab, text="VITS-Single", sticky=NW)
nb.add(vits_tab_2, text="VITS-Multi", sticky=NW)
nb.add(diff_svc_tab, text="Diff-svc", sticky=NW)
nb.add(tool_tab, text="ToolBox", sticky=NW)
nb.add(setting_tab, text="Settings", sticky=NW)

root.resizable(0,0)
root.mainloop()