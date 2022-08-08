import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog

import os
import sys

from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.io.wavfile import write

import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# functions

is_init_model= False

def file_locate(entry_box, reset_model=False):
    global is_init_model
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

def inference_taco(tts_model, hifigan_model, target_text, output):
    global is_init_model, model, generator, np, torch, text_to_sequence, json
    def synthesis():
        global label_img, img
        text = target_text
        sequence = np.array(text_to_sequence(text, ['custom_cleaners']))[None, :]
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
        ax.imshow(mel_outputs.float().data.cpu().numpy()[0], aspect='auto', origin='bottom', interpolation='none', cmap=cmap)
        ax = fig.add_subplot(1, 2, 2)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_color('#F6F6F6')
        ax.spines['bottom'].set_color('#F6F6F6')
        ax.spines['left'].set_color('#F6F6F6')
        ax.spines['right'].set_color('#F6F6F6')
        ax.tick_params(axis='x',colors='#AAAAAA')
        ax.tick_params(axis='y',colors='#AAAAAA')
        ax.imshow(alignments.float().data.cpu().numpy()[0].T, aspect='auto', origin='bottom', interpolation='none',cmap=cmap)
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
            write(os.path.join(output,'output.wav'), h.sampling_rate, audio)

    if not is_init_model:
        import numpy as np
        import torch
        from tacotron2.hparams import create_hparams

        from tacotron2.train import load_model
        sys.path.append('./custom/')
        from tacotext import text_to_sequence
        hparams = create_hparams()
        hparams.sampling_rate = 22050

        checkpoint_path = tts_model

        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
        _ = model.cpu().eval()

        import json
       
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

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def inference_vitss(tts_model, target_text, output):
    global hps, net_g, torch, SynthesizerTrn, symbols, text_to_sequence, commons
    global is_init_model
    def synthesis():
        stn_tst = get_text(target_text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cpu().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            audio = audio * 32768.0
            audio = audio.squeeze()
            audio = audio.astype('int16')
            write(os.path.join(output,'output_vitss.wav'), 22050, audio)
    if not is_init_model:
        # init model
        import torch
        import vits.commons as commons
        import vits.utils
        from vits.models import SynthesizerTrn
        sys.path.append('./custom/')
        from vitstext.symbols import symbols
        from vitstext import text_to_sequence
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
        write(os.path.join(output,'output_convert.wav'), 22050, audio)
    
    def synthesis():
        stn_tst = get_text(target_text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cpu().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
            sid = torch.LongTensor([int(speaker_id)]).cpu()
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            audio = audio * 32768.0
            audio = audio.squeeze()
            audio = audio.astype('int16')
            write(os.path.join(output,'output_vitsm.wav'), 22050, audio)
    if not is_init_model:
        import torch
        import vits.commons as commons
        import vits.utils
        from vits.models import SynthesizerTrn
        
        from vits.mel_processing import spectrogram_torch
        from vits.utils import load_wav_to_torch

        sys.path.append('./custom/')
        from vitstext.symbols import symbols
        from vitstext import text_to_sequence
        
        
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

nb = ttk.Notebook(root)
nb.pack(side=LEFT, padx=0,pady=(10, 0),  expand=YES, fill=BOTH)

# Tacotron2
taco_tab = ttk.Frame(nb)
taco_entry_ipadx = 150
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

taco_out_btn = ttk.Button(taco_tab, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :directory_locate(taco_out))
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

vitss_out_btn = ttk.Button(vits_tab, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :directory_locate(vitss_out))
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

vitsm_out_btn = ttk.Button(vits_tab_2, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :directory_locate(vitsm_out))
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
# tool box
def jtalk_g2p(text, entry_box):
    entry_box.delete(0, 'end')
    entry_box.insert(0, '正在转换...')
    root.update()
    result = os.popen(f'.\jp_g2p\japanese_g2p.exe -{jtalk_mode} {text}')
    res = result.read()
    entry_box.delete(0, 'end')
    entry_box.insert(0, res)

def set_jtalk_mode(mode):
    global jtalk_mode
    jtalk_mode = mode

tool_tab = ttk.Frame(nb)

jtalk_group = ttk.Labelframe(master=tool_tab, text="OpenJtalk g2p (base on CjangCjengh's jp_g2p tool):", padding=15)
jtalk_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=30)
# nb.pack(side=LEFT, padx=0,pady=(10, 0),  expand=YES, fill=BOTH)
jtext_label = ttk.Label(jtalk_group, text='日语文本：')
jtext_label.grid(row=1, column=1, padx=(20, 5), pady=(15,5))


jtext = ttk.Entry(jtalk_group)
jtext.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=170)

jtext_btn = ttk.Button(jtalk_group,  text="转换(g2p)", bootstyle=(INFO, OUTLINE), 
    command=lambda :jtalk_g2p(jtext.get(), jtext))
jtext_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

text_label = ttk.Label(jtalk_group, text='转换模式:')
text_label.place(x=20, y=60)
radio1 = ttk.Radiobutton(jtalk_group, text="普通转换", value=1, command=lambda:set_jtalk_mode('r'))
radio1.place(x=150, y=60)
radio2 = ttk.Radiobutton(jtalk_group, text="空格分词", value=2, command=lambda:set_jtalk_mode('rs'))
radio2.place(x=260, y=60)
radio3 = ttk.Radiobutton(jtalk_group, text="分词+调形", value=3, command=lambda:set_jtalk_mode('rsa'))
radio3.place(x=370, y=60)

nb.add(taco_tab, text="Tacotron2", sticky=NW)
nb.add(vits_tab, text="VITS-Single", sticky=NW)
nb.add(vits_tab_2, text="VITS-Multi", sticky=NW)
nb.add(tool_tab, text="ToolBox", sticky=NW)
root.resizable(0,0)
root.mainloop()