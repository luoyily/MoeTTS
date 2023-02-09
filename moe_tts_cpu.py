import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog

import os
import ctypes
import json

import yaml

from scipy.io.wavfile import write

import g2p.jp.japanese_g2p as jp_g2p
import g2p.zh.chinese_g2p as zh_g2p

# win 10 notice
try:
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    is_notice_avaliable = True
except:
    is_notice_avaliable = False

init_state = {
    'vits':False,
    'taco':False,
    'svc':False
}
vits_combobox_enable = False
vits_moe_cfg = dict()
svc_combobox_enable = False
svc_cfg = dict()


# Settings
class Settings:
    def __init__(self) -> None:
        self.notice_at_fin = False
        self.enable_batch_out = False
        self.enable_custom_filename = False
        self.vits_length_scale = 1
        self.jtalk_mode = 'rsa'
        self.replace_ts = False
        self.pinyin_mode = 'normal'
        self.pinyin_space = False
        self.use_crepe = False
        self.use_crepe_tiny = False
        # （下拉框偏好设置）在配置中有角色表时优先使用下拉框选择角色
        self.prefer_combobox = True
        self.svc_auto_key = False
        self.svc_flac_output = False

settings = Settings()

# functions

def notice():
    if settings.notice_at_fin and is_notice_avaliable:
        try:
            toaster.show_toast(
                "MoeTTS", "您的推理任务已完成", icon_path="favicon.ico", duration=3, threaded=True)
        except:
            pass


def file_locate(entry_box, reset_model=False, file_types=None, locate_files=False,check_speakers=False):
    global vits_combobox_enable,vits_moe_cfg,vits_src_combobox,vits_tgt_combobox,svc_cfg,svc_spk_combobox,svc_combobox_enable
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
        init_state[reset_model] = False
    # 重置多角色选择器状态
    if check_speakers == 'vits' and vits_combobox_enable:
        vits_src_combobox.grid_remove()
        vits_tgt_combobox.grid_remove()
        vits_src_speaker.grid(row=3, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)
        vits_tgt_speaker.grid(row=6, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)
    elif check_speakers == 'svc' and svc_combobox_enable:
        svc_spk_combobox.grid_remove()
        dsvc_spk.grid(row=6, column=2, padx=20, pady=5, ipadx=0, sticky='w')
    # 检查多角色模型配置文件并决定是否加载下拉框
    if check_speakers=='vits':
        moetts_cfg = load_moetts_cfg(file)
        # 配置存在说话者列表即加载下拉框
        if moetts_cfg:
            if 'speakers' in moetts_cfg and settings.prefer_combobox:
                vits_src_speaker.grid_remove()
                vits_tgt_speaker.grid_remove()
                speakers = [k for k,v in moetts_cfg['speakers'].items()]
                vits_src_combobox = ttk.Combobox(vits_tab,values=speakers)
                vits_src_combobox.grid(row=3, column=2, padx=20, pady=5, ipadx=global_entry_ipadx-7)
                vits_tgt_combobox = ttk.Combobox(vits_tab,values=speakers)
                vits_tgt_combobox.grid(row=6, column=2, padx=20, pady=5, ipadx=global_entry_ipadx-7)
                vits_combobox_enable = True
                vits_moe_cfg = moetts_cfg['speakers']
    elif check_speakers=='svc' and settings.prefer_combobox:
        with open(('/'.join(file.split('/')[:-1]))+'/config.yaml', encoding='utf-8') as f:
            svc_cfg = yaml.safe_load(f)
            if 'speakers' in svc_cfg:
                speakers = svc_cfg['speakers']
                dsvc_spk.grid_remove()
                svc_spk_combobox = ttk.Combobox(diff_svc_tab,values=speakers,width=8)
                svc_spk_combobox.grid(row=6, column=2, padx=20, pady=5, ipadx=0, sticky='w')
                svc_combobox_enable = True
    
    print(file)

def vits_get_speaker(mode='src|tgt'):
    if mode == 'src':
        if vits_combobox_enable:
            speaker = vits_moe_cfg[vits_src_combobox.get()]
            return speaker
        else:
            speaker = vits_src_speaker.get()
            return speaker
    elif mode == 'tgt':
        if vits_combobox_enable:
            speaker = vits_moe_cfg[vits_tgt_combobox.get()]
            return speaker
        else:
            speaker = vits_tgt_speaker.get()
            return speaker

def svc_get_speaker():
    if svc_combobox_enable:
        speaker = svc_cfg['speakers'].index(svc_spk_combobox.get())
        return speaker
    else:
        return 0 if not dsvc_spk.get() else dsvc_spk.get()

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
    if settings.enable_custom_filename and not settings.enable_batch_out:
        save_file_locate(entry_box)
    else:
        directory_locate(entry_box)


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
    if auto_cleaner.get() == 1:
        jtalk_g2p(taco_target_text.get(), taco_target_text)
    elif auto_cleaner.get() == 2:
        py_g2p(taco_target_text.get(), taco_target_text)
    target_text = taco_target_text.get()
    root.update()
    def init_taco_model():
        global taco_model, hifigan_model, symbols, torch
        if not init_state['taco']:
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
            init_state['taco'] = True

    def synthesis():
        text_list = [target_text]
        if settings.enable_batch_out:
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
                if settings.enable_custom_filename:
                    write(output_path+'.wav', sampling_rate, audio)
                else:
                    if settings.enable_batch_out:
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


def inference_vits(vits_model_path, target_text, output, speaker_id, mode='synthesis', target_speaker_id=0, src_audio=None, send_to_svc=False):
    """
    :params
    mode: synthesis, convert
    """
    if auto_cleaner.get() == 1:
        jtalk_g2p(vits_target_text.get(), vits_target_text)
    elif auto_cleaner.get() == 2:
        py_g2p(vits_target_text.get(), vits_target_text)
    target_text = vits_target_text.get()
    root.update()
    def init_vits_model():
        global vits_hps, vits_model, symbols, vits_commons, torch
        global spectrogram_torch, load_wav_to_torch
        if not init_state['vits']:
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

            init_state['vits'] = True

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
        if settings.enable_custom_filename:
            write(output+'.wav', 22050, audio)
        else:
            write(os.path.join(output, 'output_convert.wav'), 22050, audio)

    def synthesis():
        text_list = [target_text]
        if settings.enable_batch_out:
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
                                         length_scale=settings.vits_length_scale)[0][0, 0].data.cpu().float().numpy()
                audio = (audio * 32768.0).squeeze().astype('int16')
                if not send_to_svc:
                    if settings.enable_custom_filename:
                        write(output+'.wav', 22050, audio)
                    else:
                        if settings.enable_batch_out:
                            write(os.path.join(
                                output, f'output_vits_{n}.wav'), 22050, audio)
                        else:
                            write(os.path.join(output, 'output_vits.wav'), 22050, audio)
                else:
                    write(os.path.join(output, 'output_vits_for_svc.wav'), 22050, audio)
                # if not settings.enable_batch_out:
                #     write(temp_audio, 22050, audio)
                
    init_vits_model()
    if mode == 'synthesis':
        synthesis()
    elif mode == 'convert':
        convert()
    if send_to_svc:
        print('Send to SVC...')
        diff_svc_infer(model_path=dsvc_mp.get(), input_files=[os.path.join(output, 'output_vits_for_svc.wav')],
            tran=int(dsvc_pitch.get()), out_path=dsvc_out.get(), acc=int(dsvc_acc.get()),
            spk_id=svc_get_speaker(),auto_key=settings.svc_auto_key)
    print(f'File saved to {output}.')
    notice()

def jtalk_g2p(text, entry_box):
    entry_box.delete(0, 'end')
    root.update()
    res = ''
    if settings.jtalk_mode == 'r':
        res = jp_g2p.get_romaji(text)
    elif settings.jtalk_mode == 'rs':
        res = jp_g2p.get_romaji_with_space(text)
    elif settings.jtalk_mode == 'rsa':
        res = jp_g2p.get_romaji_with_space_and_accent(text)
    if settings.replace_ts:
        res = res.replace('ʦ', 'ts')
    entry_box.delete(0, 'end')
    entry_box.insert(0, res)

def py_g2p(text, entry_box):
    entry_box.delete(0, 'end')
    root.update()
    res = zh_g2p.g2p(text, settings.pinyin_mode, settings.pinyin_space)
    entry_box.delete(0, 'end')
    entry_box.insert(0, res)

# Main window
root = tk.Tk()
root.geometry("760x450")
root.title('MoeTTS-CPU')
root.iconbitmap('./favicon.ico')
style = ttk.Style("minty")

auto_cleaner = tk.IntVar(root,value=0)

# DPI adapt

ctypes.windll.shcore.SetProcessDpiAwareness(1)
ScaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)
root.tk.call('tk', 'scaling', ScaleFactor/80)
root.geometry(
    f"{int(760+760*0.5*(ScaleFactor/100-1))}x{int(450+450*0.5*(ScaleFactor/100-1))}")

nb = ttk.Notebook(root)
nb.pack(side=LEFT, padx=0, pady=(10, 0),  expand=YES, fill=BOTH)

# TODO: （非必要）模块化GUI，如：Label，Entry，Button套件整理为一个（但会导致GUI内容读取需要重新配置）
# Tacotron2
taco_tab = ttk.Frame(nb)
global_entry_ipadx = 175
# Tacotron2 model path select
taco_mp_label = ttk.Label(taco_tab, text='Tacotron2 模型：')
taco_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15, 5))


taco_mp = ttk.Entry(taco_tab)
taco_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=global_entry_ipadx)

taco_mp_btn = ttk.Button(taco_tab,  text="浏览文件", bootstyle=(INFO, OUTLINE),
                         command=lambda: file_locate(taco_mp, reset_model='taco'))
taco_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# hifigan model path select
hg_mp_label = ttk.Label(taco_tab, text='HifiGAN 模型：')
hg_mp_label.grid(row=2, column=1, padx=(20, 5), pady=5)


hg_mp = ttk.Entry(taco_tab)
hg_mp.grid(row=2, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

hg_mp_btn = ttk.Button(taco_tab, text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(hg_mp, reset_model='taco'))
hg_mp_btn.grid(row=2, column=3, padx=5, pady=5)

# output path select
taco_out_label = ttk.Label(taco_tab, text='输出目录：')
taco_out_label.grid(row=3, column=1, padx=(20, 5), pady=5)


taco_out = ttk.Entry(taco_tab)
taco_out.grid(row=3, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

taco_out_btn = ttk.Button(taco_tab, text="浏览目录", bootstyle=(
    INFO, OUTLINE), command=lambda: save_locate(taco_out))
taco_out_btn.grid(row=3, column=3, padx=5, pady=5)

# text input
taco_text_label = ttk.Label(taco_tab, text='待合成文本：')
taco_text_label.grid(row=4, column=1, padx=(20, 5), pady=5)


taco_target_text = ttk.Entry(taco_tab)
taco_target_text.grid(row=4, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

if auto_cleaner == 1:
    jtalk_g2p(taco_target_text.get(), taco_target_text)
elif auto_cleaner == 2:
    py_g2p(taco_target_text.get(), taco_target_text)

taco_text_btn = ttk.Button(taco_tab, text="合成语音", bootstyle=(SECONDARY, OUTLINE),
                           command=lambda: inference_taco(taco_mp.get(), hg_mp.get(), taco_target_text.get(), taco_out.get()))
taco_text_btn.grid(row=4, column=3, padx=5, pady=5)

# VITS
vits_tab = ttk.Frame(nb)
# VITS model path select
vits_mp_label = ttk.Label(vits_tab, text='VITS 模型：')
vits_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15, 5))

vits_mp = ttk.Entry(vits_tab)
vits_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=global_entry_ipadx)

vits_mp_btn = ttk.Button(vits_tab,  text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(vits_mp, reset_model='vits',check_speakers='vits'))
vits_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5),ipadx=27)

# output path select
vits_out_label = ttk.Label(vits_tab, text='输出目录：')
vits_out_label.grid(row=2, column=1, padx=(20, 5), pady=5)


vits_out = ttk.Entry(vits_tab)
vits_out.grid(row=2, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

vits_out_btn = ttk.Button(vits_tab, text="浏览目录", bootstyle=(
    INFO, OUTLINE), command=lambda: save_locate(vits_out))
vits_out_btn.grid(row=2, column=3, padx=5, pady=5,ipadx=27)
# speaker select
src_speaker_label = ttk.Label(vits_tab, text='原角色ID：')
src_speaker_label.grid(row=3, column=1, padx=(20, 5), pady=5)

vits_src_speaker = ttk.Entry(vits_tab)
vits_src_speaker.grid(row=3, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

# text input
vits_text_label = ttk.Label(vits_tab, text='待合成文本：')
vits_text_label.grid(row=4, column=1, padx=(20, 5), pady=5)


vits_target_text = ttk.Entry(vits_tab)
vits_target_text.grid(row=4, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

vits_text_btn = ttk.Button(vits_tab, text="合成语音", bootstyle=(SECONDARY, OUTLINE),
                           command=lambda: inference_vits(vits_mp.get(), vits_target_text.get(), vits_out.get(), vits_get_speaker(mode='src')))
vits_text_btn.grid(row=4, column=3, padx=5, pady=5,ipadx=27)

vits_text_btn = ttk.Button(vits_tab, text="合成并发送至SVC", bootstyle=(SECONDARY, OUTLINE),
                           command=lambda: inference_vits(vits_mp.get(), vits_target_text.get(), vits_out.get(), vits_get_speaker(mode='src'),send_to_svc=True))
vits_text_btn.grid(row=3, column=3, padx=5, pady=5)
# audio source
vits_src_audio_label = ttk.Label(vits_tab, text='待迁移音频：')
vits_src_audio_label.grid(row=5, column=1, padx=(20, 5), pady=5)


vits_src_audio = ttk.Entry(vits_tab)
vits_src_audio.grid(row=5, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

vits_src_audio_btn = ttk.Button(vits_tab, text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(vits_src_audio))
vits_src_audio_btn.grid(row=5, column=3, padx=5, pady=5,ipadx=27)

# voice conversion
vits_tgt_speaker_label = ttk.Label(vits_tab, text=' 目标角色ID：')
vits_tgt_speaker_label.grid(row=6, column=1, padx=(20, 5), pady=5)


vits_tgt_speaker = ttk.Entry(vits_tab)
vits_tgt_speaker.grid(row=6, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

vits_tgt_audio_btn = ttk.Button(vits_tab, text="语音迁移", bootstyle=(SECONDARY, OUTLINE),
                                command=lambda: inference_vits(vits_mp.get(), vits_target_text.get(), vits_out.get(), vits_get_speaker(mode='src'),
                                                               mode='convert', target_speaker_id=vits_get_speaker(mode='tgt'), src_audio=vits_src_audio.get()))
vits_tgt_audio_btn.grid(row=6, column=3, padx=5, pady=5,ipadx=27)

# tool box

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
                         value=1, command=lambda: settings.__setattr__('jtalk_mode','r'))
radio1.place(x=150, y=50)
radio2 = ttk.Radiobutton(jtalk_group, text="空格分词", variable=j_radio_var,
                         value=2, command=lambda: settings.__setattr__('jtalk_mode','rs'))
radio2.place(x=260, y=50)
radio3 = ttk.Radiobutton(jtalk_group, text="分词+调形", variable=j_radio_var,
                         value=3, command=lambda: settings.__setattr__('jtalk_mode','rsa'))
radio3.place(x=370, y=50)
jtalk_check1 = ttk.Checkbutton(
    jtalk_group, text='替换ʦ到ts', command=lambda: settings.__setattr__('replace_ts',~ settings.replace_ts))
jtalk_check1.place(x=480, y=50)

jtalk_auto = ttk.Radiobutton(jtalk_group,text='待合成文本自动转换',value=1,variable=auto_cleaner)
jtalk_auto.place(x=590, y=50)
# Chinese g2p




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
                                value=4, command=lambda: settings.__setattr__('pinyin_mode','normal'))
pinyin_radio1.place(x=150, y=50)
pinyin_radio2 = ttk.Radiobutton(pinyin_group, text="数字声调", variable=pinyin_radio_var,
                                value=5, command=lambda: settings.__setattr__('pinyin_mode','tone3'))
pinyin_radio2.place(x=260, y=50)
pinyin_radio3 = ttk.Radiobutton(pinyin_group, text="注音符号", variable=pinyin_radio_var,
                                value=6, command=lambda: settings.__setattr__('pinyin_mode','bopomofo'))
pinyin_radio3.place(x=370, y=50)
pinyin_check1 = ttk.Checkbutton(
    pinyin_group, text='是否空格', command=lambda: settings.__setattr__('pinyin_space',~settings.pinyin_space))
pinyin_check1.place(x=480, y=50)
pinyin_auto = ttk.Radiobutton(pinyin_group,text='待合成文本自动转换',value=2,variable=auto_cleaner)
pinyin_auto.place(x=590, y=50)
# settings tab

setting_tab = ttk.Frame(nb)
file_setting_group = ttk.Labelframe(
    master=setting_tab, text="常规设置:", padding=15)
file_setting_group.pack(fill=BOTH, side=TOP, expand=True, ipadx=740, ipady=40)

file_set_check1 = ttk.Checkbutton(
    file_setting_group, text='启用批量合成模式', command=lambda: settings.__setattr__('enable_batch_out',~settings.enable_batch_out))
file_set_check1.grid(row=1, column=1, padx=5, pady=5)
file_set_check2 = ttk.Checkbutton(
    file_setting_group, text='启用自定义文件名', command=lambda: settings.__setattr__('enable_custom_filename',~settings.enable_custom_filename))
file_set_check2.grid(row=1, column=2, padx=5, pady=5)
file_set_help = ttk.Label(file_setting_group, text='说明：启用批量合成模式后，文本框中以"|"分隔句子后，会批量转换每句到文件夹下以单独文件存放。\
    \n批量转换模式无法自定义文件名。')
file_set_help.place(x=20, y=50)
# Notice setting
notice_set_check = ttk.Checkbutton(
    file_setting_group, text='启用完成通知', command=lambda: settings.__setattr__('notice_at_fin',~settings.notice_at_fin))
notice_set_check.grid(row=1, column=3, padx=5, pady=5)

# Conbobox setting
combobox_set_check = ttk.Checkbutton(
    file_setting_group, text='关闭多角色下拉框选择', command=lambda: settings.__setattr__('prefer_combobox',not settings.prefer_combobox))
combobox_set_check.grid(row=1, column=4, padx=5, pady=5)

auto_clean_radio = ttk.Radiobutton(
    file_setting_group, text='关闭自动Clean', value=0,variable=auto_cleaner)
auto_clean_radio.grid(row=1, column=5, padx=5, pady=5)

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
                         command=lambda:settings.__setattr__('vits_length_scale',vits_ls.get()) )
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


def diff_svc_infer(model_path, input_files: list, out_path, tran=0, acc=20, spk_id=0, auto_key=False):
    global diff_svc_model, run_clip
    diff_svc_cfg = '/'.join(model_path.split('/')[:-1])+'/config.yaml'

    if not init_state['svc']:
        from diff_svc.infer_tools.infer_tool import Svc
        from diff_svc.infer import run_clip
        diff_svc_model = Svc('', diff_svc_cfg, False, model_path)
        init_state['svc'] = True

    for i, input_file in enumerate(input_files):
        if not settings.enable_custom_filename:
            # final_out_path = os.path.join(out_path, f'{input_file.split("/")[-1][:-4]}_key_{tran}_{i}')
            final_out_path = os.path.join(out_path, f'{os.path.split(input_file)[-1][:-4]}_key_{tran}_{i}')
        else:
            final_out_path = out_path
        final_out_path += '.flac' if settings.svc_flac_output else '.wav'
        # 是否使用 crepe tiny模型，通过run_clip传入hparams,最后get pitch crepe读取配置
        run_clip(raw_audio_path=input_file, svc_model=diff_svc_model, key=tran, acc=acc, use_crepe=settings.use_crepe,
                 spk_id=int(spk_id), auto_key=auto_key, out_path=final_out_path, use_crepe_tiny=settings.use_crepe_tiny)
    notice()


# diff svc gui
diff_svc_tab = ttk.Frame(nb)

dsvc_mp_label = ttk.Label(diff_svc_tab, text='Diff-SVC模型：')
dsvc_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15, 5))


dsvc_mp = ttk.Entry(diff_svc_tab)
dsvc_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=global_entry_ipadx)

dsvc_mp_btn = ttk.Button(diff_svc_tab,  text="浏览文件", bootstyle=(
    INFO, OUTLINE), command=lambda: file_locate(dsvc_mp, reset_model='svc',check_speakers='svc'))
dsvc_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# output path select
dsvc_out_label = ttk.Label(diff_svc_tab, text='输出目录：')
dsvc_out_label.grid(row=2, column=1, padx=(20, 5), pady=5)


dsvc_out = ttk.Entry(diff_svc_tab)
dsvc_out.grid(row=2, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

dsvc_out_btn = ttk.Button(diff_svc_tab, text="浏览目录", bootstyle=(
    INFO, OUTLINE), command=lambda: save_locate(dsvc_out))
dsvc_out_btn.grid(row=2, column=3, padx=5, pady=5)

# raw audio select
dsvc_input_label = ttk.Label(diff_svc_tab, text='待转换音频：')
dsvc_input_label.grid(row=3, column=1, padx=(20, 5), pady=5)

dsvc_input = ttk.Entry(diff_svc_tab)
dsvc_input.grid(row=3, column=2, padx=20, pady=5, ipadx=global_entry_ipadx)

# 点炒饭禁止！
dsvc_input_btn = ttk.Button(diff_svc_tab, text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda: file_locate(
    dsvc_input, file_types=(("wav files", "*.wav"), ("ogg files", "*.ogg"),), locate_files=settings.enable_batch_out))
dsvc_input_btn.grid(row=3, column=3, padx=5, pady=5)

# 变调，Crepe，Auto Key

dsvc_pitch_label = ttk.Label(diff_svc_tab, text='升降半音|支持正负整数:')
dsvc_pitch_label.grid(row=4, column=1, padx=(5, 5), pady=5)


dsvc_pitch = ttk.Entry(diff_svc_tab, width=10)
dsvc_pitch.grid(row=4, column=2, padx=20, pady=5, ipadx=0, sticky='w')
dsvc_pitch.insert(index=0, string="0")

dsvc_check_y = [140,180,225] if ScaleFactor == 100 else [155,195,245]
dsvc_crepe_check = ttk.Checkbutton(
    diff_svc_tab, text='启用Crepe[耗时较长|1:8]', command=lambda: settings.__setattr__('use_crepe',~settings.use_crepe))
dsvc_crepe_check.place(x=350, y=dsvc_check_y[0])
# Crepe Tiny
dsvc_crepe_tiny_check = ttk.Checkbutton(
    diff_svc_tab, text='Crepe轻量模式[耗时更短|4:1]', command=lambda: settings.__setattr__('use_crepe_tiny',~settings.use_crepe_tiny))
dsvc_crepe_tiny_check.place(x=350, y=dsvc_check_y[1])

dsvc_auto_key_check = ttk.Checkbutton(
    diff_svc_tab, text='启用自适应变调', command=lambda: settings.__setattr__('svc_auto_key',~settings.svc_auto_key))
dsvc_auto_key_check.place(x=350, y=dsvc_check_y[2])

dsvc_flac_check = ttk.Checkbutton(
    diff_svc_tab, text='保存为Flac', command=lambda: settings.__setattr__('svc_flac_output',~settings.svc_flac_output))
dsvc_flac_check.place(x=530, y=dsvc_check_y[2])
# 加速
dsvc_acc_label = ttk.Label(diff_svc_tab, text='加速倍率[推荐20]：')
dsvc_acc_label.grid(row=5, column=1, padx=(5, 5), pady=5)

dsvc_acc = ttk.Entry(diff_svc_tab, width=10)
dsvc_acc.grid(row=5, column=2, padx=20, pady=5, ipadx=0, sticky='w')
dsvc_acc.insert(0, '20')

# Multi Speakers
dsvc_spk_label = ttk.Label(diff_svc_tab, text='角色ID：')
dsvc_spk_label.grid(row=6, column=1, padx=(5, 5), pady=5)

dsvc_spk = ttk.Entry(diff_svc_tab, width=10)
dsvc_spk.grid(row=6, column=2, padx=20, pady=5, ipadx=0, sticky='w')



dsvc_input_btn = ttk.Button(diff_svc_tab, text="转换音频", bootstyle=(SECONDARY, OUTLINE),
    command=lambda: diff_svc_infer(model_path=dsvc_mp.get(), input_files=dsvc_input.get().split("|"),
        tran=int(dsvc_pitch.get()), out_path=dsvc_out.get(), acc=int(dsvc_acc.get()),spk_id=svc_get_speaker(),auto_key=settings.svc_auto_key))
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
