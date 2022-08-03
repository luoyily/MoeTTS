import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
import os
from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

is_init_model= False
cleaner_name = 'no_cleaners'

def set_cleaner(name):
    global cleaner_name
    cleaner_name = name

def file_locate(entry_box):
    global is_init_model
    file = filedialog.askopenfilename(initialdir=os.getcwd())
    entry_box.insert(0, file)
    # 重新选择模型后，重置初始化状态
    is_init_model= False
    print(file)

def directory_locate(entry_box):
    dire = filedialog.askdirectory(initialdir=os.getcwd())
    entry_box.insert(0, dire)
    print(dire)



def inference(tts_model, hifigan_model, target_text, output):
    global is_init_model, model, generator, np, torch, text_to_sequence, json, os, sys
    
    def synthesis():
        global label_img, img
        text = target_text
        sequence = np.array(text_to_sequence(text, [cleaner_name]))[None, :]
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
        label_img = ttk.Label(root, image=img)
        label_img.grid(row=5, column=2, padx=0, pady=(40,5))

        from scipy.io.wavfile import write
        with torch.no_grad():
            raw_audio = generator(mel_outputs.float())
            audio = raw_audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')
            write(os.path.join(output,'output.wav'), h.sampling_rate, audio)

    if not is_init_model:
        import numpy as np
        import torch
        from hparams import create_hparams

        from train import load_model
        from text import text_to_sequence
        hparams = create_hparams()
        hparams.sampling_rate = 22050

        checkpoint_path = tts_model

        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
        _ = model.cpu().eval()

        import json
        import os
        import sys
        sys.path.append('hifigan/')
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
    

root = tk.Tk()
root.geometry("720x420")
root.title('MoeTTS-CPU')
root.iconbitmap('./favicon.ico')
style = ttk.Style("minty")

# Tacotron2 model path select
tts_mp_label = ttk.Label(text='Tacotron2 模型：')
tts_mp_label.grid(row=1, column=1, padx=(20, 5), pady=(15,5))


tts_mp = ttk.Entry()
tts_mp.grid(row=1, column=2, padx=20, pady=(15, 5), ipadx=130)

tts_mp_btn = ttk.Button(root, text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(tts_mp))
tts_mp_btn.grid(row=1, column=3, padx=5, pady=(15, 5))

# hifigan model path select
hg_mp_label = ttk.Label(text='HifiGAN 模型：')
hg_mp_label.grid(row=2, column=1, padx=(20, 5), pady=5)


hg_mp = ttk.Entry()
hg_mp.grid(row=2, column=2, padx=20, pady=5, ipadx=130)

hg_mp_btn = ttk.Button(root, text="浏览文件", bootstyle=(INFO, OUTLINE), command=lambda :file_locate(hg_mp))
hg_mp_btn.grid(row=2, column=3, padx=5, pady=5)

# output path select
out_label = ttk.Label(text='输出目录：')
out_label.grid(row=3, column=1, padx=(20, 5), pady=5)


out = ttk.Entry()
out.grid(row=3, column=2, padx=20, pady=5, ipadx=130)

out_btn = ttk.Button(root, text="浏览目录", bootstyle=(INFO, OUTLINE), command=lambda :directory_locate(out))
out_btn.grid(row=3, column=3, padx=5, pady=5)

# text input
text_label = ttk.Label(text='待合成文本：')
text_label.grid(row=4, column=1, padx=(20, 5), pady=5)

# def test():
#     global label_img, img
#     print(cleaner_name)
#     img = ImageTk.PhotoImage(Image.open('test.png'))
#     label_img = ttk.Label(image=img)
#     label_img.grid(row=5, column=2, padx=0, pady=(40,5))

target_text = ttk.Entry()
target_text.grid(row=4, column=2, padx=20, pady=5, ipadx=130)
text_btn = ttk.Button(root, text="合成语音", bootstyle=(SECONDARY, OUTLINE), 
    command=lambda :inference(tts_mp.get(), hg_mp.get(), target_text.get(), out.get()))
    #lambda :inference(tts_mp.get(), hg_mp.get(), target_text.get(), out.get())
text_btn.grid(row=4, column=3, padx=5, pady=5)

# radio button for cleaners

text_label = ttk.Label(text='Cleaners:')
text_label.place(x=35, y=180)
radio1 = ttk.Radiobutton(root, text="no cleaner", value=1, command=lambda:set_cleaner('no_cleaners'))
radio1.place(x=150, y=180)
radio2 = ttk.Radiobutton(root, text="basic cleaner", value=2, command=lambda:set_cleaner('basic_cleaners'))
radio2.place(x=260, y=180)
radio3 = ttk.Radiobutton(root, text="trans cleaner", value=3, command=lambda:set_cleaner('transliteration_cleaners'))
radio3.place(x=370, y=180)
radio4 = ttk.Radiobutton(root, text="en cleaner", value=4, command=lambda:set_cleaner('english_cleaners'))
radio4.place(x=480, y=180)
# Draw result
res_label = ttk.Label(text='Mel out:')
res_label.grid(row=5, column=1, padx=5, pady=(100,5))
root.resizable(0,0)
root.mainloop()
# compile
# pyinstaller -i favicon.ico --noconsole 