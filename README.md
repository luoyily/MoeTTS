# MoeTTS
Speech synthesis model repo for galgame characters based on Tacotron2 and Hifigan

简体中文   English(Todo..)

## 关于

这个是一个用于存放基于Tacotron2与Hifigan的galgame角色语音合成模型库.

hifi-gan: https://github.com/jik876/hifi-gan

Tacotron2: https://github.com/NVIDIA/tacotron2

##  参数说明
- use_cuda: 是否使用GPU加速，默认cpu  
- tacotron2_checkpoint: tacotron2模型的保存路径  
- hifi_gan_checkpoint: hifi-gan的保存路径  
- hifi_gan_config: hifi-gan的配置文件路径  
- input_text: 待转换的文本  
- output: 语音输出路径
- cleaners: 数据预处理的方法，默认是no_cleaners,根据模型数据预处理灵活选择  

示例：

```python3
python3 main.py --tacotron2_checkpoint .\models\atri_v2_40000.pt --hifi_gan_checkpoint .\models\g_atri_hifigan_02510000.pt --hifi_gan_config .\models\config.json --input_text tozendesu.koseinodesukara. --output .\output --cleaners basic_cleaners
```

## 模型下载

提交格式参考：

x. 模型名

描述：角色xxx的模型

模型输入：

下载地址：

配套Hifigan模型下载：选填

详细信息：选填



1. ATRI

   描述：游戏 ATRI- My dear moments 的角色ATRI模型

   模型输入：无空格罗马音，标点可保留英文逗号句号。Cleaners选择basic_cleaners。例如：`tozendesu.koseinodesukara.`

   下载地址：链接：https://pan.baidu.com/s/1hJIbIX0r1UpI3UEtsp-6EA?pwd=jdi4 提取码：jdi4

   配套HifiGAN模型下载：链接：https://pan.baidu.com/s/1PGU8XEs5wy4ppJL6GjTgMQ?pwd=24g8 提取码：24g8
   
   详细信息：从游戏中选择1300条语音训练，训练约600 Epoch

## 常见QA

1. Q：能使用非官方Tacotron2训练的模型吗？

   A：如果模型结构与推理方式没改过的话，只是数据处理不同，应该是没问题的。

2. Q：文本该输入什么？

   A：文本一般是输入音素（日语在这里应该输入罗马音），但具体要看模型训练者的数据是怎么输入的。比如我的ATRI模型是输入无空格罗马音，标点符号只支持逗号句号。

   文本（text）会被直接送到这里经过basic_cleaners然后转为序列用于推理，如果你使用的模型训练者做了其他预处理，请把预处理结果输入进去。

   ```python
   sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
   sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()
   ```

   

## 分享模型&参与开发

欢迎分享你的预训练模型，由于模型较大，暂时不打算存放在GitHub，可以拉取该项目后将你的模型下载地址以及信息写在Readme的模型下载部分中。提交PR即可。

如果有任何优化建议或者BUG可以提issue。