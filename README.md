# MoeTTS
Speech synthesis model repo for galgame characters based on Tacotron2 and Hifigan. The repo is also used to publish precompiled GUI.

[简体中文](README.md)   [English](README_en.md)

## About

这是一个存放基于Tacotron2与Hifigan的galgame角色语音合成的模型库的仓库。另外也用于发行编译后的推理GUI。

hifi-gan: https://github.com/jik876/hifi-gan

Tacotron2: https://github.com/NVIDIA/tacotron2

## 使用方法

![软件界面](assets/start.png)

打开软件后，分别选择您的模型路径与输出目录，选择Cleaners（看模型作者要求选），最后输入待合成文本，点击`合成语音`等待一会软件会将音频输出到`输出目录/outpus.wav`

注意事项：

 1. 首次合成需要加载模型，耗时较长，相同模型再次合成不会再次加载，直接合成。
 2. 如果切换模型，再次合成会重新加载。
 3. Hifigan的config.json需要放置在与模型同路径下。
 4. 软件为64位版本，不支持32位系统。

示例：

![示例](assets/example.png)

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

1. Q：这个GUI能使用非官方Tacotron2训练的模型吗？

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
