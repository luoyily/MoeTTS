# MoeTTS
Speech synthesis model repo for galgame characters based on Tacotron2 , Hifigan and VITS. The repo is also used to publish precompiled GUI.

[简体中文](README.md)   [English](README_en.md)

## About

这是一个存放基于Tacotron2，Hifigan，VITS的galgame角色语音合成的模型库的仓库。另外也用于发行编译后的推理GUI。

hifi-gan: https://github.com/jik876/hifi-gan

Tacotron2: https://github.com/NVIDIA/tacotron2

VITS:https://github.com/jaywalnut310/vits

## 用户协议

在开源协议的基础上还请遵守以下几点：

1. （重要）不得将本软件、本仓库提供的预训练模型以及语音合成结果用于直接商业目的（例如：带有付费功能的QQ机器人，直接贩售，商业游戏等）对原作（模型数据来源作品）进行二次创作不包含在内。
2. 二次创作请遵守原作用户协议，请勿制作会对原作造成不良影响的内容，另外请注明你的作品使用了语音合成，避免造成误会。

## 使用方法

### 模型目录格式

1. 单模型可以放在任意位置，如果模型带有配置文件，请将它重命名为`config.json`并与TTS模型放置在同一目录。（例如hifigan，vits模型，它们是带有配置文件的）
2. **VITS模型请将`config.json`中的cleaners 改为`custom_cleaners`**

### 文本输入格式

文本一般是输入音素（日语在这里应该输入罗马音），但具体要看模型训练者的数据是怎么输入的。比如我的ATRI模型(Tacotron2版本)是输入无空格罗马音，标点符号只支持逗号句号。

### 自定义Cleaner与Symbols

你可以在与`moetts.exe`同级的目录下找到`custom`文件夹，这里面存放了两种模型的文本模块。

1. 自定义cleaner：找到`cleaners.py`并修改`custom_cleaners`函数即可（软件默认只会移除不在symbols中的字符，不对文本做进一步处理）
2. 自定义symbols：找到`symbols.py`，将里面的符号为你需要的符号

**注意：不同模型可能使用不同的cleaners与symbols训练，有需要请修改他们，保证模型能正常使用。**

### GUI使用方法

![tacotron2](assets/tacotron2.png)

选择您的模型路径与输出目录，最后输入待合成文本，点击`合成语音`等待一会软件会将音频输出到`输出目录/outpus.wav`

注意事项：

 1. 首次合成需要加载模型，耗时较长，相同模型再次合成不会再次加载，直接合成。
 2. 如果切换模型，再次合成会重新加载。
 3. 如果修改cleaners与symbols，重新启动软件后才能生效。
 4. 软件为64位版本，不支持32位系统。

VITS特殊说明

![vits](assets/vits.png)

1. VITS-Single，VITS-Multi分别为单角色模型与多角色模型
2. VITS-Multi中的原角色ID即待合成语音的角色ID，需要填入数字，目标角色ID为语音迁移功能的待迁移目标角色ID。
3. **待迁移音频需要22050的采样率，16位，单声道。**

## 模型下载

提交格式参考：

x. 模型名

描述：角色xxx的模型

模型输入：

下载地址：

配套Hifigan模型下载：选填

详细信息：选填

模型类型：



1. ATRI

   描述：游戏 ATRI- My dear moments 的角色ATRI模型

   模型输入：无空格罗马音，标点可保留英文逗号句号。例如：`tozendesu.koseinodesukara.`

   下载地址：链接：https://pan.baidu.com/s/1hJIbIX0r1UpI3UEtsp-6EA?pwd=jdi4 提取码：jdi4

   配套HifiGAN模型下载：链接：https://pan.baidu.com/s/1PGU8XEs5wy4ppJL6GjTgMQ?pwd=24g8 提取码：24g8
   
   详细信息：从游戏中选择1300条语音训练，训练约600 Epoch，训练时罗马音使用Bing翻译API
   
   模型类型：Tacotron2+Hifigan

2. ATRI-VITS

   描述：游戏 ATRI- My dear moments 的角色ATRI模型

   模型输入：带有分词与调形标注的罗马音，标点符号可使用`,.?!`，工具箱版本带有此转换功能。

   下载地址：链接：https://pan.baidu.com/s/1ThJPo4b7X9j5C4Ap3YFWEQ?pwd=yl7o  提取码：yl7o

   详细信息：使用VITS训练约 900 Epoch。

   模型类型：VITS

## 常见QA

1. Q：这个GUI能使用非官方Tacotron2或VITS训练的模型吗？

   A：如果模型结构与推理方式没改过的话，只是数据处理不同，应该是没问题的。

2. Q：是否有命令行版本或HttpApi？

   A：考虑到CLI与HttpApi容易遭到滥用，暂时延期开发。（1.0.0版本由用户**[ShiroDoMain](https://github.com/ShiroDoMain)**制作了CLI版本，需要可到cli分支获取）

3. Q：如何获得完整代码？

   A：完整代码请参见dev或其它分支，main分支用于发布模型与GUI。

4. Q：关于上传模型到其他平台

   A：暂时不打算上传

5. Q：如何训练自己的模型？

   A：本仓库不提供自训练支持，请到本项目使用到的各个原项目中查看帮助。

## 分享模型&参与开发

欢迎分享你的预训练模型，由于模型较大，暂时不打算存放在GitHub，可以拉取该项目后将你的模型下载地址以及信息写在Readme的模型下载部分中。提交PR即可。

如果有任何优化建议或者BUG可以提issue。

如您希望为项目追加新功能并合并到dev分支，请先查看Projects页面，避免PR冲突。

## 感谢名单&项目贡献者

1. **[ShiroDoMain](https://github.com/ShiroDoMain)**：1.0.0 CLI版本开发
2. **[menproject](https://github.com/menproject)**：1.0.0 版本英语自述文件翻译

3. **[CjangCjengh](https://github.com/CjangCjengh/)**：提供编译的g2p工具以及适用于日语调形标注的符号文件。