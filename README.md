# MoeTTS
Speech synthesis model repo for galgame characters based on Tacotron2 , Hifigan and VITS. The repo is also used to publish precompiled CLI.

[简体中文](README.md)   ~~English~~

## About

这是一个存放基于Tacotron2，Hifigan，VITS的galgame角色语音合成的模型库的仓库。另外也用于发行编译后的推理GUI。

hifi-gan: https://github.com/jik876/hifi-gan

Tacotron2: https://github.com/NVIDIA/tacotron2

VITS:https://github.com/jaywalnut310/vits

## 用户协议

在开源协议的基础上还请遵守以下几点：

1. （重要）不得将本软件、本仓库提供的预训练模型以及语音合成结果用于直接商业目的（例如：QQ机器人，直接贩售，商业游戏等）对原作（模型数据来源作品）进行二次创作不包含在内。
2. 二次创作请遵守原作用户协议，请勿制作会对原作造成不良影响的内容，另外请注明你的作品使用了语音合成，避免造成误会。

## 使用方法

### 参数详解  
> -use_cuda bool 是否使用gpu加速,默认为cpu  
> -tt2ck str tacotron2模型存放路径  
> -hgck str hifi_gan模型存放路径  
> -hgc str hifi_gan配置文件存放路径  
> -i str 待合成的文本  
> -f str 待合成的文件  
> -o str 输出路径  
> -p str 文本预处理格式,默认为no_cleaners

### 文本输入格式  

支持日语输入,标点符号只支持逗号句号。  

### cli使用方法
这里以我将tacotron2模型,hifi-gan模型以及hifi-gan的配置文件都放在了models文件夹里为准

1. 安装依赖
```shell
pip install -r requirements.txt
```
- ps: 可能会出现部分包安装失败，比如tf，只要单独拧出来安装就行了  
2. 转换语音有两种方式，一种是直接转换一句话，另一种是将文本里的所有句子都转换  
```shell
# 单句转换
python main.py -tt2ck ./models/atri_v2_40000.pt -hgck ./models/g_atri_hifigan_02510000.pt -hgc ./models/config.json -i 私は高性能です. -o output -p basic_cleaners

# 文本转换,所有要转换的都存放在了test.txt文件里,每行对应一个语音输出  
python main.py -tt2ck ./models/atri_v2_40000.pt -hgck ./models/g_atri_hifigan_02510000.pt -hgc ./models/config.json -f ./test.txt -o output -p basic_cleaners
```


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

## 常见QA

1. Q：能使用非官方Tacotron2或VITS训练的模型吗？

   A：如果模型结构与推理方式没改过的话，只是数据处理不同，应该是没问题的。

3. Q：如何获得完整代码？

   A：完整代码请参见dev或其它分支，main分支用于发布模型与GUI。

4. Q：关于上传模型到其他平台

   A：暂时不打算上传

5. Q：如何训练自己的模型？

   A：本仓库不提供自训练支持，请到本项目使用到的各个原项目中查看帮助。

## 分享模型&参与开发

欢迎分享你的预训练模型，由于模型较大，暂时不打算存放在GitHub，可以拉取该项目后将你的模型下载地址以及信息写在Readme的模型下载部分中。提交PR即可。

如果有任何优化建议或者BUG可以提issue。

## 感谢名单&项目贡献者

1. **[ShiroDoMain](https://github.com/ShiroDoMain)**：1.0.0 CLI版本开发
2. **[menproject](https://github.com/menproject)**：1.0.0 版本英语自述文件翻译

3. **[CjangCjengh](https://github.com/CjangCjengh/)**：提供编译的g2p工具以及适用于日语调形标注的符号文件。