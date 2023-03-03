# MoeTTS
Speech synthesis model repo for galgame characters based on Tacotron2 , Hifigan and VITS. The repo is also used to publish precompiled GUI.

**Notice ：** This project is only for AI study and hobby, because of the like of the characters to develop this project, no malicious purpose. If there is infringement, please submit issues, we will immediately remove the associated model.

[简体中文](README.md)   [English](README_en.md)

## About

这是一个存放基于Tacotron2，Hifigan，VITS，Diff-SVC的galgame角色语音合成的模型库的仓库。另外也用于发行编译后的推理GUI。

停止维护通知：GUI功能维护已较为完善，此项目后续将不再维护。

## 近期更新

1.3.0：

> 1. 增加openvpi版diff svc,原版diff svc 24000模型,带fs模型不再支持，需要请下载1.2.5版本
> 2. 优化GUI设置相关代码
> 3. VITS支持发送至diff svc（语音合成完毕后交给diff svc进行语音转换）
> 4. 修改cleaners逻辑，现在可以选择自动clean（待输入文本clean后再合成）
> 5. 多人模型可以使用下拉列表选择说话人
> 6. 修复diff svc界面高DPI 下控件显示错位
> 7. 更新配置文件格式

1.2.5：

> 1. 规范diff_svc import
> 2. 精简VITS与Tacotron2代码并移除了matplotlib, tensorflow依赖
> 3. 删除不必要导入及依赖，加快启动速度
> 4. diff svc增加Crepe轻量模型
> 5. 使用两种编译器（pyinstaller, nuitka）
> 6. diff svc音频处理采样类型换为fft

1.2.4：

>1. 增加最近模型与输出路径记录，方便下次打开继续。
>2. 增加完成通知，可以在设置中打开（仅win10）。
>3. 更新diff-svc(12-04版)
>4. 内置了Openvpi的nsf_hifigan权重 (详见：https://openvpi.github.io/vocoders/)

1.2.3-beta:

> 1. 更新diff-svc(同步diffsvc原项目：支持nsf hifigan,增加Crepe缓存，修复了一些BUG)
> 2. 11-22：BUG通知，加载输入音频会覆盖原始wav，记得备份。配置文件开启UV可能导致呼吸声与空白异常。下版本修复。

GPU版请见本仓库“gpu”分支（此分支更新滞后于cpu版）。代码见dev分支。

## 用户协议

在开源协议的基础上还请遵守以下几点：

1. （重要）不得将本软件、本仓库提供的预训练模型以及衍生产物用于商业目的（例如：带有付费功能的QQ机器人，直接贩售，商业游戏等）对原作进行二次创作不包含在内。
2. 二次创作请遵守原作用户协议，请勿制作会对原作造成不良影响的内容，另外请注明你的作品使用了语音合成，避免造成误会。
3. 本仓库提供的预训练模型及数据部分来自社区，使用造成的一切后果由使用者承担，与本仓库作者及贡献者无关。
4. 禁止使用本仓库任何内容（包括但不限于代码，编译后的exe，预训练模型）进行原创游戏制作。

本项目态度：

1. 本项目仅鼓励合理范围内对原作的二次创作，反对任何对原作以及相关行业的侮辱等不当行为。

## 使用方法

### 模型目录及配置文件格式

1. 单模型可以放在任意位置，如果模型带有配置文件，请将它重命名为`config.json`（diff-svc请重命名为`config.yaml`）并与模型放置在同一目录。

2. **（TTS模型配置）**使用TTS模型前，请编写一个简单的配置文件，并将它命名为`moetts.json`与你的TTS模型放在同一目录。

   **注：如果你使用的不是本仓库提供的模型，那么此步骤是必须的**。

   以下为多人模型配置示例（单人只需要symbols）：

   ```json
   {
   	"symbols":["_", ",", ".", "!", "?", "-", "A", "E", "I", "N", "O", "Q", "U", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "m", "n", "o", "p", "r", "s", "t", "u", "v", "w", "y", "z", "\u0283", "\u02a7", "\u2193", "\u2191", " "],
       "speakers":{
   		"杏璃":0,
   		"杏铃":1,
   		"Apeiria":2,
   		"明日香":3,
   		"ATRI":4,
   		"艾拉":5,
   		"彩音":6,
   		"星奏":7,
   		"由依":8,
   		"冰织":9,
   		"真白":10,
   		"美绘瑠":11,
   		"二阶堂真红":12
   	}
   }
   
   ```

   注：**此配置不是训练模型使用的config.json**,是用于指定您的模型训练时所使用的symbols，例如VITS，您可以在`vits/text/symbols.py`中找到使用的symbols，并将它按以上格式保存为json。

### GUI使用方法

#### TTS(tacotron2,vits)

![VITS界面截图](./assets/vits.png)

说明：

1. 角色ID：用于指定多人模型的角色。如果配置文件中含有角色表会优先加载下拉框供选择，你可以在设置中关闭下拉框。配置文件未提供角色表即在此输入数字选择角色。单人模型留空即可。

2. 待合成文本：语音合成使用的文本（音素）。你可以将你的文本按模型输入要求clean后在此输入，也可以在工具箱中选择自动clean，在这直接输入原始文本即可。

   例（勾选自动转换后直接输入原始文本）:

![ToolBox界面截图](./assets/tools.png)

![待合成文本](./assets/text_input.png)

3. 待迁移音频：VITS 多人模型提供的语音转换功能，可以对模型中一位角色的音频转为模型中的另一位角色。输入音频需要22050Hz,单声道wav。
4. 合成并发送至SVC：使用VITS合成语音后再使用diff svc转换音频。使用此功能需要在VITS和Diff-SVC界面中提前选择好模型及输出位置等。

#### diff-svc

![Diff-SVC界面截图](./assets/diff_svc.png)

参数说明：

1. 升降半音：默认为0，支持正负整数输入，单位为半音
2. 启用Crepe：该选项可降噪音频，启用后CPU耗时较高，约为原音频时长8倍，建议合成最终版本再开启，干净的音频无需开启。
3. Crepe轻量模式：在启用Crepe的前提下，勾选此选项后Crepe使用Tiny模型，耗时更短，约为原音频时长1/4。
4. 加速倍率：默认为20，耗时约1:3，预览可使用100，耗时约1：1（该设置会影响音频质量）
5. 待转换音频：wav或ogg纯人声音频，转换后为模型角色音色。
6. 自适应变调：自动评估适合的音域进行转换（需要配置文件包含相关信息）。
7. 角色ID：多人模型用，填入数字或使用下拉框选择。

## 在线Demo

Integrated into [Huggingface Spaces ](https://huggingface.co/spaces) using  [Gradio](https://github.com/gradio-app/gradio). Try it out [![Hugging  Face  Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/skytnt/moe-tts)

## 模型下载

### TTS：

| 模型名与下载                                                 | 类型              | 描述                                                         | 输入格式                                                     |
| ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ATRI](https://pan.baidu.com/s/1itDhrhzw6uZYxB2238BzTQ?pwd=0z3u) | Tacotron2+Hifigan | 游戏 ATRI- My dear moments 的角色ATRI模型                    | 无空格罗马音，标点可保留英文逗号句号。例如：`tozendesu.koseinodesukara.`也可使用工具箱日语g2p（普通转换） |
| [ATRI-VITS](https://pan.baidu.com/s/1_vhOx50OE5R4bE02ZMe9GA?pwd=9jo4) | VITS 单角色       | 游戏 ATRI- My dear moments 的角色ATRI模型                    | 工具箱中的日语g2p(调形标注+ts替换)                           |
| [Galgame 13位角色](https://pan.baidu.com/s/1anZ3eusmG8BVhrQQrvlkug?pwd=0i9r)     [1.3.0配置文件](https://pan.baidu.com/s/1RMscvHnJgn5w0Vi7RRW9sg?pwd=a947) | VITS 多角色       | 包含以下角色（游戏名略）：0 杏璃 1 杏铃 2 Apeiria 3 明日香 4 ATRI 5 艾拉 6 彩音 7 星奏 8 由依 9 冰织 10 真白 11 美绘瑠 12 二阶堂真红 | 工具箱中的日语g2p(调形标注+ts替换)                           |
| [Mori](https://pan.baidu.com/s/1ekfczPolJRNjqVDNgjTJSA?pwd=rm1v) | VITS 单角色       | 游戏 Fox Hime Zero 角色 茉莉                                 | 工具箱中的日语g2p(调形标注)                                  |

### Diff-svc

| 模型名称与下载                                               | 备注                         | 贡献者                                |
| ------------------------------------------------------------ | ---------------------------- | ------------------------------------- |
| [姬野星奏](https://pan.baidu.com/s/1vc7lLpyAjUDCKI_PO5CR6w?pwd=wad5) | 24000Hz(仅包含推理)          | luoyily                               |
| [小鞠由依](https://pan.baidu.com/s/1WwluFplMLjVD9ZeF6qdAxQ?pwd=i4yc) | 24000Hz(仅包含推理)          | luoyily                               |
| [ATRI](https://pan.baidu.com/s/1-jc9DSQp_fOv-kdc_4bkyQ?pwd=3jm3) | 24000Hz(仅包含推理)          | [RiceCake](https://github.com/gak123) |
| [鹰仓杏铃](https://pan.baidu.com/s/1aBL3geIXJmb7dfEzi4AF5g?pwd=86aq) | 24000Hz(仅包含推理)          | luoyily                               |
| [悠音](https://pan.baidu.com/s/18cG-DX38V8LrnFqy83Mzaw?pwd=riwm) | 24000Hz(仅包含推理)          | luoyily                               |
| [锦 明日海](https://pan.baidu.com/s/1ZNMn0hRu2MhNeLwQ9_VZ3Q?pwd=xe57) | 24000Hz(仅包含推理)          | luoyily                               |
| [白咲美绘瑠](https://pan.baidu.com/s/1lIXXxiAKShxoLkaXOJCdsw?pwd=7c1v) | 24000Hz(仅包含推理)          | luoyily                               |
| [艾拉(可塑性记忆)](https://pan.baidu.com/s/14mQfBOAIllqqanP03c5w0g?pwd=z1im) | 24000Hz(仅包含推理)          | luoyily                               |
| [伊莉雅](https://pan.baidu.com/s/14v-XkNtp8pDqXXVjljYFSw?pwd=mvh1) | 24000Hz(仅包含推理)          | luoyily                               |
| [ATRI_44100](https://pan.baidu.com/s/1KV_SiWUdBZjayyUioPhF3w?pwd=q515) | 44100Hz(仅包含推理)          | [RiceCake](https://github.com/gak123) |
| [新海 天](https://pan.baidu.com/s/1OYBMiHFPOm6fp-EV7JJWbA?pwd=iavq) | 44100Hz(仅包含推理)          | [RiceCake](https://github.com/gak123) |
| 茉莉 [百度网盘](https://pan.baidu.com/s/15FsM-NM6XXhGCb8MAzp05w?pwd=5mcg) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/mori) | 44100Hz(仅包含推理)          | luoyily                               |
| 仓科明日香 [百度网盘](https://pan.baidu.com/s/1saGS166ty3JvcwPa_xwKNw?pwd=hrqx) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/asuka441) | 44100Hz(完整权重+仅推理权重) | luoyily                               |
| [在原七海](https://pan.baidu.com/s/1sgshGvHc9Y8IGmSrgWTJHg?pwd=zr76) | 44100Hz(仅包含推理)          | [RiceCake](https://github.com/gak123) |
| [二阶堂真红](https://pan.baidu.com/s/1uBiAEr-YB4pi-W_cn3U3gg?pwd=yuf8) | 44100Hz(仅包含推理)          | [RiceCake](https://github.com/gak123) |
| 姬野星奏 [百度网盘](https://pan.baidu.com/s/1DZlPNTErn91Si4FPkplbEQ?pwd=k603) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/sena441) | 44100Hz(仅包含推理)          | luoyily                               |
| 白咲美绘瑠 [百度网盘](https://pan.baidu.com/s/1WJE5JY-rTBdPaId-nnY1Gg?pwd=kv72) [hugging face](https://huggingface.co/loyi/moetts/tree/main/diff_svc/mieru441) | 44100Hz(仅包含推理)          | luoyily                               |

注：以上diff-svc模型可能仅包含推理所必要的权重，您可能无法在他们的基础上直接继续训练。

## 常见QA

1. Q：这个GUI能使用非官方Tacotron2或VITS训练的模型吗？

   A：未修改过模型结构的可以使用，请参考使用方法中说明进行配置。（so-vits,emo-vits等修改版本不支持）

3. Q：如何获得完整代码？

   A：完整代码请参见dev或其它分支，main分支用于发布模型与GUI。

5. Q：如何训练自己的模型？

   A：本仓库不提供自训练支持，请到本项目使用到的各个原项目中查看帮助。

4. Q : 打不开，缺失DLL等？

   A：请安装常用运行库，如果依旧失败，可以使用cmd运行程序并提供尽可能详细的信息提交Issue。

## 分享模型&参与开发

此项目已暂停维护，不再添加新模型。

~~欢迎分享你的预训练模型，由于模型较大，暂时不打算存放在GitHub，可以拉取该项目后将你的模型下载地址以及信息写在Readme的模型下载部分中。提交PR即可。~~

~~如果有任何优化建议或者BUG可以提issue。~~

~~如您希望为项目追加新功能并合并到dev分支，请先查看Projects页面，避免PR冲突。~~

## 感谢名单&项目贡献者

1. **[ShiroDoMain](https://github.com/ShiroDoMain)**：1.0.0 CLI版本开发
2. **[menproject](https://github.com/menproject)**：1.0.0 版本英语自述文件翻译
3. **[CjangCjengh](https://github.com/CjangCjengh/)**：提供编译的g2p工具以及适用于日语调形标注的符号文件。
4. **[skytnt](https://huggingface.co/skytnt)**: 提供了hugging face 在线 demo

## 参考&引用

hifi-gan: https://github.com/jik876/hifi-gan

Tacotron2: https://github.com/NVIDIA/tacotron2

VITS:https://github.com/jaywalnut310/vits

diff-svc:https://github.com/prophesier/diff-svc

DiffSinger:https://github.com/MoonInTheRiver/DiffSinger

DiffSinger(openvpi):https://github.com/openvpi/DiffSinger

DiffSinger 社区声码器企划：https://openvpi.github.io/vocoders/

diff svc(openvpi): https://github.com/openvpi/diff-svc