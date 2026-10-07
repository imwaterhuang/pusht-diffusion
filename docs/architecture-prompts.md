# 架构图生成提示词

## 修订记录与最终提示词

### 总览第一轮修订（未采用）

```text
Edit this supplied architecture diagram, preserving its Chinese typography, all correct tensor shapes, two alternative denoising backbones, inference/training separation and overall layout. Fix these precise technical errors: (1) The only output from the denoising network into the sampler is predicted noise. Keep the downward box 预测噪声 [B,16,2] and its arrow INTO 20步迭代采样, but REMOVE the label 更新的带噪动作 on that arrow. Remove the extra unlabeled direct horizontal network-to-sampler arrow. (2) ADD a clear feedback arrow leaving the TOP of the 20步迭代采样 box, routed above the denoiser to the 带噪动作 [B,16,2] input box. Label this feedback arrow 更新的带噪动作. Initial Gaussian noise initializes this loop. The output to future16actions is after20 iterations. (3) In the lower training panel add a small explicit input 观测条件 c 与扩散步 k with an arrow into 同一条件去噪网络; there are NO expert actions in the inference inputs. (4) Render on fully opaque pure white background across the entire canvas. No transparency, black margins, iridescent halo or mottled texture. Title and footer deep navy on white, crisp and readable. Keep all other content and don't invent any results or trained status.
```

### 总览最终重绘（采用）

```text
Create a clean, fully opaque white, landscape educational architecture chart. Large readable Chinese text. Navy text, teal observations, violet denoising, amber control. No gradients or decoration. Title: "Push-T 扩散策略：系统设计". Subtitle: "设计规格｜模型与正式训练代码由学习者实现".

Use TWO separate horizontal panels.

UPPER PANEL title "推理与闭环控制":
Left: input box containing only "两帧图像 [B,2,3,96,96]" and "两帧位置 [B,2,2]".
Arrow to "观测编码器" then arrow labelled "条件 c [B,516]" into a large container "迭代去噪：20 步".
Inside container draw TWO boxes side by side: left "噪声预测网络" and right "采样更新". Arrow from left to right labelled "预测噪声 [B,16,2]". Clear RETURN arrow from right back to left labelled "更新的带噪动作 [B,16,2]". A small top input "高斯噪声 [B,16,2]" initializes the action input of the LEFT network box. A second small top input "扩散步 k" enters LEFT network box. Observation condition enters LEFT network box. Under LEFT network write "U-Net 或 DiT（二选一）". Do NOT draw U-Net and DiT in series.
From the RIGHT edge of the large container, one arrow labelled "20 步结束" -> "未来16步动作" -> "反归一化" -> "执行前4步" -> "Push-T 环境".
A large bottom return arrow from ENVIRONMENT back to observations, labelled "逐步更新观测；每4步重新规划".

LOWER PANEL title "仅训练时使用":
"专家动作 [B,16,2]" and "随机噪声 ε、扩散步 k" -> "加噪" -> "同一噪声预测网络" -> "预测噪声".
Directly beneath the training network write "同时输入观测条件 c 与扩散步 k".
Then prediction arrow -> lossbox text "比较预测噪声与真实噪声 ε" and next line "仅统计有效动作位置".
Do NOT connect expert actions to the upper inference panel. Do NOT omit experts or add-noise block.

Readable bottom glossary "U-Net：U-shaped Network，U形网络    DiT：Diffusion Transformer，扩散Transformer".
Final short line "t 为环境时间，k 为扩散时间；首个动作是 a_t。"
Only render the requested labels. No success metrics, no robot task-language inputs, no transparent background. Network output must be labelled predicted NOISE. A sampler produces ACTIONS only after the loop.
```

### U-Net 解码块修订

```text
Make a precise small correction to this Chinese U-Net diagram, preserving the existing style and all correct content. Two required fixes: 1. In the RIGHT decoder at temporal length8, immediately AFTER 通道拼接 [B,512,8] and BEFORE the upsampling that goes to temporal length16, INSERT a clearly separate purple block reading 条件残差块 ×2 [B,256,8]（解码器）. The arrows MUST be concat512,8 → residualblock256,8 → upsample128,16 → concat256,16 → residualblock128,16. There is room to reflow the right-side nodes horizontally if needed. The bottom [B,256,8] before concat is the first upsample result; it cannot replace the missing residual block AFTER concat. 2. At top, add a solid arrow from 观测条件 c [B,516] to 条件拼接 [B,644]. Both c and the time128 embedding feed concat. Only the [B,644] concatenated condition emits the dotted condition rails to the FIVE residual stages (encoder128, encoder256, bottleneck512, decoder256, decoder128); do not send condition arrows to downsample or concat operators. Keep skip connections linking encoder256,8 to concat512,8 and encoder128,16 to concat256,16. No other architectural changes. Keep fully opaque white background and all labels legible.
```

### U-Net 条件路径最终修订（采用）

```text
Edit ONLY the top condition panel and its dotted outgoing arrows of this U-Net drawing. Preserve the entire main U-shaped path, BOTH skip connections, both decoder residual blocks, the bottom conditional-residual inset, title and footer exactly. Replace the three teal boxes in the top condition panel with ONE simple teal information box containing exactly these three lines: 条件向量 [B,644] / 观测条件 [B,516] 与扩散时间编码 [B,128] 拼接 / 所有条件残差块使用该条件向量. Remove ALL existing dotted teal arrows/rails that descend from the top panel into the main U shape, and remove the top-right paragraph that describes those dashed arrows. We intentionally use the written sentence instead of a crowded condition arrow bus. The condition injection is already shown accurately in the bottom inset; retain its [B,644] → FiLM arrow. The c input must NOT flow through the diffusion timestep encoder: representing concat in one text box avoids that false serial relationship. No changes outside the top panel/dotted condition rails. Keep opaque white background.
```

### DiT 条件说明最终修订（采用）

```text
Edit ONLY the small teal explanatory text box at the upper right of this DiT diagram. Keep every other diagram element, arrow, formula, tensor shape, title, and footer unchanged. Replace the content of that upper-right box with exactly: 条件包含： / • 两帧图像与推杆位置的编码 [B,516] / • 扩散步 k 的时间编码 / • 分别投影到256维后相加 / • 注入6个块和最终条件层归一化. Remove references to robot body states or task information; this Push-T model has images and pusher positions only, no language or task labels. No other edits. Keep fully opaque white background.
```


生成方式：内置 imagegen 工具，通过 imagegen 技能生成；图片为设计规格，不能作为实现或训练结果的证据。

以下保留提交给生成工具的原始提示词。生成结果仍须核查，最终核查见 architecture.md。

## 系统总览

```text
Use case: scientific-educational. Create a precise, beautiful Chinese technical architecture infographic for a learner-owned Push-T Diffusion Policy project. This is a design specification, NOT evidence of implementation. Landscape wide high-resolution layout, very legible Chinese typography and mathematical tensor dimensions. White/very light ivory background, restrained deep-navy text, teal observation pipeline, violet diffusion pipeline, amber physical control. Flat vector-like boxes and deliberate arrow routing, generous spacing, no decorative machinery, no meaningless icons. Title verbatim: "Push-T 扩散策略｜系统架构". Subtitle verbatim: "设计规格 · 模型与正式训练代码由学习者实现".

Main large inference band left-to-right:
1. Two simple overhead Push-T observation thumbnails labelled "前一帧 t−1" and "当前帧 t". Show gray T block, green T-shaped target, small blue circular pusher. Under the thumbnails put "图像 [B,2,3,96,96]" and a small parallel box "推杆位置 [B,2,2]".
2. Both input arrows enter "共享观测编码器" box. Small text "每帧视觉 256 维 + 位置 2 维". Output "条件 c [B,516]". Two frames share one visual encoder within a policy; separately trained policies do not share trained weights.
3. A central large outlined box "条件去噪网络（二选一）". Inside two equal side-by-side boxes "U-Net" with small full name "U-shaped Network" and "DiT" with full name "Diffusion Transformer". Between boxes write "或"; there is NO arrow between U-Net and DiT, they are alternatives, not serial and not an ensemble. Shared input to the enclosing box from c. Two other arrows into the enclosing box labelled "带噪动作 [B,16,2]" and "扩散步 k [B]". Output from enclosing box "预测噪声 [B,16,2]".
4. Beside the network box a sampler box "20 步迭代采样" with an explicit feedback arrow sending updated noisy actions back to the denoiser input. Small starting box "高斯噪声 [B,16,2]" enters only the start of the sampling loop. c computed once is reused across the loop.
5. End of sampling arrow -> "未来 16 步动作" -> "反归一化" -> "执行前 4 步" -> Push-T simulation thumbnail. Wide closed-loop arrow beneath returns from environment to observation input; caption "每个环境步更新历史；每 4 步重新规划".
Distinguish the time meanings at bottom of main band: "t：环境时间    k：扩散时间    B：批大小".

Separate slim lower panel with clearly dashed border and title "仅训练时使用". Flow "专家动作 A [B,16,2]" and "随机噪声 ε + 扩散步 k" -> "加噪" -> "同一条件去噪网络" -> "预测噪声 ε̂" -> "有效动作位置的均方误差". Feed true noise ε as target into loss and "有效位掩码 [B,16]" into loss. This training panel is explanatory and NOT connected as an input to inference; expert actions never appear in main inference inputs. Bottom small footer "输出是推杆二维绝对目标坐标；首个动作对应 a_t。"
Do not include success percentages, training results, software screenshots, logos or a legend claiming implemented. Ensure ALL arrow directions and tensor sizes above are correct. Avoid extraneous formulas or fine print. Use only requested major labels, prefer visual clarity over redundant text.
```

## U-Net 详图

```text
Use case: scientific-educational. Create a Chinese high-resolution landscape architecture diagram matching a clean educational series: ivory-white background, deep navy text, teal conditions, violet network blocks, amber output; flat polished vector-like design, large legible Chinese and tensors. Title "Push-T 扩散策略｜一维 U-Net". Subtitle "U-Net（U-shaped Network，U 形网络）· 设计规格，待学习者实现".

Draw a clear U-shaped symmetric temporal convolution architecture, not an image-segmentation network. Time length shrinks 16→8→4 then expands4→8→16. Main path:
"带噪动作 [B,16,2]" → small "转置" → "[B,2,16]"
→ encoder block "条件残差块 ×2" with size "[B,128,16]"
→ downsample "时间长度 ÷2"
→ encoder block "条件残差块 ×2" size "[B,256,8]"
→ downsample "时间长度 ÷2"
→ bottleneck block "条件残差块 ×2" size "[B,512,4]"
→ upsample block resulting "[B,256,8]"
→ concat node labeled "通道拼接" with a clearly routed long skip arrow from encoder [B,256,8], concat output "[B,512,8]"
→ decoder block "条件残差块 ×2" result "[B,256,8]"
→ upsample resulting "[B,128,16]"
→ concat node labelled "通道拼接", skip arrow from encoder [B,128,16], concat output "[B,256,16]"
→ decoder block "条件残差块 ×2" result "[B,128,16]"
→ "输出投影 + 转置" → "预测噪声 [B,16,2]".
Both skip connections MUST join features of the SAME temporal length, not raw input added to output. Arrange the two encoder stages descending on left, bottleneck lowcenter, decoder ascending on right; clear single main path arrows. Channels are the middle axis; label once "内部张量：[批大小, 通道数, 动作时间长度]".

A top condition rail outside the U:
"观测条件 c [B,516]" and "扩散步 k → 正弦编码与投影 [B,128]" -> "条件拼接 [B,644]". Thin dotted teal arrows from this condition rail to each conditional residual block, including bottleneck and decoder, NOT to concatenate sequence length.
Inset in lower empty area "条件残差块" with a small distinct diagram: x -> Conv1d/GroupNorm/Mish -> "条件缩放与偏移" -> Conv1d/GroupNorm/Mish -> "+" residual from x, using a 1×1 projection if channels differ. Include formula h′ = γ(c,k) ⊙ h + β(c,k), carefully distinguish this modulation from skip concat.
Small config panel text "卷积核 5 · 分组数 8 · 输出预测噪声" and "FiLM（Feature-wise Linear Modulation，逐特征线性调制）". Expand the new abbreviations in footer readable text: "Conv1d：一维卷积（One-dimensional Convolution）". GroupNorm and Mish are names not acronyms no expansion needed.
Bottom label "卷积沿动作时间轴运行；此图给出本项目建议的对称解码结构。"
No results, no success badges, no extra image generation stages, no image-patch inputs. Accuracy more important than density.
```

## DiT 详图

```text
Use case: scientific-educational. Draw a polished high-resolution landscape Chinese architecture infographic, third image in a matching educational series. Ivory-white background, deep navy typography, teal observations/conditions, violet model layers, amber noise output. Flat vector-like diagram with spacious mathematically correct arrows and readable text. Title "Push-T 扩散策略｜动作 DiT". Subtitle "DiT（Diffusion Transformer，扩散 Transformer）· 设计规格，待学习者实现".

Left-to-right main pipeline across upper half:
"带噪动作 [B,16,2]" -> "线性投影 2→256" -> "[B,16,256]" -> plus node receiving "固定正弦位置编码 [1,16,256]" -> "条件 Transformer 块 ×6" -> "条件层归一化" -> "线性投影 256→2" -> "预测噪声 [B,16,2]".
Represent16 time-step tokens as a short row of small coloured tiles with caption "一个 token 对应一个动作时刻". NO image patches. No causal attention triangle.
Above block stack condition pipeline:
"观测条件 [B,516]" -> "条件投影 [B,256]"
"扩散步 k [B]" -> "正弦编码 + 时间投影 [B,256]"
these two arrows -> a plus node -> "条件向量 [B,256]".
Dotted arrows from condition vector to every one of the 6 Transformer blocks and final conditional layernorm, explicitly not to action-token concatenation.

Lower half: one enlarged inset "单个条件 Transformer 块".
Main sequence [B,16,256]:
LayerNorm -> "条件缩放与偏移" -> "非因果多头自注意力" -> "条件门控" -> residual plus with original x.
Then LayerNorm -> "条件缩放与偏移" -> "前馈网络 256→1024→256" -> "条件门控" -> residual plus with intermediate x.
The condition vector enters a small projection module producing "shift、scale、gate" for each of two branches. Dotted arrows route clearly to both conditioning/modulation and gate nodes. Formulas in a compact sidebox: "调制：h′ = (1 + scale) ⊙ h + shift" and "残差：x′ = x + gate ⊙ F(h′)".
Small accompanying text: "adaLN-Zero（Adaptive Layer Normalization with Zero Initialization，零初始化自适应层归一化）". Note "条件调制末层、门控与最终输出投影采用零初始化规则".
Bottom parameter strip "隐藏维度 256 · 注意力头 8 · 块数 6 · 前馈宽度 1024".
Footer "环境时间 t 与扩散时间 k 分开编码；网络输出噪声，不直接输出环境动作。"
Do not add learned variance, classification head, class embedding, vision patch embedding, autoregressive mask, or training-success claims. Show shared tensor dimensions unchanged through residual blocks and token self-attention. All new acronyms above must have supplied full English names.
```
