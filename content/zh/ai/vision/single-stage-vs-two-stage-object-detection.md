---
title: "单阶段 vs 双阶段目标检测：从候选集合到 NMS 的工程算账"
subtitle: "两个关键点决定选型：候选数规模（anchors vs proposals）与正负样本不平衡（focal loss vs sampling）"
date: 2026-01-24T16:36:30+08:00
summary: "从工程视角系统对比 one-stage 与 two-stage 检测：把它们统一成‘生成候选→打分→去重’的流程，然后用可复制的数字（anchors 数量、top-k、NMS 最坏复杂度）解释速度差异，并用 focal loss vs 采样策略解释训练差异。文末提供纯 NumPy 可运行的 NMS 与候选规模算账代码，帮助你做选型与排查。"
description: "系统对比单阶段与双阶段目标检测的流程、复杂度与工程场景：候选集合规模、NMS/后处理成本、focal loss 与采样策略，并给出纯 NumPy 可运行示例用于算账与验证。"
categories: ["AI", "Vision"]
tags: ["object-detection", "one-stage", "two-stage", "yolo", "ssd", "retinanet", "faster-rcnn", "rpn", "nms", "focal-loss"]
keywords: ["目标检测", "单阶段", "双阶段", "YOLO", "SSD", "RetinaNet", "Faster R-CNN", "RPN", "NMS", "Focal Loss"]
readingTime: 15
draft: false
---

## 副标题 / 摘要

“单阶段更快、双阶段更准”这句话能帮你记忆，但很难帮你落地选型。
更工程化的表述是：**两者都在做“候选→打分→去重”，差别在“候选集合何时变小”与“训练如何对抗不平衡”**。

- **单阶段（one-stage）**：在特征图上做密集预测（anchors 或网格点），直接输出类别与框；靠 `score_threshold + top-k + NMS` 控制冗余，并常用 **focal loss** 对抗海量负样本。
- **双阶段（two-stage）**：先用 RPN/Proposals 把候选集合压到可控规模，再对少量 RoI 做更贵的分类与回归；训练时常用采样（如 1:3）让 batch 内梯度更均衡。

本文不追“模型史”，只做三件更能直接用在工程上的事：

1) 把两类方法统一成同一条流水线（候选→打分→去重），你能快速定位性能与误差来自哪里
2) 用可复制的数字把“快/慢”讲清楚（anchors 数量、top-k、NMS 复杂度）
3) 用 focal loss vs 采样策略把“训练为什么难/为什么稳”说清楚

- **预计阅读时长**：约 15 分钟  
- **标签**：`object-detection`、`one-stage`、`two-stage`、`nms`  
- **SEO 关键词**：目标检测, 单阶段, 双阶段, YOLO, Faster R-CNN  
- **元描述**：用候选集合规模与训练不平衡两条主线，对比单/双阶段检测并给出可运行算账代码。  

---

## 目标读者

- 想真正理解 one-stage/two-stage 差异，并能独立做选型的工程师
- 需要对“为什么这里慢”“为什么误检多/漏检多”做定位的实践者
- 已经知道 IoU/NMS 等基础概念，但缺少“体系化心智模型”的读者

---

## 背景 / 动机（为什么“候选集合大小”是第一性问题）

目标检测要输出多个 `(bbox, class, score)`。
不管你用 YOLO、SSD、RetinaNet 还是 Faster R-CNN，本质都逃不开同一件事：

> 你先得决定“要评估多少个候选框”，然后把它们排序、过滤、去重。

候选数量直接决定三类成本：

1) head 计算量（每个候选要算分类/回归）  
2) 后处理成本（尤其 NMS，最坏可到 $O(N^2)$）  
3) 训练不平衡程度（候选越多，负样本越多）

一个最常见的规模锚点（输入 `640×640`，FPN stride `8/16/32`，每点 3 anchors）：

- P3：`80×80×3 = 19200`
- P4：`40×40×3 = 4800`
- P5：`20×20×3 = 1200`

合计候选约：

$$
N_{anchors} \approx 25200
$$

因此工程问题不是“单阶段/双阶段谁更高级”，而是：

> 你愿意在推理时评估 2.5 万个候选，还是先把它缩到 1000 个 proposals 再精修？

---

## 快速掌握地图（60–120 秒）

- 问题形状：输入 `H×W` 图像 → 输出 `M` 个框（`M` 通常几十到几百）
- 核心一句话：两者都在做“候选→打分→去重”；差别是候选集合何时变小、训练如何对抗不平衡
- 什么时候用 one-stage：延迟/吞吐优先、部署在边缘端、容忍略低 AP
- 什么时候用 two-stage：误检代价高、难例/小目标更重要、对 AP 更敏感
- 复杂度抬头：one-stage 的候选规模常见 `O(HW×A)`；two-stage 把第二阶段规模压到 `O(P)`
- 常见失败模式：人群/密集小目标里，朴素 NMS 容易抑制掉真阳性（crowded scenes）

---

## 深挖重点（PDKH Ladder：本文只深挖两条主线）

为避免写成“检测模型百科”，本文只深挖两个概念（并走完 PDKH 的关键台阶）：

1) 候选集合规模 → top-k → NMS（速度直觉来自哪里）
2) 正负样本不平衡（one-stage 的 focal loss vs two-stage 的采样/两次过滤）

---

## 主心智模型：把检测统一成一条流水线

不管是 one-stage 还是 two-stage，你都可以抽象成：

1) 生成候选（candidates）：anchors 或 proposals
2) 对候选打分：分类 score + bbox 回归
3) 去重/过滤：阈值过滤、top-k、NMS
4) 输出最终集合：几十到几百个框

单阶段与双阶段最大的差别是：**候选集合“变小”的时机**。

- one-stage：候选从一开始就很大，靠后处理把它压到可用范围
- two-stage：先用 RPN 把候选压到中等规模，再用更贵的 RoI head 精修

这条流水线也提供了很实用的定位路径：

- 慢在 head：候选太多 / head 太重
- 慢在后处理：top-k 太大 / NMS 实现路径不佳
- 漏检多：候选召回不够（RPN 或 anchor 分配策略）
- 误检多：分类不够强（two-stage 往往更强）或 NMS 阈值不合适

---

## 核心概念与术语（含公式锚点）

### IoU（交并比）

$$
IoU(B_1, B_2) = \frac{|B_1 \cap B_2|}{|B_1 \cup B_2|}
$$

IoU 既用于训练匹配（正负样本分配），也用于 NMS。

### NMS（非极大值抑制）的“合同”

给定候选框集合与 score，NMS 输出集合倾向满足：

> 在阈值 $\tau$ 下，输出集合中任意两框的 IoU 都不超过 $\tau$，并倾向于保留更高分的框。

### 候选规模符号（后面会反复用到）

- `N = Σ_l (H_l×W_l×A)`：one-stage 候选规模（多尺度求和）
- `P`：two-stage 第一阶段保留的 proposals 数（常见 300~2000）
- `M`：最终输出框数上限（常见 50~100）

一个非常粗但很好用的复杂度直觉：

$$
\text{one-stage: } O(N) \;\text{candidates} \;\to\; O(N^2) \text{ worst-case NMS}
$$

$$
\text{two-stage: } O(N) \;\text{RPN} \;\to\; O(P) \text{ RoI head} \;\to\; O(P^2) \text{ NMS}
$$

---

## 可行性与下界直觉：为什么“候选数太大”一定会反噬

对密集预测来说，候选数 `N` 近似跟面积成正比：分辨率翻倍，`N` 近似翻四倍。
这带来两个不可避免的后果：

1) 后处理会变成瓶颈：NMS 的最坏复杂度随 `N^2` 增长
2) 训练更难：正负比会恶化（大量 easy negatives），梯度会被“无聊样本”淹没

---

## 基线与瓶颈（从滑窗到两条主路线）

历史上的滑动窗口可以看作 one-stage 的祖先：在每个位置、每个尺度都做分类回归。
它的问题很朴素：候选太多、正负极不平衡、后处理开销大。

现代 one-stage 的关键改进是：用 CNN/FPN 共享特征、用更好的损失/匹配策略稳定训练；
two-stage 的关键改进是：显式把“候选缩减”做成一个模块（RPN），让后续 head 只处理少量 RoI。

---

## 关键观察：速度/精度差异大多来自两件事

1) 候选集合何时变小（影响延迟与 NMS）
2) 训练时如何对抗不平衡与难例（影响误检/漏检）

下面进入两条主线的深挖。

---

## 深挖 1：候选集合规模 → top-k → NMS（PDKH）

### P：把问题重述成“候选集合的压缩”

工程上你真正关心的是：把 `N` 个候选压缩成 `M` 个输出（`M` 往往几十）。
不管单/双阶段，你最终都会做类似：`score_threshold → top-k → NMS`。

### D：最小可算的数字例子（anchors 数量）

以 `640×640`、FPN `8/16/32`、每点 3 anchors：

$$
N \approx 80^2\cdot 3 + 40^2\cdot 3 + 20^2\cdot 3 = 25200
$$

而很多 two-stage 配置会在 RPN 后取 `P=1000` proposals。
候选规模差了 25 倍，后处理最坏比较次数（$\sim N^2/2$）差了约 625 倍，这是速度直觉的来源。

### K：NMS 的不变式/合同（输出集合“互斥”）

$$
\forall a,b \in K, a\neq b \Rightarrow IoU(a,b) \le \tau
$$

### H：工程上 top-k 为什么几乎不可省

在 one-stage 里，如果你不先做 top-k，直接对 `25200` 个框做 NMS，CPU 侧很容易成为瓶颈。
因此实践中几乎都会把进入 NMS 的规模压到 `~1e3`：

- `score_threshold` 先砍低分噪声（例如 0.25 起步）
- `global_topk` 控总量（例如 300~2000）
- `max_det` 控最终输出规模（例如 50~100）

### 工程细节：per-class top-k 可能比你想的更“贵”

COCO `C=80`，如果你每类都保留 `k=1000`，总候选上限是：

$$
N_{in} \le C\cdot k = 80000
$$

即使做 per-class NMS（每类单独做），计算量也接近：

$$
\sum_{c=1}^{C} k^2 \approx C\cdot k^2 = 80\times 10^6
$$

更稳的工程做法通常是：先用阈值 + 全局 top-k 控住总量，再按类拆分进入 NMS。

### 阈值与规模：分辨率翻倍，候选近四倍（NMS 最坏 16 倍）

在固定 stride 集合与 anchors-per-location 的前提下：

$$
N(img) \approx \sum_{l} \left(\frac{img}{stride_l}\right)^2 \cdot A
$$

因此当 `img: 640 → 1280`（长宽都翻倍）时：候选数 `N` 近似 `×4`，NMS 最坏比较次数近似 `×16`。
这也是为什么很多线上系统把 top-k 当作“延迟预算阀门”。

### 再算一笔：分类分支的输出规模（N×C）也会放大

上面我们一直在算 NMS，但 one-stage 的“密集”不仅体现在框数量上，也体现在**分类分支的输出张量规模**上。

以 COCO 为例（类别数 `C=80`），如果你有 `N≈25200` 个候选，那么仅分类 logits 的元素数量就是：

$$
N\\cdot C \\approx 25200\\times 80 \\approx 2.0\\text{ million}
$$

这意味着两件工程事实：

1) **推理时你必须做大量 score 处理**：阈值、top-k、（可能还有 per-class 策略），这些操作虽然看起来是“小算子”，但在低 batch/CPU 后处理路径上会很显著。  
2) **训练时不平衡会被进一步放大**：在绝大多数位置与类别上，标签都是 “background / negative”，这也是 focal loss 等方法能“救回训练”的原因。

因此当你在 one-stage 上做性能优化时，一个常见的优先级是：

- 先把 `score_threshold` 提上来一点点（砍掉大量低分框）  
- 再把 `global_topk` 压到延迟预算能承受的范围  
- 最后才去讨论更重的结构改动（更小 backbone、蒸馏、量化）

### 失败模式（反例）：密集目标 + 朴素 NMS

在人群、停车场、密集小物体等场景：多个真目标之间 IoU 可能也很高。
朴素 NMS 可能会把相邻目标直接抑制掉（漏检）。
常见对策包括：调整 `tau`/per-class 策略、使用 Soft-NMS/DIoU-NMS。

---

## 深挖 2：正负样本不平衡（focal loss vs sampling）（PDKH）

### P：把问题重述成“梯度预算分配”

one-stage 的密集候选意味着大量负样本。
如果你让所有样本在损失里“票数相同”，训练会被 easy negatives 主导，模型学不到关键难例。

### D：最小数量级例子（1:1000 不是夸张）

还是 `N≈25200` 这个规模。
一张图里如果只有 10~30 个目标，那么正样本（按 IoU 匹配后的 positives）可能只有几十级别。
于是正负比很容易到：

$$
\text{pos:neg} \approx 1:1000
$$

### H：focal loss 的形式化（one-stage 常用）

RetinaNet 提出的 focal loss（对二分类）常写作：

$$
FL(p_t) = -\alpha (1-p_t)^\gamma \log(p_t)
$$

其中：`p_t` 是“预测对的概率”，`(1-p_t)^\gamma` 会让容易样本的梯度被压小，让难例占更多梯度预算。

#### 具体数字：focal loss 到底压掉了多少 easy negatives？

取 `γ=2`：

| $p_t$ | 难度直觉 | $(1-p_t)^2$ |
|---:|---|---:|
| 0.99 | 极容易 | $10^{-4}$ |
| 0.50 | 中等 | $0.25$ |

权重比值是：

$$
\frac{0.25}{10^{-4}} = 2500
$$

也就是说，一个中等难度样本的权重相当于 2500 个极易样本的总和。
这就是 focal loss 能在 one-stage 的“海量背景”设定下仍然学得动的核心原因之一。

### K：two-stage 常用“采样 + 两次过滤”来解决

two-stage 的做法更“结构性”：

1) RPN 先过滤掉大量明显背景（候选集合先变小）
2) RoI head 训练时对 positives/negatives 做采样（例如 1:3），让 batch 内梯度更均衡

带数字锚点：很多实现会固定每张图采样 `R=256` 个 RoI，正样本比例 `r_pos=0.25`：

- positives：`64`
- negatives：`192`

这相当于把训练梯度预算硬性钉在可控范围内——无论 RPN 原始产出了多少 proposals。

#### 一个很实用的上界：最终 recall ≤ proposals recall

two-stage 常见的错觉是“第二阶段很强所以一定更准”，但它有一个非常硬的上界：

> 如果一个真目标在 proposals 阶段就没被召回，后面的 RoI head 再强也无能为力。

因此你可以把最终 recall 近似看成：

$$
\\text{final recall} \\le \\text{proposals recall}
$$

工程上排查 two-stage 漏检时，经常第一步不是看 RoI head，而是看：

- RPN 的正负样本分配是否合理（IoU 阈值/采样）  
- RPN NMS/top-k 是否把真目标“挤掉了”（尤其密集场景）  
- proposals 数量 `P` 是否过小（比如为了提速把 `P=1000` 压到 `P=100`，很容易直接掉 recall）  

### 失败模式：focal loss 不是银弹

如果 `γ` 太大、或者分类 head 校准差，focal loss 可能带来两个典型问题：

1) **训练不稳定**：梯度过度集中在极少数样本上，batch 间波动变大  
2) **recall 下降**：模型变得过于谨慎，低分真目标更难被推上来（尤其小目标/遮挡）  

工程上判断是否“focal 过头”有个很实用的信号：你把 `γ` 从 2 降到 1（甚至 0）时，如果 recall 明显回升但 precision 下降，说明你在“难例强调”与“整体召回”之间需要重新平衡。

#### 补充：除了 focal loss，还有哪些不平衡处理手段？

不平衡本质是“训练预算分配”问题，focal loss 只是其中一种。
常见的工程替代/补充方案包括：

- **Hard Negative Mining / OHEM**：从海量负样本里挑一小部分“最难的”来训练。  
  例如你有 `N≈25200` 个候选，但每张图只取 top-1024 个 hardest negatives 参与分类损失，其余负样本不回传梯度。  
  这和 focal loss 的目标一致（减少 easy negatives 的影响），区别是 focal 是“连续加权”，OHEM 更像“离散筛选”。
- **采样策略（采样比/采样上限）**：two-stage 的 1:3 采样就是最典型的结构性手段；one-stage 也可以在 loss 计算上做采样。
- **更合理的正负分配**：通过 IoU 阈值、中心采样、或更强的匹配策略减少“含糊样本”，让训练信号更干净。

这些方法没有谁“绝对更好”。最稳的做法是：先用一个最简单的 baseline（例如 `γ=2` 的 focal 或固定采样比）跑通；然后用你业务的误检/漏检代价与线上延迟预算，决定把优化精力投入到哪里。

---

## 算法步骤（Practice Guide：把 one-stage/two-stage 写成可执行 checklist）

### One-stage（YOLO/SSD/RetinaNet）推理 checklist

1) Backbone + FPN：得到多尺度特征图
2) Dense head：对每个位置输出 `class logits + bbox`
3) Decode：把 head 输出还原为候选框坐标
4) 过滤：score threshold + top-k（避免 NMS 输入太大）
5) NMS：按 IoU 阈值去重，得到最终输出

### Two-stage（Faster R-CNN/Mask R-CNN）推理 checklist

1) Backbone + FPN
2) RPN：对密集 anchors 预测 objectness + bbox，生成 proposals
3) RPN NMS + top-k：把 proposals 压到 `P≈300~2000`
4) RoIAlign：对每个 proposal 抽取固定尺寸特征
5) RoI head：分类 + bbox 精修
6) 输出 NMS：得到最终结果

---

## Decision Criteria（选型指南：给出可直接用的阈值与问题）

- 延迟硬指标是否 < 30ms（单路 30FPS）？是 → 优先 one-stage，并严格控制 top-k/NMS
- 误检代价是否极高（医疗/工业）？是 → 优先 two-stage 或 one-stage + second-stage re-score
- 小目标占比是否很高（远距离、密集场景）？是 → 倾向 two-stage 或更高分辨率/更细 FPN

一个可作为起步的“保守配置”（先对齐延迟与效果，再细调）：

- `score_threshold`: 0.25
- `global_topk`: 1000（CPU 吃紧优先降到 300~500）
- `max_det`: 100（很多业务只需要 20~50）

如果你必须做 per-class NMS，建议先用全局 top-k 控住总量，再按类拆分进入 NMS。

一个很实用的“线上化”建议：把 `N / P / global_topk / max_det / NMS τ` 这些关键旋钮写进监控与配置变更记录里。

当线上出现“延迟尖刺”或“误检激增/漏检变多”时，你才有可能在 5 分钟内回答：是数据漂移、阈值变了、还是候选规模偷偷变大了。

最简单的做法是：把这些值随模型版本一起打到日志里，并在离线评估中固定它们做可比对。
当这些旋钮可控且可复现时，很多“模型问题”会立刻变成可验证的工程问题。
---

## 工程场景（Engineering Scenarios）

### 场景 1：边缘端实时视频（延迟/功耗优先）

- 目标：单路 30FPS（约 33ms/帧）甚至更高帧率
- 倾向：one-stage（YOLO/轻量 SSD），并把 `global_topk` 与 `max_det` 作为一等公民参数管理
- 常见优化顺序：先控后处理（top-k/NMS）→ 再换更小 backbone → 最后考虑蒸馏/量化

### 场景 2：工业/医疗（误检代价高，复核链路强）

- 目标：误检少、定位更稳（尤其高 IoU 质量更重要）
- 倾向：two-stage（Faster/Mask/Cascade 等）或“one-stage + second-stage re-score”的混合方案
- 关键检查：RPN/proposals 的 recall 是否足够（因为它会成为最终 recall 的上界）

### 场景 3：密集小目标（人群/车流/遥感）

- 风险：朴素 NMS 容易把相邻目标抑制掉（漏检）
- 倾向：先从“候选召回与后处理策略”入手（更高分辨率、更细 FPN、调整 NMS/阈值、必要时换 Soft-NMS/DIoU-NMS）

---

## Worked Example（Trace：用“候选数 + top-k + NMS”把差异跑一遍）

我们用一个可复制的 toy trace 来模拟“候选压缩”：

1) 假设 one-stage 输出 `N=2000` 个候选（真实可到 2.5 万，这里缩小以便演示）
2) 先取 `topk=300`
3) 再做 NMS（`tau=0.5`）

你会看到：top-k 是把 NMS 从 $N^2$ 拉回现实的关键。

### 最小手算 trace：3 个框跑一次 NMS（带 IoU 数字）

假设有 3 个候选框（`xyxy`）与分数：

- `b0=[10,10,50,50]`, score=0.90
- `b1=[12,12,48,48]`, score=0.80
- `b2=[60,60,90,90]`, score=0.70

阈值 `τ=0.5`，按分数排序后先选 `b0`。
计算 `b0` 与 `b1` 的 IoU：

- `b0` 面积：`40×40 = 1600`
- `b1` 面积：`36×36 = 1296`
- 交集：`[12,12,48,48]`，面积 `36×36 = 1296`

因此：

$$
IoU(b0,b1) = \frac{1296}{1600} = 0.81 > 0.5
$$

所以 `b1` 会被抑制；`b2` 与 `b0` 不相交，IoU=0，会被保留。
最终结果集就是 `{b0, b2}`。

---

## Correctness（Proof Sketch）：NMS 为什么保证“互斥”但不保证全局最优

NMS 是一个贪心算法：

1) 按 `score` 从高到低排序
2) 每次取当前最高分框加入结果集
3) 删除所有与它的 IoU 大于阈值 `τ` 的框
4) 重复直到候选耗尽

它能保证“互斥合同”，因为任何与已选框重叠超过阈值的候选都会在第 3 步被删除。

但它不保证全局最优：你可以构造反例让“先选一个高分框”删掉两个本应保留的中分框。
工程上我们仍大量使用 NMS，因为它足够简单、稳定、可加速。

---

## Runnable Implementation（纯 NumPy：候选规模算账 + NMS）

下面代码不依赖 PyTorch/torchvision，复制即可运行：

```python
import time
from typing import Optional, Tuple

import numpy as np


def iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area_a + area_b - inter, 1e-12)
    return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5, topk: Optional[int] = None) -> np.ndarray:
    order = np.argsort(-scores)
    if topk is not None:
        order = order[:topk]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_xyxy(boxes[i], boxes[rest])
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def fpn_anchor_count(img: int = 640, strides: Tuple[int, ...] = (8, 16, 32), anchors_per_loc: int = 3) -> int:
    total = 0
    for s in strides:
        h = img // s
        w = img // s
        total += h * w * anchors_per_loc
    return total


def random_boxes(rng: np.random.Generator, n: int, img: int = 640) -> np.ndarray:
    xy = (rng.random((n, 4)) * img).astype(np.float32)
    x1 = np.minimum(xy[:, 0], xy[:, 2])
    y1 = np.minimum(xy[:, 1], xy[:, 3])
    x2 = np.maximum(xy[:, 0], xy[:, 2])
    y2 = np.maximum(xy[:, 1], xy[:, 3])
    x2 = np.maximum(x2, x1 + 1.0)
    y2 = np.maximum(y2, y1 + 1.0)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


if __name__ == "__main__":
    n_anchors_640 = fpn_anchor_count(img=640, strides=(8, 16, 32), anchors_per_loc=3)
    n_anchors_1280 = fpn_anchor_count(img=1280, strides=(8, 16, 32), anchors_per_loc=3)
    print("anchors@640 :", n_anchors_640)   # ~25200
    print("anchors@1280:", n_anchors_1280)  # ~100800 (≈4x)

    rng = np.random.default_rng(0)
    boxes = random_boxes(rng, n=2000, img=640)
    scores = rng.random(2000).astype(np.float32)

    t0 = time.time()
    keep_topk300 = nms_xyxy(boxes, scores, iou_threshold=0.5, topk=300)
    t1 = time.time()

    t2 = time.time()
    keep_topk1000 = nms_xyxy(boxes, scores, iou_threshold=0.5, topk=1000)
    t3 = time.time()

    print("kept(topk=300):", keep_topk300.shape[0], "time(ms):", round((t1 - t0) * 1000, 2))
    print("kept(topk=1000):", keep_topk1000.shape[0], "time(ms):", round((t3 - t2) * 1000, 2))
```

---

## 复杂度与常数项（把“快/慢”落到可算的量）

### One-stage：密集 head + 强后处理

- 候选规模：`N = Σ_l (H_l×W_l×A)`
- head 计算：近似 `O(N)`
- 后处理：`topk + NMS`（有效输入规模取决于 topk）

### Two-stage：先压候选，再做重 head

- RPN：`O(N)` 生成 proposals
- proposals 规模：`P`（常见 300~2000）
- RoI head：`O(P)`（但常数更大：RoIAlign + FC/conv）
- 后处理：对 `P` 做 NMS（相对可控）

---

## 常数项与工程现实：为什么同样“检测”延迟差一大截

上面用大 O 写出来的复杂度非常有用，但它会隐藏很多工程上真正决定延迟的常数项。
下面用“能算得出来的量”把两个体系的常数项拆开（你可以直接拿去对照 profiler）。

### One-stage：常见瓶颈不是算子多，而是“密集 + 后处理”

1) **分类分支输出是 `N×C` 量级**  
   以 COCO 为例 `C=80`，`N≈25200` 时：

   $$
   N\\cdot C \\approx 2.0\\text{ million logits}
   $$

   这意味着：即使 backbone 很快，你仍然要做大量 score 的阈值/排序/top-k（并且可能要做 per-class 逻辑）。

2) **decode + NMS 经常走 CPU 路径**  
   在 batch 很小（线上常见 batch=1）或 CPU 较弱的环境里，“看起来很小的后处理”反而可能成为 P99 延迟主因。
   这也是为什么 one-stage 的工程优化常常从 `score_threshold / global_topk / max_det` 三个参数入手。

### Two-stage：常见瓶颈是 RoI 的“二阶段算得更贵”

two-stage 的直觉是：候选先变少，所以后面可以做更贵的分类/回归。
这里的“更贵”不是抽象词，是可以量化的。

一个常见设置是 RoIAlign 输出 `R×R`（例如 `R=7`），特征通道数 `C_feat=256`（FPN 常见）。
那么二阶段需要处理的 RoI 特征元素数大致是：

$$
P\\cdot C_{feat}\\cdot R^2
$$

代入 `P=1000, C_feat=256, R=7`：

$$
1000\\times 256\\times 49 \\approx 12.5\\text{ million}
$$

这还没算后面 FC/conv head 的计算。
因此 two-stage 的“调参旋钮”非常明确：

- `P` 越大：召回更好但延迟更高  
- `P` 越小：延迟更低但更容易直接掉 recall（因为 proposals recall 是最终 recall 的上界）  

### 用 profiler 定位的三步法（建议照着做）

1) **先量化规模**：记录 `N`（候选数）、`P`（proposals 数）、`M`（最终输出数上限）  
2) **看拆分时间**：backbone / head / post-processing（NMS、decode）分别占多少  
3) **只动一个旋钮做 A/B**：  
   - one-stage：先改 `global_topk`（1000→300），观察延迟是否线性下降  
   - two-stage：先改 `P`（1000→300），观察延迟下降幅度与 recall 掉点  

你会发现：大多数“为什么这套模型线上慢”的问题，不需要先换结构，先把规模与后处理路径跑通就能解决一半。

---

## Alternatives and Tradeoffs（替代路线与取舍）

- Anchor-free one-stage（FCOS/CenterNet）：减少 anchor 设计负担，但仍有密集候选与后处理成本
- Transformer 检测（DETR 系列）：用集合预测减少手工 NMS，但训练/部署形态不同

---

## 常见坑与边界条件（Pitfalls）

1) 把 NMS 当成“调参细节”：阈值/实现/是否 per-class 会直接改变 recall 与延迟
2) 密集场景盲目增大 top-k：后处理更慢且 NMS 漏检更严重
3) 把 two-stage 当成必然更准：如果 RPN recall 不够，后面 head 再强也救不回来

---

## 最佳实践（Best Practices）

- 从业务指标反推：延迟预算（ms）、最大输出框数（M）、可容忍误检/漏检类型
- 固定一个可复制的算账模板：`N（候选数）/topk/NMS 阈值/CPU-GPU 后处理路径`
- 先保证召回（recall），再用二阶段/校准/更强 head 降误检

---

## 总结 / Takeaways

1) one-stage/two-stage 都是“候选→打分→去重”；差别是候选集合何时变小、训练如何处理不平衡
2) 候选规模是第一性变量：`N≈25200` vs `P≈1000` 的数量级差会反映到延迟与 NMS 成本上
3) top-k 是工程必需品：它把 NMS 从最坏 $N^2$ 拉回可控预算
4) one-stage 常靠 focal loss 把梯度预算从 easy negatives 转向难例；two-stage 常靠“RPN 过滤 + 采样”实现结构性平衡
5) 密集目标要警惕 NMS 误伤真阳性：漏检可能来自后处理而非 backbone

---

## 参考与延伸阅读

- Faster R-CNN: https://arxiv.org/abs/1506.01497
- YOLOv1: https://arxiv.org/abs/1506.02640
- SSD: https://arxiv.org/abs/1512.02325
- Focal Loss / RetinaNet: https://arxiv.org/abs/1708.02002
- DETR: https://arxiv.org/abs/2005.12872

---

## 元信息

- **阅读时长**：约 15 分钟
- **标签**：object-detection、one-stage、two-stage、nms
- **SEO 关键词**：目标检测, 单阶段, 双阶段, YOLO, Faster R-CNN
- **元描述**：用候选集合规模与训练不平衡两条主线，对比单/双阶段检测并给出可运行算账代码。

---

## 行动号召（CTA）

把代码跑起来，然后做两件事：

1) 把 `topk=300/1000/2000` 各跑一次，观察 NMS 时间变化，画出你的“预算曲线”
2) 把输入分辨率从 `640 → 1280`，直观看到候选数近四倍与后处理压力上升
