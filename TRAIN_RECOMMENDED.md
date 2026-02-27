# DDColor Cond（参考图条件）训练：推荐做法

这份文档把「结合 README 里原模型用法」的一条**推荐训练路线**写成可直接照做的步骤，并给出一些**训练策略**与**后续迭代方向**。

---

## 目标（你现在这条线在做什么）

在 `condB` 这类配置下，常见做法是：

- **固定（freeze）DDColor 主干 `net_g`**，让它保持原本的自动上色能力与稳定性
- **只训练条件分支（例如 `net_c` / cond tokens）**，学习“如何利用参考图（ref）的特征，生成合适的多尺度 tokens”
- 各种 loss（pixel / perceptual / ref-style / GAN / colorfulness / gate push 等）的梯度会**穿过 frozen 的 `net_g`**回传到 cond tokens，再回到 `net_c`

等价理解：你在训练一个 **style adaptor**（参考图 -> 条件 tokens），让一个固定的 DDColor 主干去完成“听参考图的风格/配色”的上色。

---

## 推荐训练流程（按一条最稳的路径走）

### 1) 准备原始 DDColor 的预训练权重（非常关键）

按 `README.md` 的 Train 部分准备依赖权重：

- 下载 **ConvNeXt large** 的预训练：`pretrain/convnext_large_22k_224.pth`
- 从 `MODEL_ZOO.md` 里选一个你要对齐的 **DDColor 主干模型**（例如 `ddcolor_paper` 或 `ddcolor_modelscope`），把权重放到 `pretrain/`

> 注：有些模型来自 HuggingFace，会是 `pytorch_model.bin`。一般来说 `.bin` / `.pth` 只是文件扩展名不同，本质都是 PyTorch 权重文件；你可以直接在配置里指向它，或把它重命名为 `.pth` 以统一命名习惯。

### 2) 配置 `condB` 的训练（主干对齐 + 条件学习）

以你现在的 `condB` 配置（如 `options/train/train_ddcolor_condB_p1_grid_posenc_2k.yml`）为例，推荐这样接：

- **`path.pretrain_network_g`**：指向你选的 DDColor 主干权重  
  例如：`pretrain/ddcolor_paper.pth`
- **`path.pretrain_network_d`**：如果你有对应的 D 权重就填；没有也可以先留空，让 D 从头训练（重训）

### 3) Loss 与“只训 cond”的核心点

保持你现有 loss 配置不动，让它们都对 cond 生效（你当前已经写了这些模块）：

- `pixel_opt`
- `perceptual_opt`
- `ref_style_opt`
- `gan_opt`
- `colorfulness_opt`
- `cond_gate_push_opt`

关键点是：即使 `net_g` 冻结，它仍然是可微的计算图，loss 梯度会回传到 cond 分支，从而实现“只训 cond”的有效学习。

---

## 训练策略建议（实战向）

### 学习率（LR）

你现在 `optim_g.lr = 1.2e-5`。如果你**实际只训练 `net_c`**，一般可以略微加一点：

- 建议区间：$2 \sim 5 \times 10^{-5}$
- 推荐做法：先保持偏保守（如 $2\times10^{-5}$），跑一个短的 smoke test 看趋势，再决定是否加大

### 迭代数（Iter）

你这份 `2k` 作为 **smoke test** 很合理，用来验证：

- grid + pos enc + cond-only 是否稳定
- loss 曲线是否正常
- 可视化结果是否朝预期收敛

如果效果 OK，再开长跑实验（例如 `20k` 或 `50k`）：

- 按比例放大 `total_iter`
- 相应按比例放大 `scheduler.milestones`

### ref encoder 要不要冻（`freeze_ref_encoder`）

你目前设置为 `false`，建议先保持：

- 在“只训 cond”的设定下，ref encoder 往往是 `net_g` encoder 的一部分，本身来自大模型预训练
- 如果再把 ref encoder 冻得很死，cond 只能适配一个完全固定的 ref 特征分布，可能更僵硬、不够“听话”

### GAN loss 稳定性（是否关 / 降权重）

如果训练初期你发现 GAN 不稳定，可以先把权重降一点，让主要信号来自 perceptual / ref-style：

- 例如把 `gan_opt.loss_weight` 从 `0.24` 降到 `0.1`（甚至更低）
- 等模型稳定后，再逐步调回更强的 GAN 权重

---

## 后续两个方向（你跑通后怎么继续）

### 方向 A：保持“只训 cond + 固定 DDColor 主干”

- **好处**：不容易破坏原模型在普通自动上色上的能力；你只是新增“参考图 -> 色彩/风格”的映射
- **适用**：你这种手动准备的 `ref/photo pair`（如 `assets/test_images_condB` + `assets/test_ref_images`）场景，整体更像一个 **style adaptor**

### 方向 B：如果 cond 不够强，再微调整个 decoder / gate

可以再开一个实验：

- 仍然冻结 ConvNeXt encoder
- 只放开 decoder + cond gate + `net_c`
- 让模型对参考图更“听话”

这通常需要在 `ColorModel` 里更细粒度地区分哪些参数可训练；建议你先把 pure cond-only 跑通并稳定复现，再进入这个方向。

---

## 建议的落地方式（最省时间）

- 先跑 `2k` smoke test（你现在这份就是），确认训练与可视化链路完全 OK
- 再复制一份配置改成 `20k/50k` 长跑，并把 milestones 同步按比例放大

