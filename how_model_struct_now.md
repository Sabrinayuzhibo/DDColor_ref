# 当前模型结构总览（DDColor Cond-B）

## 1. 目标与定位
- 任务：参考图引导的灰度图上色（style transfer colorization）。
- 目标：让头发/五官等语义区域更强地跟随参考图，同时控制渗色。

## 2. 核心组成

### 2.1 生成器（net_g）
- 主体为 DDColor（encoder + MultiScaleColorDecoder）。
- 输入：内容图灰度结构（由 L 通道构造灰度 RGB）。
- 输出：ab 色彩通道，最终与原始 L 拼接重建彩色图。

### 2.2 判别器（net_d）
- GAN 对抗训练分支。
- 训练时采用常规交替：更新 G 时冻结 D，更新 D 时解冻 D。

### 2.3 条件分支（cond_conditioner）
- 模块：
	- `dense` 模式：MultiScaleDenseTokenConditioner（当前 CondB 主路径）。
- 作用：把参考图多尺度特征压缩为条件 token（tokens）与位置编码（pos）。
- 参考特征来自 encoder hook（norm1 / norm2 / norm3）。
- 条件注入：解码阶段通过 cond_tokens + cond_pos 引导上色。
- 新增：多层门控注入（gated cond-attn）。
	- 每个 decoder layer 都会执行一次条件 cross-attn。
	- 每层有一个可学习 gate（sigmoid 后为 0~1），自动控制该层参考注入强度。
	- `cond_gate_init` 决定初始注入强度（当前默认约 0.18，对应 -1.5）。

## 3. 训练阶段数据流
1. 读取 lq / gt / ref_rgb。
2. 参考图经 encoder 得到多尺度特征。
3. cond_conditioner 生成条件 token。
4. net_g(lq_rgb, cond_tokens, cond_pos) 预测 ab。
5. 计算损失并反向：
	 - 像素损失（pixel）
	 - 感知/风格损失（perceptual/style）
	 - GAN 损失
	 - 可选：reference style loss（含区域版）
	 - 可选：dense warp loss
6. 更新优化器：
	 - optimizer_g 现在包含 net_g + cond_conditioner 参数
	 - optimizer_d 更新 net_d

### 3.1 模型里怎么用参考图（详细版）
1. 数据侧提供参考输入：
	- `ref_rgb` 是参考图 RGB（当前 CondB 仅依赖该输入）。
2. 参考图进入生成器编码器（与内容图共享编码器）：
	- 在 `color_model.py` 的 `_build_cond()` 里，先对 `ref_rgb` 做 normalize，再过 `net_g.encoder`。
	- 从 hook 中取 `norm1 / norm2 / norm3` 三个尺度特征。
3. `cond_conditioner` 生成条件序列（由 `train.cond_opt.token_mode` 控制）：
	- `dense`：不使用掩码，对多尺度特征做固定网格池化（`grid_size`）生成更密集 `cond_tokens`。
	- 同时生成 `cond_pos` 供解码器 cross-attn 使用。
4. 解码时注入条件：
	- 前向不是普通 `net_g(lq_rgb)`，而是 `net_g(lq_rgb, cond_tokens, cond_pos)`。
	- 这一步决定了“结果颜色是否跟参考图走”。
5. `freeze_ref_encoder` 的含义：
	- `false`：参考编码器在训练中可反传更新（CondB 迁移常用）。
	- `true`：参考编码器不更新（更稳，但风格迁移可塑性更弱，常用于 CondA 打底阶段）。

### 3.2 损失是怎么组成的（CondB 详细版）
> 在每个 iteration 中，先更新 G（冻结 D），再更新 D（解冻 D）。

#### G 侧总损失（`l_g_total`）
- `l_g_pix`（L1）：约束输出 `ab` 不要完全跑飞，保证基本可用性。
- `l_g_percep`（VGG 感知）：约束结果与目标图在语义纹理层面接近。
- `l_g_gan`（对抗）：推动结果分布更像真实彩色图。
- `l_g_color`（colorfulness）：提升颜色丰富度，减少过灰。
- `l_g_ref_style`（参考风格项，CondB核心）：
	- 由 `ref_style_opt` 提供，训练中权重较高（用于强化“向参考图靠拢”）。
	- 当前主线按整图风格统计约束，不依赖显式语义掩码。
	- 区域化机制至少包含：最小面积过滤、按区域平均/加权融合；可选边界门控、置信度门控、landmark 代理增强。
- （可选）`l_g_warp`：若开启 dense warp，则加上 correspondence 约束。

#### D 侧损失
- `l_d = l_d(real) + l_d(fake)`，标准 GAN 判别器更新。
- 记录 `real_score / fake_score` 观察判别器状态。

#### 关键训练特性（为什么 CondB 更像风格迁移）
- `cond_ref_mode=random`：内容图与参考图随机配对，逼模型学“跨图传递颜色风格”。
- `ref_style_opt.style_weight` 较高：明确加强参考图风格监督。
- `pixel` 权重较低：减少“死贴 GT 颜色”的惯性，为参考迁移留空间。

## 4. 关键修复（已生效）

### 4.1 conditioner 真正参与训练
- 已修复：optimizer_g 包含 cond_conditioner.parameters()。
- 避免“前向参与但参数不更新”的问题。

### 4.2 conditioner checkpoint 闭环
- 训练保存：新增 net_c_*.pth（conditioner 权重）。
- 训练加载：支持 path.pretrain_network_c。
- 推理加载：支持 --cond_ckpt；若不传，会从 --ckpt net_g_xxx.pth 自动匹配 net_c_xxx.pth。

### 4.3 新增 dense token（去掩码）路线
- 训练配置可设：`train.cond_opt.token_mode: dense`。
- 数据配置可设：`cond_mask_mode: none`，即不再构建语义掩码。
- 目的：减少硬掩码误差带来的渗色/错配，让参考信息以更细粒度 token 参与解码。
- 当前 CondB 训练配置已统一到该路线。

### 4.4 新增多层门控 cross-attn（第一步优化）
- 位置：`MultiScaleColorDecoder` 的条件注意力分支。
- 机制：`output <- output + gate_i * (cond_output - output)`。
- 价值：
	- 让模型在不同层自动决定参考注入力度；
	- 抑制“所有层同强度注入”导致的过染色/不稳定；
	- 强化头发/五官等局部语义跟随时的层级控制能力。

### 4.5 损失函数回退（当前生效）
- 已回退到原始 CondB 主损失栈：`pixel + perceptual + ref_style + gan + colorfulness`。

### 4.6 可选 gate 强化损失（仅在新实验配置启用）
- 位置：`ColorModel.optimize_parameters()`。
- 形式：对 decoder 的 `cond_layer_gates` 增加软约束，鼓励 `sigmoid(gate)` 不低于目标值（`target_sigmoid`）。
- 配置项：`train.cond_gate_push_opt`（`enable/loss_weight/target_sigmoid`）。
- 默认关闭，不影响历史配置；仅在 `train_ddcolor_condB_p1_dense_gated_refboost_20k.yml` 启用，用于增强参考注入并保持可回滚。

## 5. 推理阶段流程（scripts/infer_style_transfer.py）
1. 加载 net_g 与 cond_conditioner（优先加载训练得到的 net_c）。
2. 内容图转灰度输入，保留原始 L。
3. 直接由参考图多尺度特征生成 dense token，无需掩码。
4. conditioner 生成 token，引导生成 ab。
5. 重建输出图。
6. 当前策略：推理侧禁用所有掩码相关路径（region token、region_color_transfer、MODNet matte）。

### 5.1 全参考结构化输出（scripts/infer_allpairs_structured.py）
- 用途：每张内容图 × 所有参考图（all-pairs）批量推理。
- 输出组织：每个内容图一个子文件夹，内部包含：
	- content_original.png（原始内容图）
	- ref_lowres/（所有参考图的低分辨率版）
	- outputs/（该内容图对应每个参考图的推理结果）
- 	- diff_pairs/（同一内容图在不同参考图下的结果差异对比）
- 当前默认流程不生成语义掩码可视化文件。

## 6. MODNet 接入（当前策略）
- 代码中仍保留 MODNet 相关实现，但当前推理策略已禁用其入口。
- 原因：统一满足“推理侧不使用掩码”的约束，避免块状伪影与额外掩码依赖。

## 7. 文件命名与目录约定（当前）
- 模型：
	- net_g_xxx.pth：生成器
	- net_d_xxx.pth：判别器
	- net_c_xxx.pth：conditioner
- 结果目录示例：
	- results_condB_transfer12k_dense
	- results_condB_transfer12k_dense_modnet
	- results_condB_transfer2k_allpairs_structured_dense
- 推理图命名：{content}__ref_{reference}.jpg/png
	- 目的：同一内容在多参考下可直接对齐对比。

### 7.1 目录命名原因说明
- `results_condB_transfer2k_allpairs_structured`：
	- `condB`：条件参考上色路线。
	- `transfer2k`：使用 2k 训练节点权重。
	- `allpairs`：每个内容图对全部参考图。
	- `structured`：每个内容图单独子目录，内含原图/参考缩略图/推理输出。
- `results_condB_transfer2k_allpairs_structured_dense`：
	- 在上面基础上显式使用 dense 无掩码条件分支。

## 8. 当前已知风险与注意点
- 若推理时未加载 net_c，会退化为随机初始化 conditioner，参考驱动会变弱。
- MODNet 的导入路径使用动态 sys.path，静态分析可能提示告警，但运行可用。

## 9. 一句话总结
- 当前架构是“DDColor 主干 + CondB 默认 dense 无掩码 token 引导（推理侧严格禁用掩码分支）”，并且 conditioner 现已实现训练/保存/加载全链路闭环。
- 在此基础上，解码器已升级为“多层 gated 条件注意力注入”，用于提升参考语义迁移强度与稳定性。
