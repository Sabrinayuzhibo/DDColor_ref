# DDColor-CondB 模型构建说明（当前代码版）

本文档说明当前工程里，模型是如何被构建、条件分支如何接入，以及训练/推理时各模块的实际数据流。

---

## 1. 总体结构

当前训练模型由三部分组成：

- `net_g`：主生成器（`DDColor`）
- `net_d`：判别器（`DynamicUNetDiscriminator`）
- `net_c`：条件分支（`MultiScaleDenseTokenConditioner` 或 `MultiScaleRegionTokenConditioner`）

其中 `cond-B` 的核心是：

1. 参考图先走 `net_g` 的特征提取路径；
2. 提取三尺度特征送入 `net_c` 生成条件 token；
3. 条件 token 注入到 `net_g.decoder.color_decoder` 的 cond cross-attn；
4. 生成目标图 AB 通道并与输入 L 通道融合得到 RGB 输出。

---

## 2. `net_g`（DDColor）如何构建

`net_g` 在 `ColorModel.__init__` 中通过 `build_network(opt['network_g'])` 构建。

`DDColor` 内部主要包含：

- `encoder`：ConvNeXt（`convnext-l/s/b/t`）
- `decoder`：UNet 解码 + `MultiScaleColorDecoder`
- `refine_net`：最终 1x1 卷积细化输出

关键点：

- 输入是灰度扩展成 3 通道的 RGB（或标准 RGB）；
- 输出是 AB（2 通道），再与 L 拼回 Lab 后转 RGB。

---

## 3. `net_c`（conditioner）如何构建

当 `train.cond_opt.enable=true` 时，会创建 `net_c`：

- `token_mode in ('dense','grid')` -> `MultiScaleDenseTokenConditioner`
- `token_mode in ('region','mask')` -> `MultiScaleRegionTokenConditioner`

常用配置：

- `num_scales=3`
- `hidden_dim=256`
- `grid_size=16`

---

## 4. 当前版本的条件特征来源（重点）

当前代码已经改成：**conditioner 使用 pixel decoder 三尺度特征**。

调用路径：

1. `net_g.extract_condition_features(ref_rgb, use_pixel_decoder=True)`
2. 内部通过 `decoder.get_condition_features()` 返回 `[out0, out1, out2]`
3. `net_c(ref_feats)` 产出 `cond_tokens_per_scale, cond_pos_per_scale`

这意味着现在 cond token 的来源不是 backbone hook，而是 pixel decoder 三尺度输出。

---

## 5. 训练时参数冻结策略

由以下开关控制（三者互斥）：

- `train_only_cond`
- `train_decoder_cond`
- `train_color_decoder_cond`

常用的 `train_color_decoder_cond=true` 逻辑：

1. 先冻结 `net_g` 全部参数；
2. 仅解冻 `net_g.decoder.color_decoder`；
3. `net_c` 按 `requires_grad=True` 正常训练；
4. `net_d` 独立优化。

---

## 6. `freeze_ref_encoder` 到底冻结了什么

`freeze_ref_encoder` 只影响“参考图特征提取时是否构图”：

- `true`：参考特征提取放在 `torch.no_grad()` 内，省显存/省算力；
- `false`：参考特征提取参与梯度图。

注意：

- 它不直接决定 `net_c` 是否训练；
- `net_c` 是否更新取决于：是否进了优化器 + 是否有有效反传路径。

---

## 7. 一次训练迭代的数据流

每个 iteration（`optimize_parameters`）核心步骤：

1. `feed_data` 读取 `lq/gt/ref_rgb`；
2. 从参考图提取三尺度特征（当前为 pixel decoder 三尺度）；
3. `net_c` 生成 cond tokens，并乘 `cond_gain`；
4. `net_g(self.lq_rgb, cond_tokens=..., cond_pos=...)` 得到 `output_ab`；
5. 计算损失：`pixel/perceptual/ref_style/ref_moment/ref_cov/ref_contrast/gan/colorfulness/cond_gate_push/...`；
6. 更新 `optimizer_g`；
7. 再更新 `optimizer_d`。

---

## 8. 推理时的数据流

`scripts/infer_condB_pair_compare.py` 当前也与训练保持一致：

- 优先调用 `model.extract_condition_features(..., use_pixel_decoder=True)`；
- 再走 `conditioner(ref_feats)` 构建 cond token；
- 将 cond token 注入生成器进行上色；
- 支持 neutral/aggressive 两类后处理参数。

---

## 9. 你最关心的结论

- 当前版本已经是：**参考图 -> pixel decoder 三尺度特征 -> net_c grid token -> color decoder cond cross-attn**。
- 训练/推理两边的 cond 特征来源已统一。
- 若想进一步增强“参考感”，优先调：`cond_gain`、`cond_gate_push_opt`、`ref_style/ref_moment/ref_cov/ref_contrast`。

