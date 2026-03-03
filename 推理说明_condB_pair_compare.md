# condB 双参考推理使用说明

本说明对应脚本：`scripts/infer_condB_pair_compare.py`

适用场景：
- 1 张灰度内容图（或内容图文件夹）
- 2 张参考图
- 输出两路上色结果 + 差异可视化 + 条件对比报告

---

## 1. 环境准备

```bash
cd /root/autodl-tmp/ddcolor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ddcolor
```

---

## 2. 快速开始（推荐模板）

```bash
OMP_NUM_THREADS=8 python scripts/infer_condB_pair_compare.py \
  --ckpt experiments/train_ddcolor_condB_p1_grid_posenc_20k_artist_color_decoder_cond/models/net_g_20000.pth \
  --cond_ckpt experiments/train_ddcolor_condB_p1_grid_posenc_20k_artist_color_decoder_cond/models/net_c_20000.pth \
  --content assets/test_images_condB \
  --ref1 assets/test_ref_images/23a244c3-d5ac-4fc2-b924-ba1b743108a5.jpg \
  --ref2 assets/test_ref_images/aff9b6d2-e168-4598-af54-d6b6c198be0f.jpg \
  --input_size 384 \
  --model_size large \
  --num_queries 100 \
  --num_scales 3 \
  --dec_layers 9 \
  --cond_gain1 2.2 \
  --cond_gain2 2.2 \
  --contrast_boost 0.9 \
  --center_cond_tokens \
  --target_cond_rms 0.20 \
  --ab_ref_delta_gain 2.8 \
  --ab_pair_contrast_boost 1.8 \
  --ab_chroma_gain 1.30 \
  --ab_clip 1.15 \
  --override_cond_gate_raw 2.5 \
  --output results_condB_pair_compare_aggressive
```

> 如果你想看“模型原始表现”，建议先把增强项关掉（见下文“中性参数”）。

---

## 3. 中性参数（排查是否模型本体偏灰）

```bash
OMP_NUM_THREADS=8 python scripts/infer_condB_pair_compare.py \
  --ckpt <net_g_xxx.pth> \
  --cond_ckpt <net_c_xxx.pth> \
  --content <content_file_or_dir> \
  --ref1 <ref1.jpg> \
  --ref2 <ref2.jpg> \
  --input_size 384 \
  --model_size large \
  --num_queries 100 \
  --num_scales 3 \
  --dec_layers 9 \
  --cond_gain1 1.2 \
  --cond_gain2 1.2 \
  --contrast_boost 0.0 \
  --ab_ref_delta_gain 1.0 \
  --ab_pair_contrast_boost 0.0 \
  --ab_chroma_gain 1.0 \
  --ab_clip 1.0 \
  --output results_condB_pair_compare_neutral
```

---

## 4. 主要参数说明

### 必选参数
- `--ckpt`：生成器权重（`net_g_*.pth`）
- `--content`：内容图路径（单图或文件夹）
- `--ref1` / `--ref2`：两张参考图

### conditioner 相关
- `--cond_ckpt`：conditioner 权重（`net_c_*.pth`）
  - 不传时会尝试从 `--ckpt` 自动推断同目录 `net_c_*`
  - 找不到时默认报错（避免误用随机 conditioner）
- `--allow_random_conditioner`：允许随机 conditioner（不推荐）

### 模型结构匹配参数
需要与训练时一致：
- `--model_size`（`tiny` / `large`）
- `--num_queries`
- `--num_scales`
- `--dec_layers`
- `--input_size`

### 条件强度与后处理
- `--cond_gain1`, `--cond_gain2`：两路参考 token 强度
- `--contrast_boost`：两路 token 差异拉开
- `--center_cond_tokens`：token 去均值
- `--target_cond_rms`：token RMS 归一目标
- `--override_cond_gate_raw`：强制 gate logit（例如 `2.5`）
- `--ab_ref_delta_gain`：相对无条件基线的 AB 增强
- `--ab_pair_contrast_boost`：两路 AB 拉开
- `--ab_chroma_gain`：全局色度放大
- `--ab_clip`：AB 裁剪范围

---

## 5. 输出目录结构

脚本会在 `--output` 下生成：

- `ref_lowres/`：参考图缩略图
- `<content_key>/content_original.png`：原内容图
- `<content_key>/dif.png`：两路输出差异热力图
- `<content_key>/outputs/<ref1_key>.png`：参考图 1 的上色结果
- `<content_key>/outputs/<ref2_key>.png`：参考图 2 的上色结果
- `<content_key>/condition_compare.txt`：
  - cond token 差异统计（raw / post）
  - ab 差异统计
  - 像素差异统计

---

## 6. 常见问题

### Q1：报 `Conditioner checkpoint not found`
- 检查 `--cond_ckpt` 是否存在
- 或确保 `--ckpt` 同目录有匹配的 `net_c_*.pth`

### Q2：报 shape mismatch / size mismatch
- 推理结构参数必须与训练时一致：
  - `model_size / num_queries / num_scales / dec_layers`

### Q3：输出几乎灰色
- 先用“中性参数”判断模型本体
- 再切“增强参数”看推理端是否可拉回
- 常见现象：gate sigmoid 很低时，参考影响会弱

### Q4：两张参考图结果几乎一样
- 提高 `cond_gain*`
- 打开 `--contrast_boost`
- 尝试 `--override_cond_gate_raw 2.5`

---

## 7. 实用建议

- 先跑 `neutral`，再跑 `aggressive`，做 A/B 对照。
- 若只看模型真实能力，不要启用 `override_cond_gate_raw` 和激进 AB 增强。
- 若目的是“更像参考图”，可逐步加大：
  1) `cond_gain` → 2) `contrast_boost` → 3) `override_cond_gate_raw` → 4) AB 后处理项。
