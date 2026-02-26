## 0）你现在已有的前提

你最终喂给 DDColor 的条件是：

- [cond_tokens](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html): `(S, B, 256)`
- [cond_pos](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html): `(S, B, 256)`
  其中 `S` 是条件序列长度。

在方法 2 里：

- 区域数 `R = 6`（skin/hair/lips/eyes/cloth/background）
- 尺度数 `S_scales = 3`（来自 `norm1/norm2/norm3`）
  所以最终序列长度：`S = R * S_scales = 18`

------

## 1）先拿到参考图的 3 个尺度特征

把参考图 `ref_rgb` 过 encoder（ConvNeXt），你会得到多尺度 feature maps，例如：

- `F1 = norm1`: `(B, C1, H1, W1)` （分辨率较高，细节多）
- `F2 = norm2`: `(B, C2, H2, W2)` （折中）
- `F3 = norm3`: `(B, C3, H3, W3)` （分辨率最低，语义强）

注意：`C1,C2,C3` 不是 256，通常是 backbone 的通道数（convnext-t/l 不同）。

------

## 2）把 masks 对齐到每个尺度（很关键）

你有 6 类 masks：`M`，通常在原图分辨率上：

- `M`: `(B, R, H, W)`

但 `F1/F2/F3` 的空间尺寸是 `(H1,W1)/(H2,W2)/(H3,W3)`，所以要 resize：

- `M1 = resize(M -> H1,W1)` 形状 `(B,R,H1,W1)`
- `M2 = resize(M -> H2,W2)`
- `M3 = resize(M -> H3,W3)`

我们代码里用 nearest 插值（保持 one-hot 语义）。

------

## 3）每个尺度分别做“区域池化”得到 6 个区域 token（每尺度 6 个）

对每个尺度 k（k=1..3），对每个区域 r（r=1..6）做 masked pooling：

### 3.1 masked mean / std（对每个通道做）

以某个尺度特征 `Fk (B,Ck,Hk,Wk)` 和对应 mask `Mk (B,R,Hk,Wk)` 为例：

- 对区域 r 的 **masked mean**（得到一个长度为 Ck 的向量）：
  [
  $$\mu_{k,r} \in \mathbb{R}^{C_k}$$
  ]
- 对区域 r 的 **masked std**：
  [
  \sigma_{k,r} \in \mathbb{R}^{C_k}
  ]

拼起来：

- `pooled_k`: `(B, R, 2*Ck)`
  （我们代码可选再拼一个 `area_frac`，变成 `(B,R,2*Ck+1)`，用来表示这个区域面积占比，帮助稳定训练）

直觉：

- 这一步相当于把“区域内所有像素的参考特征”压缩成一个“区域描述向量”（包含纹理/材质/光照等深层信息）。
- **不同尺度的 pooled 表达不同层次信息**：高分辨率更偏纹理，低分辨率更偏语义。

### 3.2 投影到 256 维

因为 Transformer decoder 的 hidden_dim 是 256，我们用一个投影层把不同通道数的 pooled 变成统一维度：

- `tok_k = Linear_k(pooled_k)` → `(B, R, 256)`
- 转成 attention 序列格式：
  - `tok_k.transpose(0,1)` → `(R, B, 256)`

所以你会得到：

- `cond_tokens_1`: `(R,B,256)`
- `cond_tokens_2`: `(R,B,256)`
- `cond_tokens_3`: `(R,B,256)`

------

## 4）不要合并三尺度 token —— 按尺度分别保留并独立使用（建议）

在实践与论文设计上，我们建议**不要**把三尺度 token 在序列维上简单拼接成一条长序列再一次性交给 decoder 的 cond cross-attn。相反，保留每个尺度的一组 token，并在 decoder 调用时按尺度独立使用：

- `cond_tokens_1`、`cond_pos_1` 对应 `norm1`（高分辨率）
- `cond_tokens_2`、`cond_pos_2` 对应 `norm2`（中分辨率）
- `cond_tokens_3`、`cond_pos_3` 对应 `norm3`（低分辨率）

理由与优点：
- 空间/语义对齐更明确：每次 image cross-attn 只对该尺度的图像 memory 生效，同时 cond cross-attn 也只看同尺度的 cond token，降低跨尺度干扰；
- 计算与内存可控：不必让每一层对超长序列（如 336 tokens）做 attention，改为每层只对与当前尺度对应的 token 做 attention（token 长度更小）；
- 可实现更细粒度的尺度自适应：可以为每个尺度单独设置 gate、gain 或 dropout，便于做 ablation 与论文对比。

建议 API/实现变更（文档级别说明）：
- `cond_conditioner(ref_feats, ...)` 返回 `cond_tokens_per_scale, cond_pos_per_scale`，其中每项为 list/tuple，长度等于 `num_scales`，每个元素形状为 `(Sg_i, B, C)` 和 `(Sg_i, B, C)`；或返回 dict `{scale_idx: (tokens,pos)}`；
- `MultiScaleColorDecoder.forward(..., cond_tokens_per_scale=None, cond_pos_per_scale=None)` 接受按尺度的条件记忆；在第 i 层，它从 `cond_tokens_per_scale[level_index]` 取出对应尺度的条件并执行 cond cross-attn；
- 或者保持现有接口但在 pre-processing 阶段给出 `cond_key_padding_mask` 和 `cond_pos` 的分段信息，使得 decoder 能在内部按段读取对应的子序列（实现上略微复杂但向后兼容）。

下面的层内顺序示例将按“按尺度独立 cond-attn”来说明 decoder 行为。

------

## 5）为什么一定要加 `scale_embed`（以及 [cond_pos](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 怎么构成）

仅有 `cond_tokens_all` 还不够好，因为 Transformer 在 cross-attn 里看到的是一串向量，它并不知道：

- 第 0 个和第 6 个 token 都是 “skin”，但来自不同尺度
- 第 2 个 token 是 “lips”，第 3 个 token 是 “eyes”
  这些“身份信息”必须显式编码进去，否则注意力要靠自己去猜，训练会更慢、也更不稳。

所以我们构造 [cond_pos](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（位置/身份编码）：

### 5.1 region_embed：告诉它“这是什么区域”

- `region_embed[r]`：第 r 个区域（skin/hair/...）的可学习向量，维度 256
- 得到 `region_pos`: `(R,B,256)`（对 batch repeat）

### 5.2 scale_embed：告诉它“来自哪个尺度”

- [scale_embed[k\]](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html)：第 k 个尺度（norm1/norm2/norm3）的可学习向量，维度 256
- 得到每个尺度的 `scale_pos_k`: `(R,B,256)`（对 region 和 batch repeat）

### 5.3 合成 cond_pos（每个 token 都有 “区域+尺度” 身份）

对每个尺度：

- `pos_k = region_pos + scale_pos_k` 形状 `(R,B,256)`

最后拼起来：

- `cond_pos_all = cat([pos_1, pos_2, pos_3], dim=0)` → `(18,B,256)`

在 cross-attn 里，key 会用 [memory + pos](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（你项目里的 [CrossAttentionLayer.with_pos_embed(memory, pos)](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html)），等价于：

- **同一个区域在不同尺度**会带上不同的偏移
- 同一个尺度不同区域也能区分开

这就非常像 DDColor 原本的 [level_embed](vscode-file://vscode-app/c:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/bdd88df003/resources/app/out/vs/code/electron-browser/workbench/workbench.html)：原模型给图像的每个尺度特征加一个 level embedding，告诉注意力“这段 memory 来自第几层特征”。我们这里是把同样思想用在“条件 tokens 的尺度区分”上。

------

## 6）这些 18 tokens 进入 decoder 后到底怎么用？

在每个 decoder layer 里（我加的部分）：


Then模型再去对图像特征做原来的 cross-attn，所以你得到的是：

- 先注入“多尺度参考区域信息”
- 再结合“输入图像内容特征”
- 输出更符合参考风格且不易串色的颜色预测

------

## 7）Pyramid Grid（多尺度不对称网格）——替代 face/region 掩码（已实现）

### 概要
为了在论文实验中既展示多尺度参考能力又避免掩码带来的硬边伪影，我们实现了 Pyramid Grid：对三尺度特征分别使用不对称网格（例如 norm1→16x16, norm2→8x8, norm3→4x4），每尺度独立池化并投影为 token，最后在序列维拼接为 `cond_tokens`。

这种设计逻辑自洽：高分辨率尺度负责细节（发丝、瞳孔高光）、中间尺度负责局部语义（肤色、衣物主色）、低分辨率提供全局色调兜底。

### 实现位置
- `MultiScaleGridTokenConditioner`：`basicsr/archs/ddcolor_arch_utils/region_tokens.py`（支持 `grid_sizes` 列表）。
- 模型入口：`basicsr/models/color_model.py`（通过 `train.cond_opt.token_mode='grid'` 并传 `grid_sizes`）。
- 数据集端：`basicsr/data/lab_dataset.py`（参数 `cond_need_masks=false` 时不生成掩码，仅返回 `ref_rgb`）。

### 建议配置（示例）
```yaml
datasets:
  train:
    cond_enable: true
    cond_need_masks: false

train:
  cond_opt:
    enable: true
    token_mode: grid
    num_scales: 3
    hidden_dim: 256
    grid_sizes: [16, 8, 4]
```

### Token 数量与性能
- 若使用 `[16,8,4]`，则 token 数 S = 256 + 64 + 16 = 336。
- `dec_layers=9` 时，cond cross-attn 每层会对这 336 个 token 做一次 attention，总共 9 次 cond-attn。
- 注意：token 数增长会线性增加 attention 的时间/显存成本（复杂度约为 O(Q*S*d)）。建议在显存受限时降低 `grid_sizes` 或采用下采样/合并策略。

### 工程建议（资源与质量折中）
- 论文优先级：保留 `[16,8,4]` 展示效果；工程部署时可改为 `[8,4,2]` 或仅对 norm1 使用 8x8。 
- 若需节省显存：在 `MultiScaleGridTokenConditioner` 对高分辨率 pooled 特征先降通道（1x1 conv），或对 token 做 2x2 合并/随机采样。

### 为什么更稳健
- 避免了基于人脸检测/分割失败导致的硬边界与色块伪影；
- 允许注意力自由选择参考上对当前 query 有用的局部信息；
- 与多尺度 image cross-attn 结合，参考信息可在多次迭代中逐层细化并正确落到空间位置。

如果你同意，我可以：
- 生成一组对比 config（`configs/pyramid_grid/`），包含 baseline region、uniform grid(4×4) 与 pyramid grid(16/8/4)；
- 或运行一次小规模 smoke 测试（`total_iter=200`）记录显存与步时。ss