# 条件 Token 的 Position Embedding 设计说明

## 当前实现（问题所在）

### 代码位置
`region_tokens.py` 第 112-114 行：
```python
# 此处直接把 token 本身作为 pos：在 CrossAttentionLayer 里会做 memory+pos，
# 等价于把 token 作为 K/V 的"偏置"。
pos = tokens
```

### 当前流程

1. **生成条件 tokens**：
   - 从参考图的多尺度特征中，通过 adaptive pooling 得到 grid tokens
   - 例如：尺度 0 得到 16×16=256 个 tokens，尺度 1 得到 8×8=64 个 tokens

2. **Position embedding 的使用**：
   - 在 `CrossAttentionLayer` 中（`transformer_utils.py` 第 94 行）：
     ```python
     key = self.with_pos_embed(memory, pos)  # key = memory + pos
     ```
   - 当前实现：`pos = tokens`，所以 `key = tokens + tokens = 2 * tokens`
   - 这意味着：**token 的内容既作为 memory（值），又作为 position embedding（加到 key 上）**

### 问题分析

#### 问题 1：缺少显式的尺度信息
- **现状**：不同尺度的 tokens 在数值上可能相似，模型难以区分它们来自哪个尺度
- **影响**：模型可能混淆不同尺度的条件信息
- **类比**：就像给不同楼层的人发邮件，但没有标注楼层号

#### 问题 2：缺少空间位置信息
- **现状**：grid token 在 2D 空间中的位置（x, y 坐标）没有被编码
- **影响**：模型不知道某个 token 在参考图的哪个位置（左上角？右下角？）
- **类比**：就像知道房间里有家具，但不知道家具在房间的哪个位置

---

## 改进方案

### 方案 1：添加 Scale Embedding（尺度嵌入）

**目的**：让模型明确知道每个 token 来自哪个尺度

**实现思路**：
```python
# 在 MultiScaleGridTokenConditioner.__init__ 中添加
self.scale_embed = nn.Parameter(torch.randn(num_scales, hidden_dim))

# 在 forward 中
for level, feat in enumerate(feats):
    tokens = ...  # 生成 tokens
    # 添加尺度嵌入
    scale_emb = self.scale_embed[level].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
    pos = tokens + scale_emb  # 每个 token 都加上对应尺度的嵌入
```

**效果**：
- 尺度 0 的所有 tokens 都会加上 `scale_embed[0]`
- 尺度 1 的所有 tokens 都会加上 `scale_embed[1]`
- 模型可以学习到不同尺度的特征差异

---

### 方案 2：添加 2D 空间位置编码

**目的**：让模型知道每个 grid token 在参考图中的空间位置

**实现思路**：

#### 方法 A：Sinusoidal 位置编码（类似 Transformer）
```python
def get_2d_sinusoidal_pos_embed(hidden_dim, grid_h, grid_w):
    """生成 2D sinusoidal 位置编码"""
    # 为每个 (h, w) 位置生成编码
    pos_emb = []
    for h in range(grid_h):
        for w in range(grid_w):
            # 使用 sin/cos 编码 x 和 y 坐标
            pos_h = [math.sin(h / (10000 ** (2*i/hidden_dim))) 
                     if i % 2 == 0 else math.cos(h / (10000 ** (2*i/hidden_dim)))
                     for i in range(hidden_dim // 2)]
            pos_w = [math.sin(w / (10000 ** (2*i/hidden_dim))) 
                     if i % 2 == 0 else math.cos(w / (10000 ** (2*i/hidden_dim)))
                     for i in range(hidden_dim // 2)]
            pos_emb.append(pos_h + pos_w)
    return torch.tensor(pos_emb, dtype=torch.float32)  # (grid_h*grid_w, hidden_dim)

# 在 forward 中
for level, feat in enumerate(feats):
    tokens = ...  # (S_k, B, hidden_dim)
    gh, gw = ...  # grid 大小
    
    # 生成 2D 位置编码
    spatial_pos = get_2d_sinusoidal_pos_embed(hidden_dim, gh, gw)
    spatial_pos = spatial_pos.unsqueeze(1).to(tokens.device)  # (S_k, 1, hidden_dim)
    
    # 组合：token 内容 + 尺度嵌入 + 空间位置
    pos = tokens + scale_emb + spatial_pos
```

#### 方法 B：可学习的 2D 位置编码
```python
# 在 __init__ 中，为每个尺度创建可学习的位置编码
self.spatial_pos_emb = nn.ParameterList([
    nn.Parameter(torch.randn(grid_sizes[k]**2, hidden_dim))
    for k in range(num_scales)
])

# 在 forward 中
for level, feat in enumerate(feats):
    tokens = ...  # (S_k, B, hidden_dim)
    spatial_pos = self.spatial_pos_emb[level].unsqueeze(1)  # (S_k, 1, hidden_dim)
    pos = tokens + scale_emb + spatial_pos
```

**效果**：
- 每个 grid token 都有明确的空间位置信息
- 模型可以学习到"左上角的 token"和"右下角的 token"的区别
- 有助于模型理解参考图的空间结构

---

## 完整改进示例

### 改进后的 `MultiScaleGridTokenConditioner`

```python
class MultiScaleGridTokenConditioner(nn.Module):
    def __init__(
        self,
        num_scales: int = 3,
        hidden_dim: int = 256,
        grid_sizes: Sequence[int] = (16, 8, 4),
        use_scale_embed: bool = True,
        use_spatial_pos: bool = True,
        spatial_pos_type: str = "learnable",  # "sinusoidal" or "learnable"
    ) -> None:
        super().__init__()
        self.num_scales = int(num_scales)
        self.hidden_dim = int(hidden_dim)
        self.grid_sizes = [max(1, int(g)) for g in grid_sizes[: self.num_scales]]
        self.use_scale_embed = use_scale_embed
        self.use_spatial_pos = use_spatial_pos
        self.spatial_pos_type = spatial_pos_type
        
        # 尺度嵌入：每个尺度一个 learnable embedding
        if self.use_scale_embed:
            self.scale_embed = nn.Parameter(torch.randn(num_scales, hidden_dim))
            nn.init.normal_(self.scale_embed, std=0.02)
        
        # 空间位置编码
        if self.use_spatial_pos:
            if self.spatial_pos_type == "learnable":
                # 为每个尺度创建可学习的位置编码
                self.spatial_pos_emb = nn.ParameterList([
                    nn.Parameter(torch.randn(grid_sizes[k]**2, hidden_dim))
                    for k in range(num_scales)
                ])
                for pos_emb in self.spatial_pos_emb:
                    nn.init.normal_(pos_emb, std=0.02)
            # sinusoidal 位置编码在 forward 中动态生成，不需要参数
        
        self.input_proj: Optional[nn.ModuleList] = None

    def _get_2d_sinusoidal_pos_embed(self, hidden_dim, grid_h, grid_w, device):
        """生成 2D sinusoidal 位置编码"""
        pos_emb = []
        for h in range(grid_h):
            for w in range(grid_w):
                emb = torch.zeros(hidden_dim, device=device)
                # 交替使用 sin/cos 编码 x 和 y
                for i in range(hidden_dim // 2):
                    emb[2*i] = math.sin(h / (10000 ** (2*i / (hidden_dim // 2))))
                    emb[2*i+1] = math.cos(h / (10000 ** (2*i / (hidden_dim // 2))))
                for i in range(hidden_dim // 2, hidden_dim):
                    j = i - hidden_dim // 2
                    emb[2*j] = math.sin(w / (10000 ** (2*j / (hidden_dim // 2))))
                    emb[2*j+1] = math.cos(w / (10000 ** (2*j / (hidden_dim // 2))))
                pos_emb.append(emb)
        return torch.stack(pos_emb)  # (grid_h*grid_w, hidden_dim)

    def forward(self, ref_feats: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        # ... 前面的代码不变 ...
        
        for level, feat in enumerate(feats):
            # ... 生成 tokens 的代码不变 ...
            tokens = pooled.flatten(2).permute(2, 0, 1).contiguous()  # (S_k, B, hidden_dim)
            
            # 构建 position embedding
            pos = tokens.clone()  # 从 token 内容开始
            
            # 添加尺度嵌入
            if self.use_scale_embed:
                scale_emb = self.scale_embed[level].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
                pos = pos + scale_emb
            
            # 添加空间位置编码
            if self.use_spatial_pos:
                if self.spatial_pos_type == "learnable":
                    spatial_pos = self.spatial_pos_emb[level].unsqueeze(1)  # (S_k, 1, hidden_dim)
                else:  # sinusoidal
                    spatial_pos = self._get_2d_sinusoidal_pos_embed(
                        self.hidden_dim, gh, gw, tokens.device
                    ).unsqueeze(1)  # (S_k, 1, hidden_dim)
                pos = pos + spatial_pos
            
            cond_tokens_per_scale.append(tokens)
            cond_pos_per_scale.append(pos)
        
        return cond_tokens_per_scale, cond_pos_per_scale
```

---

## 为什么这样改进？

### 1. **语义更清晰**
- 当前：`pos = tokens` → `key = tokens + tokens`（语义模糊）
- 改进后：`pos = tokens + scale_emb + spatial_pos` → 明确包含内容、尺度、位置三种信息

### 2. **更好的可解释性**
- 模型可以学习到：
  - `scale_embed[0]` 代表"粗尺度"的特征
  - `scale_embed[1]` 代表"中等尺度"的特征
  - `spatial_pos` 代表"空间位置"的特征

### 3. **更强的表达能力**
- 不同尺度的相同空间位置可以有不同的表示
- 例如：尺度 0 的"左上角"和尺度 1 的"左上角"会有不同的 position embedding

### 4. **符合 Transformer 设计原则**
- 标准的 Transformer 使用位置编码来区分序列中的不同位置
- 这里扩展到 2D 空间和多尺度，是自然的扩展

---

## 总结

**当前问题**：
- ❌ `pos = tokens` 缺少显式的尺度信息
- ❌ 缺少空间位置编码

**改进方案**：
- ✅ 添加 `scale_embed`：每个尺度一个 learnable embedding
- ✅ 添加 `spatial_pos`：2D 空间位置编码（sinusoidal 或 learnable）

**预期效果**：
- 模型能更好地区分不同尺度的条件信息
- 模型能理解条件 token 的空间位置关系
- 提升条件上色的准确性和一致性
