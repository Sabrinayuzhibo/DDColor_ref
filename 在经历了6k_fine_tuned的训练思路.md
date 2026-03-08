Stage A：自我重建与厚涂领域特化 (Self-Reconstruction & Domain Specialization)1. 核心目标 (Objectives)当前模型致力于打造一个垂直于“厚涂人像”风格的高精度条件上色网络。Stage A 的核心目标是：打破坐标迷信：通过非对称空间增强，逼迫 Cross-Attention 放弃“抄写相同坐标的颜色”，真正学会“通过厚涂的深层语义（如发丝走向、五官高光）寻找并匹配颜色”。厚涂领域特化 (Domain Specialization)：解冻 Backbone，让原本在真实世界 (ImageNet) 预训练的 ConvNeXt 底座进行一次彻底的“知识洗牌”。用 100% 的纯厚涂数据，将它的特征提取器完全改造为“厚涂特征敏感型”雷达。完美重建：利用 self 模式（参考图 = 目标图），以极致的 L1 损失逼迫模型学会在厚涂画风下的极致光影与色彩还原。2. 核心机制：非对称空间数据增强 (Asymmetric Spatial Augmentation)这是 Stage A 防作弊与建立“语义制导”的决定性机制。问题背景：当 ref = target 时，两张图的空间坐标完全重合。Cross-Attention 会退化为“对角线映射”，不学语义，只死记硬背像素的 $(X, Y)$ 坐标。解决方案：在 Dataloader 阶段，当目标图（Target 黑白线稿/灰度图）保持原样时，对提取特征用的参考图（Reference 彩色图）施加强烈的随机裁剪 (Random Crop)、随机缩放 (Resize) 和 随机翻转 (Flip)。物理意义：此时 Reference 依然包含完美的 Target 颜色，但头发、眼睛的空间坐标全被打乱了。模型为了完成重建任务，被迫激活 Cross-Attention 在全图范围内进行“找同类”的语义搜索，从而真正建立起“目标头发特征 $\rightarrow$ 参考头发特征 $\rightarrow$ 提取对应颜色”的坚实映射路径。3. 网络解冻与特化策略 (Network Unfreezing Strategy)在 Stage A 中，我们放弃局部微调，采用全参领域自适应训练（Full Parameter Domain Adaptation）。Backbone (ConvNeXt)：🔓 彻底解冻 (freeze_ref_encoder: false)。让其彻底洗掉 ImageNet 的写实风特征，重新学习厚涂特有的笔触、色块和边缘表达。Pixel Decoder：🔓 解冻。Color Decoder & Conditioner：🔓 解冻。训练步数：建议 30,000 - 50,000 步，确保底座完成从写实到厚涂的彻底蜕变。4. 损失函数重构策略 (Loss Strategy)Stage A 的唯一目标是“完美复刻 GT 的厚涂光影”，必须压制会导致颜色在局部胡乱发散的风格损失。YAML  # 1. 像素级厚涂重建（绝对核心）
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0              # 🌟 猛烈拉高！逼迫模型精准还原每一笔厚涂色彩
    reduction: mean

  # 2. 语义护盾与结构约束
  perceptual_opt:
    perceptual_weight: 1.0        # 🌟 极高权重！保护厚涂动漫脸型、眼睛等高维结构不崩坏
    style_weight: 0               # 严禁学习自身风格，防止特征混叠

  # 3. 压制宏观风格溢出
  ref_style_opt:
    style_weight: 0.1             # 🌟 压制 Gram 矩阵！不需要它在重建阶段干扰精确对应
  
  # 4. 唤醒特征排斥
  ref_contrast_opt:
    enable: true
    loss_weight: 0.1              # 适度保留，开始培养正负样本的特征排斥意识（防止颜色糊成一团）

  # 5. 厚涂画质兜底
  gan_opt:
    loss_weight: 0.1              # 恢复正常水平，由判别器把关最终颜色的“厚涂感”与真实度
5. 数据集纯度策略 (Data Purity)数据集构成：100% 纯厚涂人像数据集。理论支撑：不再担心“灾难性遗忘”。相反，我们主动期望模型遗忘写实照片的特征，从而将有限的参数容量（Capacity）全部用于拟合复杂多变的厚涂二次元色彩规律。经过这个阶段，你的 Backbone 将成为业界独一无二的“厚涂专属特征提取器”。