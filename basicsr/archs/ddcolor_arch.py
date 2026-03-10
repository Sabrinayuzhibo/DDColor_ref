import torch
import torch.nn as nn

from basicsr.archs.ddcolor_arch_utils.unet import Hook, CustomPixelShuffle_ICNR,  UnetBlockWide, NormType, custom_conv_layer
from basicsr.archs.ddcolor_arch_utils.convnext import ConvNeXt
from basicsr.archs.ddcolor_arch_utils.transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from basicsr.archs.ddcolor_arch_utils.position_encoding import PositionEmbeddingSine
from basicsr.archs.ddcolor_arch_utils.transformer import Transformer
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DDColor(nn.Module):

    def __init__(self,
                 encoder_name='convnext-l',
                 decoder_name='MultiScaleColorDecoder',
                 num_input_channels=3,
                 input_size=(256, 256),
                 nf=512,
                 num_output_channels=3,
                 last_norm='Weight',
                 do_normalize=False,
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 use_cond_gate=False,
                 cond_gate_init=0.0,
                 encoder_from_pretrain=False):
        super().__init__()

        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'], from_pretrain=encoder_from_pretrain)
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)

        with torch.no_grad():
            self.encoder(test_input)

        self.decoder = Decoder(
            self.encoder.hooks,
            nf=nf,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
            use_cond_gate=use_cond_gate,
            cond_gate_init=cond_gate_init,
            decoder_name=decoder_name
        )
        self.refine_net = nn.Sequential(custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False, norm_type=NormType.Spectral))
    
        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def extract_condition_features(self, x, use_pixel_decoder=True):
        """提取给 conditioner 使用的三尺度特征。"""
        if x.shape[1] == 3:
            x = self.normalize(x)

        self.encoder(x)
        if use_pixel_decoder:
            return self.decoder.get_condition_features()

        hooks = self.encoder.hooks
        return [hooks[1].feature, hooks[2].feature, hooks[3].feature]

    def forward(
        self,
        x,
        cond_tokens_per_scale=None,
        cond_pos_per_scale=None,
        # 向后兼容：支持旧脚本使用的别名参数
        cond_tokens=None,
        cond_pos=None,
    ):
        # 同时支持新接口 `(cond_tokens_per_scale, cond_pos_per_scale)`
        # 与旧接口 `(cond_tokens, cond_pos)`；若旧接口传入则覆盖前者。
        if cond_tokens is not None:
            cond_tokens_per_scale = cond_tokens
        if cond_pos is not None:
            cond_pos_per_scale = cond_pos

        if x.shape[1] == 3:
            x = self.normalize(x)
        
        self.encoder(x)
        out_feat = self.decoder(
            cond_tokens_per_scale=cond_tokens_per_scale,
            cond_pos_per_scale=cond_pos_per_scale,
        )
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)

        if self.do_normalize:
            out = self.denormalize(out)
        return out


class Decoder(nn.Module):

    def __init__(self,
                 hooks,
                 nf=512,
                 blur=True,
                 last_norm='Weight',
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 use_cond_gate=False,
                 cond_gate_init=0.0,
                 decoder_name='MultiScaleColorDecoder'):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)
        self.decoder_name = decoder_name

        self.layers = self.make_layers()
        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)


        if self.decoder_name == 'MultiScaleColorDecoder':
            self.color_decoder = MultiScaleColorDecoder(
                in_channels=[512, 512, 256],
                num_queries=num_queries,
                num_scales=num_scales,
                dec_layers=dec_layers,
                use_cond_gate=use_cond_gate,
                cond_gate_init=cond_gate_init,
            )
        else:
            self.color_decoder = SingleColorDecoder(
                in_channels=hooks[-1].feature.shape[1], 
                num_queries=num_queries,
            )


    def forward(self, cond_tokens_per_scale=None, cond_pos_per_scale=None):
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0) 
        out2 = self.layers[2](out1) 
        out3 = self.last_shuf(out2) 

        if self.decoder_name == 'MultiScaleColorDecoder':
            out = self.color_decoder(
                [out0, out1, out2],
                out3,
                cond_tokens_per_scale=cond_tokens_per_scale,
                cond_pos_per_scale=cond_pos_per_scale,
            )
        else:
            out = self.color_decoder(out3, encode_feat)

        return out

    def get_condition_features(self):
        """返回 pixel decoder 的三尺度特征（out0/out1/out2）。"""
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0)
        out2 = self.layers[2](out1)
        return [out0, out1, out2]

    def make_layers(self):
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c

        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(
                    in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c
        return nn.Sequential(*decoder_layers)


class Encoder(nn.Module):

    def __init__(self, encoder_name, hook_names, from_pretrain, **kwargs):
        super().__init__()
 
        if encoder_name == 'convnext-t' or encoder_name == 'convnext':
            self.arch = ConvNeXt()
        elif encoder_name == 'convnext-s':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == 'convnext-b':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == 'convnext-l':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError

        self.encoder_name = encoder_name
        self.hook_names = hook_names
        self.hooks = self.setup_hooks()

        if from_pretrain:
            self.load_pretrain_model()

    def setup_hooks(self):
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks

    def forward(self, x):
        return self.arch(x)
    
    def load_pretrain_model(self):
        if self.encoder_name == 'convnext-t' or self.encoder_name == 'convnext':
            self.load('pretrain/convnext_tiny_22k_224.pth')
        elif self.encoder_name == 'convnext-s':
            self.load('pretrain/convnext_small_22k_224.pth')
        elif self.encoder_name == 'convnext-b':
            self.load('pretrain/convnext_base_22k_224.pth')
        elif self.encoder_name == 'convnext-l':
            self.load('pretrain/convnext_large_22k_224.pth')
        else:
            raise NotImplementedError
        print('Loaded pretrained convnext model.')

    def load(self, path):
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        if not path:
            logger.info("No checkpoint found. Initializing model from scratch")
            return
        logger.info("[Encoder] Loading from {} ...".format(path))
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        incompatible = self.arch.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            msg = "Some model parameters or buffers are not found in the checkpoint:\n"
            msg += str(incompatible.missing_keys)
            logger.warning(msg)
        if incompatible.unexpected_keys:
            msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
            msg += str(incompatible.unexpected_keys)
            logger.warning(msg)


class MultiScaleColorDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        pre_norm=False,
        color_embed_dim=256,
        enforce_input_project=True,
        num_scales=3,
        use_cond_gate=False,
        cond_gate_init=0.0,
    ):
        super().__init__()

        # 位置编码
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # 构建 Transformer 解码器
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        # image cross-attn（对编码器图像特征）
        self.transformer_cross_attention_layers = nn.ModuleList()
        # 额外的 cond cross-attn（对参考条件 tokens）
        self.transformer_cond_cross_attention_layers = nn.ModuleList()
        # cond 分支：在 cond cross-attn 后补充 Add&Norm + cond self-attn
        self.cond_dropout1 = nn.ModuleList()
        self.cond_norm1 = nn.ModuleList()
        self.cond_self_atten = nn.ModuleList()
        self.cond_dropout2 = nn.ModuleList()
        self.cond_norm2 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cond_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.cond_dropout1.append(nn.Dropout(0.0))
            self.cond_norm1.append(nn.LayerNorm(hidden_dim))
            self.cond_self_atten.append(nn.MultiheadAttention(hidden_dim, nheads, dropout=0.0))
            self.cond_dropout2.append(nn.Dropout(0.0))
            self.cond_norm2.append(nn.LayerNorm(hidden_dim))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # 可学习的颜色 query 特征
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # 可学习的颜色 query 位置编码
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 多尺度 level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # 条件 cross-attn 的可选 gate（按尺度设置，而非按层）
        # 理由：同一尺度的 cond token 应该使用相同的 gate，更符合语义
        self.use_cond_gate = use_cond_gate
        if self.use_cond_gate:
            init = float(cond_gate_init)
            # 改为按尺度设置：num_feature_levels 个 gate（通常是 3）
            self.cond_gate_logit = nn.Parameter(torch.full((self.num_feature_levels,), init))
        else:
            self.register_parameter('cond_gate_logit', None)

        # 输入投影层
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        # 输出 FFN
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

        # Debug flag: log cond branch statistics only once.
        self._cond_debug_logged = False

    def forward(self, x, img_features, cond_tokens_per_scale=None, cond_pos_per_scale=None):
        # x 是多尺度特征列表
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # 将 NxCxHxW 展平为 HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)    

        _, bs, _ = src[0].shape

        # Query 形状：QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # 1) 先对当前尺度的条件 tokens 做 cross-attn（若提供）
            if cond_tokens_per_scale is not None:
                query_before_cond = output
                cond_tokens = None
                cond_pos = None

                if isinstance(cond_tokens_per_scale, (list, tuple)):
                    cond_tokens = cond_tokens_per_scale[level_index]
                    if cond_pos_per_scale is not None:
                        cond_pos = cond_pos_per_scale[level_index]
                elif torch.is_tensor(cond_tokens_per_scale):
                    # Flatten+Concat mode: cond tokens are (B, N_total, C).
                    # Convert to MHA memory layout (N_total, B, C).
                    if cond_tokens_per_scale.dim() != 3:
                        raise ValueError(
                            f'Expected cond tokens with 3 dims, got shape={tuple(cond_tokens_per_scale.shape)}'
                        )
                    cond_tokens = cond_tokens_per_scale.transpose(0, 1).contiguous()
                    if cond_pos_per_scale is not None:
                        if not torch.is_tensor(cond_pos_per_scale) or cond_pos_per_scale.dim() != 3:
                            raise ValueError('cond_pos in flatten mode must be a 3D tensor or None.')
                        cond_pos = cond_pos_per_scale.transpose(0, 1).contiguous()
                else:
                    raise TypeError('cond_tokens_per_scale must be list/tuple or tensor.')

                # 阶段 1：Reference Cross-Attention
                cond_out = self.transformer_cond_cross_attention_layers[i](
                    output,
                    cond_tokens,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=cond_pos,
                    query_pos=query_embed,
                    gate=None,
                    return_attn_only=True,
                )

                # 阶段 2：Add & Norm 1（融合 cond_gate）
                gate_val = 1.0
                if self.use_cond_gate and self.cond_gate_logit is not None:
                    gate_val = torch.sigmoid(self.cond_gate_logit[level_index]).view(1, 1, 1)
                output = query_before_cond + gate_val * self.cond_dropout1[i](cond_out)
                output = self.cond_norm1[i](output)

                # 阶段 3：Condition Self-Attention（Query 间统筹）
                q = output + query_embed
                k = output + query_embed
                cond_self_out = self.cond_self_atten[i](q, k, output)[0]
                output = output + self.cond_dropout2[i](cond_self_out)
                output = self.cond_norm2[i](output)

                # One-time runtime sanity check for condition branch statistics.
                if (not self._cond_debug_logged) and i == 0:
                    with torch.no_grad():
                        logger = get_root_logger()
                        cond_out_d = cond_out.detach()
                        cond_self_d = cond_self_out.detach()
                        logger.info(
                            '[CondDebug] level=%d cond_out shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f',
                            level_index,
                            tuple(cond_out_d.shape),
                            float(cond_out_d.mean().item()),
                            float(cond_out_d.std(unbiased=False).item()),
                            float(cond_out_d.min().item()),
                            float(cond_out_d.max().item()),
                        )
                        logger.info(
                            '[CondDebug] level=%d cond_self_out shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f',
                            level_index,
                            tuple(cond_self_d.shape),
                            float(cond_self_d.mean().item()),
                            float(cond_self_d.std(unbiased=False).item()),
                            float(cond_self_d.min().item()),
                            float(cond_self_d.max().item()),
                        )
                    self._cond_debug_logged = True

            # 2) 再对当前尺度图像特征做原始 cross-attn
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out


class SingleColorDecoder(nn.Module):

    def __init__(
        self,
        in_channels=768, 
        hidden_dim=256,
        num_queries=256,  # 100
        nheads=8,
        dropout=0.1,
        dim_feedforward=2048,
        enc_layers=0,
        dec_layers=6,
        pre_norm=False,
        deep_supervision=True,
        enforce_input_project=True,
    ):
        
        super().__init__()

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        self.num_queries = num_queries
        self.transformer = transformer
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            nn.init.kaiming_uniform_(self.input_proj.weight, a=1)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0)
        else:
            self.input_proj = nn.Sequential()


    def forward(self, img_features, encode_feat):
        pos = self.pe_layer(encode_feat)
        src = encode_feat
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        color_embed = hs[-1]
        color_preds = torch.einsum('bqc,bchw->bqhw', color_embed, img_features)
        return color_preds

