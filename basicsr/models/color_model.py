import os
import torch
import itertools
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np

from basicsr.archs import build_network
from basicsr.archs.ddcolor_arch_utils.region_tokens import (
    MultiScaleDenseTokenConditioner,
    MultiScaleFlattenTokenConditioner,
    MultiScaleRegionTokenConditioner,
    RegionTokenSpec,
)
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.img_util import tensor_lab2rgb
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.utils.color_enhance import color_enhacne_blend

# 可选依赖：`custom_fid` 需要 SciPy。
# 这里做成惰性/可选导入，保证在最小环境下（不计算 FID）也能正常训练。
try:
    from basicsr.metrics.custom_fid import (
        INCEPTION_V3_FID,
        get_activations,
        calculate_activation_statistics,
        calculate_frechet_distance,
    )
except Exception:  # pragma: no cover
    INCEPTION_V3_FID = None
    get_activations = None
    calculate_activation_statistics = None
    calculate_frechet_distance = None


@MODEL_REGISTRY.register()
class ColorModel(BaseModel):
    """Colorization model for single image colorization."""

    def __init__(self, opt):
        super(ColorModel, self).__init__(opt)

        # 构建主生成器 net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Finetune 模式：只训练 conditioner（net_c），冻结主干生成器 net_g 的参数
        # 通过 options/train/*.yml 中 train.train_only_cond: true 控制
        self.train_only_cond = bool(self.opt.get('train', {}).get('train_only_cond', False)) if self.opt.get('is_train', False) else False
        # 方向B：冻结 encoder，只训练 decoder(+cond gate) 与 conditioner（net_c）
        self.train_decoder_cond = bool(self.opt.get('train', {}).get('train_decoder_cond', False)) if self.opt.get('is_train', False) else False
        # 更细粒度：仅训练 color decoder(+cond gate) 与 conditioner（net_c），不训练 pixel decoder
        self.train_color_decoder_cond = bool(self.opt.get('train', {}).get('train_color_decoder_cond', False)) if self.opt.get('is_train', False) else False

        if self.is_train:
            mode_flags = [self.train_only_cond, self.train_decoder_cond, self.train_color_decoder_cond]
            if sum(bool(x) for x in mode_flags) > 1:
                raise ValueError(
                    'train_only_cond/train_decoder_cond/train_color_decoder_cond 互斥，'
                    '请仅启用一个训练模式。'
                )

        # 可选参考条件分支 net_c（DDColor cond-B）
        self.cond_enable = bool(self.opt.get('train', {}).get('cond_opt', {}).get('enable', False)) if self.opt.get('is_train', False) else False
        self.net_c = None
        if self.cond_enable:
            cond_opt = self.opt.get('train', {}).get('cond_opt', {})
            token_mode = str(cond_opt.get('token_mode', 'dense')).lower()
            num_scales = int(cond_opt.get('num_scales', 3))
            hidden_dim = int(cond_opt.get('hidden_dim', 256))
            grid_size = int(cond_opt.get('grid_size', 16))
            include_area_frac = bool(cond_opt.get('include_area_frac', True))

            if token_mode in ('dense', 'grid'):
                self.net_c = MultiScaleDenseTokenConditioner(
                    num_scales=num_scales,
                    hidden_dim=hidden_dim,
                    grid_size=grid_size,
                )
            elif token_mode in ('multiscale_flatten', 'flatten'):
                self.net_c = MultiScaleFlattenTokenConditioner(
                    num_scales=num_scales,
                    hidden_dim=hidden_dim,
                )
            elif token_mode in ('region', 'mask'):
                self.net_c = MultiScaleRegionTokenConditioner(
                    spec=RegionTokenSpec(),
                    num_scales=num_scales,
                    hidden_dim=hidden_dim,
                    include_area_frac=include_area_frac,
                )
            else:
                raise ValueError(f"Unknown cond_opt.token_mode={token_mode!r}")

            self.net_c = self.model_to_device(self.net_c)
            self.print_network(self.net_c)

        # 如果只想训练 conditioner，则先冻结 net_g 参数
        if self.is_train and self.train_only_cond:
            for p in self.net_g.parameters():
                p.requires_grad = False
        # 方向B：冻结 encoder，只允许 decoder 分支学习
        elif self.is_train and self.train_decoder_cond:
            enc = getattr(self.net_g, 'encoder', None)
            if enc is None:
                raise ValueError('train_decoder_cond=True 但 net_g.encoder 不存在，无法仅冻结 encoder。')
            for p in enc.parameters():
                p.requires_grad = False
        # 仅训练 color decoder（含 cond gate）；冻结 encoder + pixel decoder + refine
        elif self.is_train and self.train_color_decoder_cond:
            for p in self.net_g.parameters():
                p.requires_grad = False

            dec = getattr(self.net_g, 'decoder', None)
            color_dec = getattr(dec, 'color_decoder', None) if dec is not None else None
            if color_dec is None:
                raise ValueError('train_color_decoder_cond=True 但 net_g.decoder.color_decoder 不存在。')
            for p in color_dec.parameters():
                p.requires_grad = True
        
        # 加载 net_g 预训练权重
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # 加载 net_c 预训练权重（可选）
        if self.cond_enable and self.net_c is not None:
            load_path_c = self.opt['path'].get('pretrain_network_c', None)
            if load_path_c is not None:
                param_key_c = self.opt['path'].get('param_key_c', 'params')
                self.load_network(self.net_c, load_path_c, self.opt['path'].get('strict_load_c', True), param_key_c)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # 构建 EMA 版本的 net_g
            # net_g_ema 仅用于单卡测试和保存，无需再包 DDP
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # 加载预训练模型
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # 拷贝当前 net_g 权重
            self.net_g_ema.eval()

        # 构建判别器 net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # 加载 net_d 预训练权重
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()
        if self.cond_enable and self.net_c is not None:
            self.net_c.train()

        # 构建各类损失
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # 参考图风格/感知损失（输出图 vs 参考图）
        if train_opt.get('ref_style_opt'):
            self.cri_ref_style = build_loss(train_opt['ref_style_opt']).to(self.device)
        else:
            self.cri_ref_style = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        if train_opt.get('colorfulness_opt'):
            self.cri_colorfulness = build_loss(train_opt['colorfulness_opt']).to(self.device)
        else:
            self.cri_colorfulness = None

        # 可选局部 AB 损失：强调高梯度区域（如头发/边缘）
        local_ab_opt = train_opt.get('local_ab_opt', None)
        self.local_ab_opt = local_ab_opt if (local_ab_opt and local_ab_opt.get('enable', False)) else None
        if self.local_ab_opt is not None:
            loss_weight = float(self.local_ab_opt.get('loss_weight', 0.0))
            reduction = str(self.local_ab_opt.get('reduction', 'mean'))
            if loss_weight <= 0:
                self.cri_local_ab = None
            else:
                self.cri_local_ab = build_loss({
                    'type': 'L1Loss',
                    'loss_weight': loss_weight,
                    'reduction': reduction,
                }).to(self.device)
        else:
            self.cri_local_ab = None

        # 可选参考矩匹配损失（无 mask）：对齐 RGB 全局均值/方差。
        # 作用：直接增强“颜色分布跟随参考图”的能力。
        ref_moment_opt = train_opt.get('ref_moment_opt', None)
        self.ref_moment_opt = ref_moment_opt if (ref_moment_opt and ref_moment_opt.get('enable', False)) else None

        # 可选批内对比参考损失（无 mask）：
        # 输出应更接近本样本参考图，而不是其他样本参考图。
        ref_contrast_opt = train_opt.get('ref_contrast_opt', None)
        self.ref_contrast_opt = ref_contrast_opt if (ref_contrast_opt and ref_contrast_opt.get('enable', False)) else None

        # 可选参考协方差损失（无 mask）：
        # 对齐 RGB 通道协方差/相关性，增强调色盘耦合。
        ref_cov_opt = train_opt.get('ref_cov_opt', None)
        self.ref_cov_opt = ref_cov_opt if (ref_cov_opt and ref_cov_opt.get('enable', False)) else None

        # 在收集优化器参数前，先构建 conditioner 的惰性模块（如 grid input_proj）。
        self._warmup_conditioner_lazy_modules()

        # 创建优化器与学习率调度器
        self.setup_optimizers()
        self.setup_schedulers()

        # 准备 FID 计算所需的真实数据统计缓存
        self.real_mu, self.real_sigma = None, None
        if self.opt['val'].get('metrics') is not None and self.opt['val']['metrics'].get('fid') is not None:
            self._prepare_inception_model_fid()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # 若只训练 conditioner，优化器里仅包含 net_c 参数
        optim_params_g = []
        if getattr(self, 'train_only_cond', False):
            if not (self.cond_enable and self.net_c is not None):
                raise ValueError("train_only_cond=True 但未启用 cond_opt / net_c 为空，请检查配置。")
            modules = [self.net_c]
        elif self.cond_enable and self.net_c is not None:
            modules = [self.net_g, self.net_c]
        else:
            modules = [self.net_g]

        for module in modules:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    optim_params_g.append(param)
                else:
                    logger = get_root_logger()
                    logger.info(f'Params {name} are frozen and excluded from optimizer.')

        if len(optim_params_g) == 0:
            raise ValueError('No trainable parameters collected for optimizer_g. Please check freeze settings.')

        # 生成器优化器
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # 判别器优化器
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_rgb = tensor_lab2rgb(torch.cat([self.lq, torch.zeros_like(self.lq), torch.zeros_like(self.lq)], dim=1))
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_lab = torch.cat([self.lq, self.gt], dim=1)
            self.gt_rgb = tensor_lab2rgb(self.gt_lab)

            # 可选参考图（RGB，[0,1]）
            self.ref_rgb = data.get('ref_rgb', None)
            if self.ref_rgb is not None:
                self.ref_rgb = self.ref_rgb.to(self.device)
            self.ref_path = data.get('ref_path', None)

            if self.opt['train'].get('color_enhance', False):
                for i in range(self.gt_rgb.shape[0]):
                    self.gt_rgb[i] = color_enhacne_blend(self.gt_rgb[i], factor=self.opt['train'].get('color_enhance_factor'))

    def _build_local_ab_weight(self):
        """基于 L 通道梯度构建逐像素 AB 权重（无需额外标注）。"""
        if self.local_ab_opt is None:
            return None

        focus_ratio = float(self.local_ab_opt.get('focus_ratio', 0.25))
        boost = float(self.local_ab_opt.get('boost', 4.0))
        edge_power = float(self.local_ab_opt.get('edge_power', 1.0))

        l = self.lq
        n, _, h, w = l.shape
        eps = 1e-6

        gh = torch.zeros_like(l)
        gw = torch.zeros_like(l)
        gh[:, :, 1:, :] = torch.abs(l[:, :, 1:, :] - l[:, :, :-1, :])
        gw[:, :, :, 1:] = torch.abs(l[:, :, :, 1:] - l[:, :, :, :-1])
        edge = gh + gw

        if edge_power != 1.0:
            edge = edge.pow(edge_power)

        flat = edge.view(n, -1)
        if 0.0 < focus_ratio < 1.0:
            q = max(0.0, min(1.0, 1.0 - focus_ratio))
            thr = torch.quantile(flat, q, dim=1, keepdim=True).view(n, 1, 1, 1)
            local_mask = (edge >= thr).float()
        else:
            local_mask = torch.ones_like(edge)

        edge_mean = edge.mean(dim=(2, 3), keepdim=True)
        edge_norm = edge / (edge_mean + eps)
        w_local = (1.0 + boost * local_mask) * edge_norm
        w_local = w_local.repeat(1, 2, 1, 1)
        return w_local

    def _rgb_channel_moment_distance(self, pred_rgb, ref_rgb, use_std: bool = True):
        """计算两张 RGB 图在通道全局矩上的 L1 距离。"""
        pred_mean = pred_rgb.mean(dim=(2, 3))
        ref_mean = ref_rgb.mean(dim=(2, 3))
        dist = torch.abs(pred_mean - ref_mean).mean(dim=1)
        if use_std:
            pred_std = pred_rgb.std(dim=(2, 3), unbiased=False)
            ref_std = ref_rgb.std(dim=(2, 3), unbiased=False)
            dist = dist + torch.abs(pred_std - ref_std).mean(dim=1)
        return dist

    def _rgb_channel_cov_distance(self, pred_rgb, ref_rgb, eps: float = 1e-6):
        """计算两张 RGB 图在通道协方差矩阵上的 L1 距离。"""
        b, c, h, w = pred_rgb.shape
        if c != 3:
            raise ValueError(f'Expected RGB with 3 channels, got {c}.')

        n = h * w
        pred_flat = pred_rgb.view(b, c, n)
        ref_flat = ref_rgb.view(b, c, n)

        pred_centered = pred_flat - pred_flat.mean(dim=2, keepdim=True)
        ref_centered = ref_flat - ref_flat.mean(dim=2, keepdim=True)

        denom = max(n - 1, 1)
        pred_cov = torch.bmm(pred_centered, pred_centered.transpose(1, 2)) / float(denom)
        ref_cov = torch.bmm(ref_centered, ref_centered.transpose(1, 2)) / float(denom)

        pred_std = torch.sqrt(torch.diagonal(pred_cov, dim1=1, dim2=2).clamp_min(eps))
        ref_std = torch.sqrt(torch.diagonal(ref_cov, dim1=1, dim2=2).clamp_min(eps))
        pred_norm = pred_std.unsqueeze(2) * pred_std.unsqueeze(1)
        ref_norm = ref_std.unsqueeze(2) * ref_std.unsqueeze(1)
        pred_corr = pred_cov / pred_norm.clamp_min(eps)
        ref_corr = ref_cov / ref_norm.clamp_min(eps)

        cov_dist = torch.abs(pred_cov - ref_cov).mean(dim=(1, 2))
        corr_dist = torch.abs(pred_corr - ref_corr).mean(dim=(1, 2))
        return cov_dist + corr_dist

    def _warmup_conditioner_lazy_modules(self):
        """先 warmup 一次 conditioner，确保惰性参数在构建优化器前可见。"""
        if not (self.cond_enable and self.net_c is not None):
            return

        net_g = self.get_bare_model(self.net_g)
        net_c = self.get_bare_model(self.net_c)
        logger = get_root_logger()

        grid_cond = getattr(net_c, 'grid_conditioner', None)
        if grid_cond is None:
            return
        if getattr(grid_cond, 'input_proj', None) is not None:
            return

        train_gt_size = int(self.opt.get('datasets', {}).get('train', {}).get('gt_size', 256))
        train_gt_size = max(64, train_gt_size)
        was_g_train = net_g.training
        was_c_train = net_c.training

        try:
            net_g.eval()
            net_c.eval()
            with torch.no_grad():
                dummy_ref = torch.zeros((1, 3, train_gt_size, train_gt_size), device=self.device)
                ref_feats = net_g.extract_condition_features(dummy_ref, use_pixel_decoder=True)
                _ = net_c(ref_feats)

            built = getattr(getattr(net_c, 'grid_conditioner', None), 'input_proj', None)
            if built is not None:
                logger.info(f'Conditioner warmup built input_proj with {len(built)} levels before optimizer setup.')
            else:
                logger.warning('Conditioner warmup finished but input_proj is still None.')
        except Exception as e:
            logger.warning(f'Conditioner warmup failed: {e}')
        finally:
            if was_g_train:
                net_g.train()
            if was_c_train:
                net_c.train()

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        # 优化 net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()
        
        cond_tokens_per_scale = None
        cond_pos_per_scale = None
        if self.cond_enable and self.net_c is not None and getattr(self, 'ref_rgb', None) is not None:
            cond_opt = self.opt['train'].get('cond_opt', {})
            # 历史命名沿用 freeze_ref_encoder；当前语义是：
            # 参考分支特征提取（extract_condition_features）是否在 no_grad 下执行。
            freeze_ref_encoder = bool(cond_opt.get('freeze_ref_encoder', False))
            cond_gain = float(cond_opt.get('gain', 1.0))

            # 从参考图提取条件特征（当前使用 pixel decoder 三尺度特征）并构建 cond tokens。
            # 随后对内容图执行主前向（net_g），把 cond tokens 注入 color decoder。
            if freeze_ref_encoder:
                with torch.no_grad():
                    ref_feats = self.net_g.extract_condition_features(self.ref_rgb, use_pixel_decoder=True)
            else:
                ref_feats = self.net_g.extract_condition_features(self.ref_rgb, use_pixel_decoder=True)

            cond_tokens_per_scale, cond_pos_per_scale = self.net_c(ref_feats)
            if cond_gain != 1.0 and cond_tokens_per_scale is not None:
                if isinstance(cond_tokens_per_scale, (list, tuple)):
                    cond_tokens_per_scale = [t * cond_gain for t in cond_tokens_per_scale]
                else:
                    cond_tokens_per_scale = cond_tokens_per_scale * cond_gain
        
        # 前向：可选注入条件 tokens（DDColor 支持 cond_tokens/cond_pos 别名）
        self.output_ab = self.net_g(self.lq_rgb, cond_tokens=cond_tokens_per_scale, cond_pos=cond_pos_per_scale)
        self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
        self.output_rgb = tensor_lab2rgb(self.output_lab)

        l_g_total = 0
        loss_dict = OrderedDict()
        # 像素损失
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output_ab, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        if self.cri_local_ab is not None:
            local_w = self._build_local_ab_weight()
            if local_w is not None:
                l_g_local_ab = self.cri_local_ab(self.output_ab, self.gt, weight=local_w)
                l_g_total += l_g_local_ab
                loss_dict['l_g_local_ab'] = l_g_local_ab

        # 感知损失
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output_rgb, self.gt_rgb)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # 参考风格损失（输出 vs 参考图）
        if self.cri_ref_style and getattr(self, 'ref_rgb', None) is not None:
            l_r_percep, l_r_style = self.cri_ref_style(self.output_rgb, self.ref_rgb)
            if l_r_percep is not None:
                l_g_total += l_r_percep
                loss_dict['l_ref_percep'] = l_r_percep
            if l_r_style is not None:
                l_g_total += l_r_style
                loss_dict['l_ref_style'] = l_r_style

        # 参考矩匹配（增强全局颜色跟随）
        if self.ref_moment_opt is not None and getattr(self, 'ref_rgb', None) is not None:
            w = float(self.ref_moment_opt.get('loss_weight', 0.0))
            use_std = bool(self.ref_moment_opt.get('use_std', True))
            if w > 0:
                d_pos = self._rgb_channel_moment_distance(self.output_rgb, self.ref_rgb, use_std=use_std)
                l_ref_moment = d_pos.mean() * w
                l_g_total += l_ref_moment
                loss_dict['l_ref_moment'] = l_ref_moment

        # 参考协方差匹配（增强跨通道风格跟随）
        if self.ref_cov_opt is not None and getattr(self, 'ref_rgb', None) is not None:
            w = float(self.ref_cov_opt.get('loss_weight', 0.0))
            if w > 0:
                d_cov = self._rgb_channel_cov_distance(self.output_rgb, self.ref_rgb)
                l_ref_cov = d_cov.mean() * w
                l_g_total += l_ref_cov
                loss_dict['l_ref_cov'] = l_ref_cov

        # 批内对比参考损失：本样本参考图应比负样本参考图更近
        if self.ref_contrast_opt is not None and getattr(self, 'ref_rgb', None) is not None:
            w = float(self.ref_contrast_opt.get('loss_weight', 0.0))
            margin = float(self.ref_contrast_opt.get('margin', 0.05))
            use_std = bool(self.ref_contrast_opt.get('use_std', True))
            hard_negative = bool(self.ref_contrast_opt.get('hard_negative', False))
            if w > 0 and self.output_rgb.shape[0] > 1:
                d_pos = self._rgb_channel_moment_distance(self.output_rgb, self.ref_rgb, use_std=use_std)
                if hard_negative:
                    bsz = self.output_rgb.shape[0]
                    d_neg_list = []
                    for shift in range(1, bsz):
                        neg_ref_s = torch.roll(self.ref_rgb, shifts=shift, dims=0)
                        d_neg_s = self._rgb_channel_moment_distance(self.output_rgb, neg_ref_s, use_std=use_std)
                        d_neg_list.append(d_neg_s.unsqueeze(1))
                    d_neg = torch.cat(d_neg_list, dim=1).min(dim=1).values
                else:
                    neg_ref = torch.roll(self.ref_rgb, shifts=1, dims=0)
                    d_neg = self._rgb_channel_moment_distance(self.output_rgb, neg_ref, use_std=use_std)
                l_ref_contrast = torch.relu(margin + d_pos - d_neg).mean() * w
                l_g_total += l_ref_contrast
                loss_dict['l_ref_contrast'] = l_ref_contrast

        # 可选：推动 cond gate 适度打开（避免“参考图不起作用”）
        gate_push_opt = train_opt.get('cond_gate_push_opt', None)
        if gate_push_opt and gate_push_opt.get('enable', False):
            try:
                w = float(gate_push_opt.get('loss_weight', 0.0))
                tgt = float(gate_push_opt.get('target_sigmoid', 0.45))
                if w > 0:
                    gate_logit = self.net_g.decoder.color_decoder.cond_gate_logit
                    if gate_logit is not None:
                        gate = torch.sigmoid(gate_logit)
                        l_gate = (gate - tgt).pow(2).mean() * w
                        l_g_total += l_gate
                        loss_dict['l_cond_gate_push'] = l_gate
            except Exception:
                # 即使 gate 属性路径变化，也尽量不影响训练流程
                pass
            # GAN 损失
        if self.cri_gan:
            fake_g_pred = self.net_d(self.output_rgb)
            l_g_gan = self.cri_gan(fake_g_pred, target_is_real=True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
        # 色彩丰富度损失
        if self.cri_colorfulness:
            l_g_color = self.cri_colorfulness(self.output_rgb)
            l_g_total += l_g_color
            loss_dict['l_g_color'] = l_g_color

        l_g_total.backward()
        self.optimizer_g.step()

        # 优化 net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        real_d_pred = self.net_d(self.gt_rgb)
        fake_d_pred = self.net_d(self.output_rgb.detach())
        l_d = self.cri_gan(real_d_pred, target_is_real=True, is_disc=True) + self.cri_gan(fake_d_pred, target_is_real=False, is_disc=True)
        loss_dict['l_d'] = l_d
        loss_dict['real_score'] = real_d_pred.detach().mean()
        loss_dict['fake_score'] = fake_d_pred.detach().mean()

        l_d.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq_rgb.detach().cpu()
        out_dict['result'] = self.output_rgb.detach().cpu()
        if self.opt['logger'].get('save_snapshot_verbose', False):  # 仅 verbose 模式保存
            self.output_lab_chroma = torch.cat([torch.ones_like(self.lq) * 50, self.output_ab], dim=1)
            self.output_rgb_chroma = tensor_lab2rgb(self.output_lab_chroma)
            out_dict['result_chroma'] = self.output_rgb_chroma.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt_rgb.detach().cpu()
            if self.opt['logger'].get('save_snapshot_verbose', False):  # 仅 verbose 模式保存
                self.gt_lab_chroma = torch.cat([torch.ones_like(self.lq) * 50, self.gt], dim=1)
                self.gt_rgb_chroma = tensor_lab2rgb(self.gt_lab_chroma)
                out_dict['gt_chroma'] = self.gt_rgb_chroma.detach().cpu()
        return out_dict

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output_ab = self.net_g_ema(self.lq_rgb)
                self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
                self.output_rgb = tensor_lab2rgb(self.output_lab)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output_ab = self.net_g(self.lq_rgb)
                self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
                self.output_rgb = tensor_lab2rgb(self.output_lab)
            self.net_g.train()
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # 仅首次验证时初始化
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # 初始化该数据集的最优指标记录（支持多验证集）
        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
        # 清空当前指标累计值
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        
        if self.opt['val']['metrics'].get('fid') is not None:
            fake_acts_set, acts_set = [], []

        for idx, val_data in enumerate(dataloader):
            # if idx == 100:
            #     break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            if hasattr(self, 'gt'):
                del self.gt
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_dir = osp.join(self.opt['path']['visualization'], img_name)
                    for key in visuals:
                        save_path = os.path.join(save_dir, '{}_{}.png'.format(current_iter, key))
                        img = tensor2img(visuals[key])
                        imwrite(img, save_path)
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)

            if with_metrics:
                # 计算指标
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'fid':
                        pred, gt = visuals['result'].cuda(), visuals['gt'].cuda()
                        fake_act = get_activations(pred, self.inception_model_fid, 1)
                        fake_acts_set.append(fake_act)
                        if self.real_mu is None:
                            real_act = get_activations(gt, self.inception_model_fid, 1)
                            acts_set.append(real_act)
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            if self.opt['val']['metrics'].get('fid') is not None:
                if self.real_mu is None:
                    acts_set = np.concatenate(acts_set, 0)
                    self.real_mu, self.real_sigma = calculate_activation_statistics(acts_set)
                fake_acts_set = np.concatenate(fake_acts_set, 0)
                fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)

                fid_score = calculate_frechet_distance(self.real_mu, self.real_sigma, fake_mu, fake_sigma)
                self.metric_results['fid'] = fid_score

            for metric in self.metric_results.keys():
                if metric != 'fid':
                    self.metric_results[metric] /= (idx + 1)
                # 更新最优指标
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def _prepare_inception_model_fid(self, path='pretrain/inception_v3_google-1a9a5a14.pth'):
        if INCEPTION_V3_FID is None:
            raise ImportError("FID metric requires optional dependencies (e.g. SciPy). Please `pip install scipy` to enable FID.")
        incep_state_dict = torch.load(path, map_location='cpu')
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
        self.inception_model_fid.cuda()
        self.inception_model_fid.eval()

    @master_only
    def save_training_images(self, current_iter):
        visuals = self.get_current_visuals()
        save_dir = osp.join(self.opt['root_path'], 'experiments', self.opt['name'], 'training_images_snapshot')
        os.makedirs(save_dir, exist_ok=True)

        for key in visuals:
            save_path = os.path.join(save_dir, '{}_{}.png'.format(current_iter, key))
            img = tensor2img(visuals[key])
            imwrite(img, save_path)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        if self.cond_enable and self.net_c is not None:
            self.save_network(self.net_c, 'net_c', current_iter)
        self.save_training_state(epoch, current_iter)
