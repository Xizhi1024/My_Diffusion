"""
连续时间高斯扩散模型 Continuous Time Gaussian Diffusion
基于论文: https://openreview.net/forum?id=2LdBqxc1Yv

这是与原 Diffusion/diffusion 项目中相同的连续时间扩散实现
"""

import math
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# neural net helpers

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

class MonotonicLinear(nn.Module):
    """单调线性层 - 用于学习噪声调度"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())

# continuous schedules
# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def beta_linear_log_snr(t):
    """线性 beta 调度的 log(SNR)"""
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):
    """余弦 alpha 调度的 log(SNR)"""
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

class learned_noise_schedule(nn.Module):
    """
    学习噪声调度
    described in section H and then I.2 of the supplementary material for variational ddpm paper
    """

    def __init__(
        self,
        *,
        log_snr_max,
        log_snr_min,
        hidden_dim = 1024,
        frac_gradient = 1.
    ):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        self.net = nn.Sequential(
            Rearrange('... -> ... 1'),
            MonotonicLinear(1, 1),
            Residual(nn.Sequential(
                MonotonicLinear(1, hidden_dim),
                nn.Sigmoid(),
                MonotonicLinear(hidden_dim, 1)
            )),
            Rearrange('... 1 -> ...'),
        )

        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device

        out_zero = self.net(torch.zeros_like(x))
        out_one =  self.net(torch.ones_like(x))

        x = self.net(x)

        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)

class ContinuousTimeGaussianDiffusion(nn.Module):
    """
    连续时间高斯扩散模型

    与原 Diffusion/diffusion 项目中的实现完全相同

    使用连续时间参数 (0 到 1) 而不是离散时间步
    支持：
    - 线性噪声调度
    - 余弦噪声调度
    - 学习噪声调度
    - Min-SNR 损失加权
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        noise_schedule = 'linear',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,   # between 0 and 1, determines what percentage of gradients go back, so one can update the learned noise schedule more slowly
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond, "模型需要支持随机或学习的正弦位置嵌入"
        assert not model.self_condition, '暂不支持 self-condition'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # continuous noise schedule related stuff

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # proposed https://arxiv.org/abs/2303.09556

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        """
        计算预测均值和方差

        参考: reviewer found an error in the equation in the paper (missing sigma)
        following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt
        """
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred_noise = self.model(x, batch_log_snr)

        if self.clip_sample_denoised:
            x_start = (x - sigma * pred_noise) / alpha

            # in Imagen, this was changed to dynamic thresholding
            x_start.clamp_(-1., 1.)

            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        """单步采样"""
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """完整采样循环"""
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        """采样接口"""
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # training related functions - noise prediction

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, times, noise = None):
        """
        前向扩散过程 q(x_t | x_0)

        添加噪声到原始图像
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    def random_times(self, batch_size):
        """
        采样随机时间步

        与离散版本不同，连续版本使用 [0, 1] 范围的均匀分布
        """
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise = None):
        """
        计算损失

        支持 Min-SNR 损失加权: https://arxiv.org/abs/2303.09556
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        model_out = self.model(x, log_snr)

        losses = F.mse_loss(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min = self.min_snr_gamma) / snr
            losses = losses * loss_weight

        return losses.mean()

    def forward(self, img, *args, **kwargs):
        """
        前向传播 - 训练模式

        Args:
            img: 输入图像，范围 [0, 1]
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, *args, **kwargs)


# 条件生成版本 - 适配 CT->PET 任务
class ContinuousTimeGaussianDiffusionConditional(nn.Module):
    """
    条件连续时间扩散模型

    扩展原始的 ContinuousTimeGaussianDiffusion 以支持条件输入 (如 CT -> PET)
    """
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 1,  # 单通道输出 (PET)
        condition_channels = 1,  # 单通道条件 (CT)
        noise_schedule = 'linear',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        super().__init__()
        # 基础模型参数
        self.base_diffusion = ContinuousTimeGaussianDiffusion(
            model=model,
            image_size=image_size,
            channels=channels,
            noise_schedule=noise_schedule,
            num_sample_steps=num_sample_steps,
            clip_sample_denoised=clip_sample_denoised,
            learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim,
            learned_noise_schedule_frac_gradient=learned_noise_schedule_frac_gradient,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma
        )

        self.channels = channels
        self.condition_channels = condition_channels
        self.image_size = image_size

    @property
    def device(self):
        return self.base_diffusion.device

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, times, noise = None):
        """前向扩散过程 - 只对目标图像加噪"""
        return self.base_diffusion.q_sample(x_start, times, noise)

    def random_times(self, batch_size):
        """采样随机时间步"""
        return self.base_diffusion.random_times(batch_size)

    def p_losses(self, x_start, cond_img, times, noise = None):
        """
        计算条件损失

        Args:
            x_start: 目标图像 (PET)
            cond_img: 条件图像 (CT)
            times: 时间步
            noise: 噪声
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)

        # 拼接噪声图像和条件图像
        model_input = torch.cat([x, cond_img], dim = 1)
        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])

        model_out = self.base_diffusion.model(model_input, batch_log_snr)

        losses = F.mse_loss(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        if self.base_diffusion.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min = self.base_diffusion.min_snr_gamma) / snr
            losses = losses * loss_weight

        return losses.mean()

    def forward(self, target_img, cond_img, *args, **kwargs):
        """
        前向传播 - 训练模式

        Args:
            target_img: 目标图像 (PET), 范围 [0, 1]
            cond_img: 条件图像 (CT), 范围 [0, 1]
        """
        b, c, h, w, device, img_size = *target_img.shape, target_img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        times = self.random_times(b)

        target_img = normalize_to_neg_one_to_one(target_img)
        cond_img = normalize_to_neg_one_to_one(cond_img)

        return self.p_losses(target_img, cond_img, times, *args, **kwargs)


# 示例使用
if __name__ == '__main__':
    # 需要一个支持随机或学习正弦嵌入的模型
    # 这里创建一个简单的示例模型

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.random_or_learned_sinusoidal_cond = True
            self.self_condition = False

        def forward(self, x, log_snr):
            # 简单返回噪声
            return torch.randn_like(x)

    # 测试连续时间扩散
    model = DummyModel()

    diffusion = ContinuousTimeGaussianDiffusion(
        model=model,
        image_size=64,
        channels=3,
        noise_schedule='linear',
        num_sample_steps=100
    )

    # 测试训练
    images = torch.randn(4, 3, 64, 64)
    loss = diffusion(images)
    print(f"Training loss: {loss.item()}")

    # 测试采样
    samples = diffusion.sample(batch_size=2)
    print(f"Sampled shape: {samples.shape}")

    print("ContinuousTimeGaussianDiffusion test passed!")
