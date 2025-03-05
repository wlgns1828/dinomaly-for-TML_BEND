import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.optim.optimizer import Optimizer
from functools import partial
import cv2


def compute_anomaly_map(fs_list, ft_list, out_size=(392,392)):
    if not isinstance(out_size, tuple):
        out_size = (out_size[0], out_size[1])
    anomaly_map = torch.zeros((fs_list[0].shape[0], 1, out_size[0], out_size[1]), device=fs_list[0].device)   
    for fs, ft in zip(fs_list, ft_list):
        cos_sim = F.cosine_similarity(fs, ft, dim=1)
        a_map = 1 - cos_sim  # (B, H, W)
        a_map = a_map.unsqueeze(1)  # (B, 1, H, W)
        a_map = F.interpolate(a_map, size=(out_size[0],out_size[1]), mode='bilinear', align_corners=False)
        anomaly_map += a_map
    return anomaly_map



def image_transform(image):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    transform_mvtec = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),        
        transforms.Normalize(mean=mean_train,
                                std=std_train)
    ])
    
    image = transform_mvtec(image)
    return image


import os

import os
import matplotlib.pyplot as plt

def visualize(original_images, anomaly_map, save_path="test_result", image_name = ''):
    # Tensor를 CPU로 옮기고 numpy로 변환
    original_images_np = original_images.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    if len(anomaly_map.shape) == 4:
        # 예상 shape: (1, 1, H, W)
        anomaly_map_np = anomaly_map.squeeze(0).squeeze(0).detach().cpu().numpy()
    elif len(anomaly_map.shape) == 3:
        # 예상 shape: (1, H, W)
        anomaly_map_np = anomaly_map.squeeze(0).detach().cpu().numpy()
    elif len(anomaly_map.shape) == 2:
        # 예상 shape: (H, W)
        anomaly_map_np = anomaly_map
    else:
        raise ValueError(f"Unsupported anomaly_map shape: {anomaly_map.shape}")

    anomaly_score = np.max(anomaly_map_np)
    
    # Normalize the data to [0, 1] range
    #anomaly_map_np = min_max_norm(anomaly_map_np)
    #original_images_np = min_max_norm(original_images_np)
    
    # Matplotlib를 이용한 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Anomaly Score : {anomaly_score:.4f}", fontsize=16, fontweight="bold")
    
    axes[0].imshow(original_images_np)
    axes[0].set_title("Image")
    axes[0].axis('off')

    # Heatmaps
    axes[1].imshow(anomaly_map_np, vmin = 0, vmax=0.4)
    axes[1].set_title("Unsupervised Head")
    axes[1].axis('off')

    plt.tight_layout()

    # Save the figure to the specified path
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    save_path = os.path.join(save_path, image_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # 메모리 누수 방지를 위해 plt를 닫아줍니다.
    print(f"Figure saved to {save_path}")



def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)



class StableAdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_threshold: float = 1.0
                 ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, clip_threshold=clip_threshold
                        )
        super(StableAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(StableAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    # lr_scale = torch.rsqrt(max_exp_avg_sq + 1e-16).mul_(grad)

                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    # lr_scale = torch.rsqrt(exp_avg_sq + 1e-16).mul_(grad)

                lr_scale = grad / denom
                lr_scale = max(1.0, self._rms(lr_scale) / group["clip_threshold"])

                step_size = group['lr'] / bias_correction1 / (lr_scale)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x