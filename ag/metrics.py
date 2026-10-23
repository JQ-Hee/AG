import torch
from torch.nn import functional as F


METRIC_NAMES = ('DSC', 'TPR', 'PPV', 'PSNR', 'SSIM', 'MSE-ROI', 'MSE-PN')


def ssim_per_sample(pred, target, win_size=11, sigma=1.5):
    pred, target = pred.float(), target.float()
    dist = torch.arange((1 - win_size) / 2, (1 + win_size) / 2,
                        step=1, device=pred.device)
    gaussian = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    gaussian = gaussian / gaussian.sum()
    kernel_2d = torch.matmul(gaussian.unsqueeze(1), gaussian.unsqueeze(0))
    kernel_3d = torch.mul(kernel_2d.unsqueeze(-1).repeat(1, 1, win_size),
                          gaussian[None, None, :].expand(win_size, win_size, win_size))
    channels = pred.shape[1]
    kernel = kernel_3d[None, None, ...].expand(channels, 1, win_size, win_size, win_size)
    mu_p = F.conv3d(pred, kernel, groups=channels)
    mu_t = F.conv3d(target, kernel, groups=channels)
    mu_p2 = F.conv3d(pred * pred, kernel, groups=channels)
    mu_t2 = F.conv3d(target * target, kernel, groups=channels)
    mu_pt = F.conv3d(pred * target, kernel, groups=channels)
    var_p, var_t = mu_p2 - mu_p * mu_p, mu_t2 - mu_t * mu_t
    cov_pt = mu_pt - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov_pt + c2)) / (
        (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)
    )
    return ssim.reshape(pred.shape[0], -1).mean(dim=1)


def expanded_bbox_mse(pred, target, mask, expand=1.1):
    roi = torch.zeros_like(mask, dtype=pred.dtype)
    for index in range(mask.shape[0]):
        foreground = torch.where(mask[index, 0] > 0)
        if foreground[0].numel() == 0:
            continue
        slices = []
        for coordinates, size in zip(foreground, mask.shape[2:]):
            minimum, maximum = int(coordinates.min()), int(coordinates.max())
            center = (maximum + minimum) / 2.0
            edge = (maximum - minimum + 1) / 2.0 * expand
            slices.append(slice(int(max(0, center - edge)),
                                int(min(size - 1, center + edge))))
        roi[(index, 0, *slices)] = 1.0
    difference = (pred.float() - target.float()).square() * roi.float()
    return difference.sum(dim=(1, 2, 3, 4)) / roi[0].numel()


def batch_metrics(output, batch, threshold=0.5):
    dims = (1, 2, 3, 4)
    pred_prob = output["pred_mask"].clamp(0, 1)
    pred = (pred_prob > threshold).float()
    target = (batch["mask_t1"] > 0.5).float()
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    smooth = 1e-5
    mse = (output["pred_ct"].float() - batch["ct_t1"].float()).square().mean(dim=dims)
    return {
        "PSNR": -10.0 * torch.log10(mse + 1e-10),
        "SSIM": ssim_per_sample(output["pred_ct"], batch["ct_t1"]),
        "MSE-ROI": mse,
        "MSE-PN": expanded_bbox_mse(output["pred_ct"], batch["ct_t1"], target),
        "DSC": (2 * tp + smooth) / (2 * tp + fp + fn + smooth),
        "PPV": (tp + smooth) / (tp + fp + smooth),
        "TPR": (tp + smooth) / (tp + fn + smooth),
    }
