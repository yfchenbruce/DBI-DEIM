import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EdgeKernels:
    """边缘提取相关卷积核。"""

    @staticmethod
    def gaussian_kernel_2d(ksize: int, sigma: float = 0.0) -> torch.Tensor:
        if ksize <= 1:
            return torch.ones((1, 1), dtype=torch.float32)
        if sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        center = ksize // 2
        x = torch.arange(ksize, dtype=torch.float32) - center
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        return gauss_2d / gauss_2d.sum()

    @staticmethod
    def sobel_kernel_x(ksize: int = 3) -> torch.Tensor:
        if ksize == 3:
            return torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        if ksize == 5:
            return torch.tensor(
                [
                    [-1.0, -2.0, 0.0, 2.0, 1.0],
                    [-4.0, -8.0, 0.0, 8.0, 4.0],
                    [-6.0, -12.0, 0.0, 12.0, 6.0],
                    [-4.0, -8.0, 0.0, 8.0, 4.0],
                    [-1.0, -2.0, 0.0, 2.0, 1.0],
                ],
                dtype=torch.float32,
            )
        raise ValueError(f"sobel_ksize 必须为 3 或 5, got {ksize}")

    @staticmethod
    def sobel_kernel_y(ksize: int = 3) -> torch.Tensor:
        return EdgeKernels.sobel_kernel_x(ksize).t()

    @staticmethod
    def scharr_kernel_x() -> torch.Tensor:
        return torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            dtype=torch.float32,
        )

    @staticmethod
    def scharr_kernel_y() -> torch.Tensor:
        return EdgeKernels.scharr_kernel_x().t()


class EdgeFeatureGenerator(nn.Module):
    """
    轻量边缘强度生成网络（GPU/CPU 统一）。

    输出为单通道 edge strength（建议 normalize=True 后处于 0~1）。
    """

    def __init__(
        self,
        gaussian_ksize: int = 5,
        sobel_ksize: int = 3,
        use_scharr: bool = False,
        normalize: bool = True,
        gamma: float = 1.0,
        binarize: bool = False,
        threshold: Optional[float] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        self.gaussian_ksize = int(gaussian_ksize)
        self.sobel_ksize = int(sobel_ksize)
        self.use_scharr = bool(use_scharr)
        self.normalize = bool(normalize)
        self.gamma = float(gamma)
        if self.gamma <= 0:
            raise ValueError("gamma 必须 > 0")
        self.binarize = bool(binarize)
        self.threshold = threshold
        if self.threshold is not None and not (0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("threshold 必须在 [0,1] 内（归一化后阈值）")

        if device is None:
            raise ValueError("device 不能为空：请在 EdgeGen 中显式指定 device（或使用其默认 device 参数）。")
        self.device = torch.device(device)
        self._register_kernels()
        self.to(self.device)

    def _register_kernels(self) -> None:
        gauss = EdgeKernels.gaussian_kernel_2d(self.gaussian_ksize)
        self.register_buffer("gaussian_kernel", gauss.view(1, 1, self.gaussian_ksize, self.gaussian_ksize))

        if self.use_scharr:
            gx = EdgeKernels.scharr_kernel_x()
            gy = EdgeKernels.scharr_kernel_y()
            k = 3
        else:
            gx = EdgeKernels.sobel_kernel_x(self.sobel_ksize)
            gy = EdgeKernels.sobel_kernel_y(self.sobel_ksize)
            k = self.sobel_ksize
        self.register_buffer("grad_x", gx.view(1, 1, k, k))
        self.register_buffer("grad_y", gy.view(1, 1, k, k))

    def _conv2d_reflect(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        ksize = kernel.shape[-1]
        pad = ksize // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_padded, kernel)

    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        if x.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
            return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        return x.mean(dim=1, keepdim=True)

    def _normalize_01(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_flat = x.view(b, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        denom = torch.clamp(max_val - min_val, min=1e-8)
        return (x - min_val) / denom

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        # 约定：输入为 0~1 浮点 RGB
        x = x.clamp_(0.0, 1.0)
        gray = self._rgb_to_gray(x)

        # 平滑（降低噪声对梯度的影响）
        if self.gaussian_ksize > 1:
            gray = self._conv2d_reflect(gray, self.gaussian_kernel)

        gx = self._conv2d_reflect(gray, self.grad_x)
        gy = self._conv2d_reflect(gray, self.grad_y)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)

        if self.normalize:
            edge = self._normalize_01(edge)

        if self.gamma != 1.0:
            edge = torch.pow(edge, self.gamma)

        if self.binarize:
            thr = float(self.threshold) if self.threshold is not None else 0.5
            edge = (edge >= thr).to(edge.dtype)

        return edge  # (B,1,H,W)


