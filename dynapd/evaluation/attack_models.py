"""Self-contained attack input adapters and neural attacker models for DynaPD."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, :, 0]
    raise ValueError(f"Expected 1D, 2D, or single-channel 3D trace array, got {arr.shape}")


def crop_or_pad_2d(x: np.ndarray, max_len: int = 5000, pad_value: float = 0.0) -> np.ndarray:
    arr = _as_2d(x)
    out = np.full((arr.shape[0], int(max_len)), pad_value, dtype=arr.dtype)
    copy_len = min(arr.shape[1], int(max_len))
    if copy_len:
        out[:, :copy_len] = arr[:, :copy_len]
    return out


def build_df_input(raw: np.ndarray, max_len: int = 5000) -> np.ndarray:
    direction = np.sign(_as_2d(raw)).astype(np.float32, copy=False)
    direction = crop_or_pad_2d(direction, max_len=max_len, pad_value=0.0)
    return direction[:, None, :]


def build_rf_tam_input(
    raw: np.ndarray,
    *,
    max_len: int = 5000,
    max_load_time: float = 80.0,
    num_slots: int = 1800,
) -> np.ndarray:
    rows = _as_2d(raw)
    tam = np.zeros((rows.shape[0], 2, int(num_slots)), dtype=np.float32)
    scale = float(int(num_slots) - 1) / float(max_load_time)
    for row_idx, row in enumerate(rows):
        nonzero = row[row != 0][: int(max_len)]
        if nonzero.size == 0:
            continue
        outgoing = nonzero[nonzero > 0]
        incoming = -nonzero[nonzero < 0]
        if outgoing.size:
            slots = np.floor(outgoing * scale).astype(np.int64)
            slots[outgoing >= float(max_load_time)] = int(num_slots) - 1
            slots = np.clip(slots, 0, int(num_slots) - 1)
            np.add.at(tam[row_idx, 0], slots, 1)
        if incoming.size:
            slots = np.floor(incoming * scale).astype(np.int64)
            slots[incoming >= float(max_load_time)] = int(num_slots) - 1
            slots = np.clip(slots, 0, int(num_slots) - 1)
            np.add.at(tam[row_idx, 1], slots, 1)
    return tam


class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        return self.conv(F.pad(x, (pad // 2, pad - pad // 2), "constant", 0))


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size: int, stride_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride_size)
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        return self.max_pool(F.pad(x, (pad // 2, pad - pad // 2), "constant", 0))


class ProjectDF(nn.Module):
    def __init__(self, length: int, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.length = int(length)
        self.layer1 = nn.Sequential(
            MyConv1dPadSame(in_channels, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            MyConv1dPadSame(32, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            MyConv1dPadSame(32, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            MyConv1dPadSame(64, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            MyConv1dPadSame(64, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            MyConv1dPadSame(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1),
        )
        self.layer4 = nn.Sequential(
            MyConv1dPadSame(128, 256, 8, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(256 * self.linear_input(), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc = nn.Linear(512, int(num_classes))

    def linear_input(self) -> int:
        result = self.length
        for _ in range(4):
            result = int(np.ceil(result / 8))
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        return self.fc(self.layer5(out))


class WflibConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size, pool_stride, dropout_p, activation):
        super().__init__()
        padding = int(kernel_size) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.MaxPool1d(pool_size, pool_stride, padding=0),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WflibDF(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        filter_num = [32, 64, 128, 256]
        self.feature_extraction = nn.Sequential(
            WflibConvBlock(1, filter_num[0], 8, 1, 8, 4, 0.1, nn.ELU),
            WflibConvBlock(filter_num[0], filter_num[1], 8, 1, 8, 4, 0.1, nn.ReLU),
            WflibConvBlock(filter_num[1], filter_num[2], 8, 1, 8, 4, 0.1, nn.ReLU),
            WflibConvBlock(filter_num[2], filter_num[3], 8, 1, 8, 4, 0.1, nn.ReLU),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filter_num[3] * 18, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extraction(x))


def make_rf_layers(cfg, in_channels=32):
    layers = []
    for value in cfg:
        if value == "M":
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv = nn.Conv1d(in_channels, int(value), kernel_size=3, stride=1, padding=1)
            layers += [conv, nn.BatchNorm1d(int(value), eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = int(value)
    return nn.Sequential(*layers)


def make_rf_first_layers(in_channels=1, out_channel=32):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
        nn.MaxPool2d((1, 3)),
        nn.Dropout(0.1),
        nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.1),
    )


class ProjectRF(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.first_layer_out_channel = 32
        self.first_layer = make_rf_first_layers()
        self.features = make_rf_layers([128, 128, "M", 256, 256, "M", 512, int(num_classes)])
        self.classifier = nn.AdaptiveAvgPool1d(1)
        self._initialize_weights()

    def _raw_to_tam(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[1] != 1 or x.shape[2] != 2:
                raise ValueError(f"RF expects TAM [B, 1, 2, L], got {tuple(x.shape)}")
            return x
        if x.dim() == 3 and x.shape[1] == 2:
            return x.unsqueeze(1)
        raise ValueError(f"RF expects TAM [B, 2, L] or [B, 1, 2, L], got {tuple(x.shape)}")

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._raw_to_tam(x)
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


def make_attack_model(attacker: str, num_classes: int, *, max_trace_length: int = 5000, df_architecture: str = "project") -> nn.Module:
    name = str(attacker).upper()
    if name == "DF":
        if str(df_architecture).lower() == "wflib":
            return WflibDF(int(num_classes))
        return ProjectDF(int(max_trace_length), int(num_classes))
    if name == "RF":
        return ProjectRF(int(num_classes))
    raise ValueError(attacker)

