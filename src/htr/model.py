"""CRNN: CNN feature extractor + BiLSTM + linear head, trained with CTC."""

from __future__ import annotations

import torch
from torch import nn

from .charset import NUM_CLASSES
from .data import IMG_H, IMG_W


class CRNN(nn.Module):
    """Input: (B, 1, 32, 128). Output: (T, B, C) log-probs with T = 32.

    Downsampling: width /4 (two 2x2 pools), height /16 (two 2x2 pools +
    two (2,1) pools), leaving a 2-high feature map collapsed into the
    feature dimension for the recurrent layers.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 x 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 x 32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4 x 32
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 2 x 32
        )
        self.rnn = nn.LSTM(
            input_size=128 * 2,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    @property
    def time_steps(self) -> int:
        return IMG_W // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)  # (B, 128, 2, 32)
        b, c, h, w = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B, T, C*H)
        out, _ = self.rnn(feat)
        logits = self.fc(out)  # (B, T, num_classes)
        return logits.permute(1, 0, 2).log_softmax(2)  # (T, B, C) for CTC
