# Code inspired by U-Sleep article
# and https://github.com/neergaard/utime-pytorch

import torch
import torch.nn as nn
import math

class ConvBNELU(nn.Module):
    def __init__(
        self, in_channels, out_channels=6, kernel_size=9, dilation=1, ceil_pad=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1
        ) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(
                padding=(self.padding, self.padding), value=0
            ),  # https://iq.opengenus.org/output-size-of-convolution/
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.ELU(),
            nn.BatchNorm1d(self.out_channels),
        )
        self.ceil_pad = ceil_pad
        self.ceil_padding = nn.Sequential(nn.ConstantPad1d(padding=(0, 1), value=0))

        nn.init.xavier_uniform_(
            self.layers[1].weight
        )  # Initializing weights for the conv1d layer
        nn.init.zeros_(
            self.layers[1].bias
        )  # Initializing biases as zeros for the conv1d layer

    def forward(self, x):
        x = self.layers(x)

        # Added padding after since changing decoder kernel from 9 to 2 introduced mismatch
        if (self.ceil_pad) and (x.shape[2] % 2 == 1):  # Pad 1 if dimension is uneven
            x = self.ceil_padding(x)

        return x

class Encoder(nn.Module):
    def __init__(
        self,
        filters,
        max_filters,
        in_channels=2,
        maxpool_kernel=2,
        kernel_size=9,
        dilation=1,
    ):
        super().__init__()

        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernel = maxpool_kernel
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.depth = len(self.filters)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNELU(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                        ceil_pad=True,
                    )
                )
                for k in range(self.depth)
            ]
        )

        self.maxpools = nn.ModuleList(
            [nn.MaxPool1d(self.maxpool_kernel) for k in range(self.depth)]
        )

        self.bottom = nn.Sequential(  #
            ConvBNELU(
                in_channels=self.filters[-1],
                out_channels=max_filters,
                kernel_size=self.kernel_size,
            )
        )

    def forward(self, x):
        shortcuts = []  # Residual connections
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        encoded = self.bottom(x)

        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(
        self,
        filters,
        max_filters,
        upsample_kernel=2,
        kernel_size=9,
    ):
        super().__init__()

        self.filters = filters
        self.upsample_kernel = upsample_kernel
        self.in_channels = max_filters
        self.kernel_size = kernel_size

        self.depth = len(self.filters)

        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=self.upsample_kernel),
                    ConvBNELU(
                        in_channels=self.in_channels
                        if k == 0
                        else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.upsample_kernel,
                        ceil_pad=True,
                    ),
                )
                for k in range(self.depth)
            ]
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNELU(
                        in_channels=self.filters[k] * 2,
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        ceil_pad=True,
                    )
                )
                for k in range(self.depth)
            ]
        )

    def CropToMatch(self, input, shortcut):
        diff = max(0, input.shape[2] - shortcut.shape[2])
        start = diff // 2 + diff % 2

        return input[:, :, start : start + shortcut.shape[2]]

    def forward(self, z, shortcuts):
        for upsample, block, shortcut in zip(
            self.upsamples, self.blocks, shortcuts[::-1]
        ):  # [::-1] data is taken in reverse order
            z = upsample(z)

            if z.shape[2] != shortcut.shape[2]:
                z = self.CropToMatch(z, shortcut)

            z = torch.cat([shortcut, z], dim=1)

            z = block(z)

        return z


class Dense(nn.Module):
    def __init__(self, in_channels, num_classes=6, kernel_size=1):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.dense = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                kernel_size=self.kernel_size,
                bias=True,
            ),
            nn.Tanh(),
        )

        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.zeros_(self.dense[0].bias)

    def forward(self, z):
        z = self.dense(z)

        return z


class SegmentClassifier(nn.Module):
    def __init__(self, num_classes=5, avgpool_kernel=3840, conv1d_kernel=1):
        super().__init__()
        self.num_classes = num_classes
        self.avgpool_kernel = avgpool_kernel
        self.conv1d_kernel = conv1d_kernel

        self.avgpool = nn.AvgPool1d(self.avgpool_kernel)

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_classes + 1,
                out_channels=self.num_classes,
                kernel_size=self.conv1d_kernel,
            ),
            nn.Conv1d(
                in_channels=self.num_classes,
                out_channels=self.num_classes,
                kernel_size=self.conv1d_kernel,
            ),
        )

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, z):
        z = self.avgpool(z)
        z = self.layers(z)
        return z


class USleep(nn.Module):
    def __init__(
        self,
        num_channels = 2,
        initial_filters = 5,
        complexity_factor = 1.67,
        progression_factor = 2,
    ):
        super().__init__()

        self.initial_filters = initial_filters
        self.new_filter_factor = math.sqrt(complexity_factor)
        self.progression_factor = math.sqrt(progression_factor)
        
        encoder_filters, decoder_filters, max_filters = self.create_filters()
        
        self.encoder = Encoder(filters=encoder_filters, max_filters=max_filters, in_channels=num_channels)
        self.decoder = Decoder(filters=decoder_filters, max_filters=max_filters)
        self.dense = Dense(encoder_filters[0])
        self.classifier = SegmentClassifier()

    def create_filters(self) -> (list, list, int):
        encoder_filters = []
        decoder_filters = []
        current_filters = self.initial_filters

        for _ in range(13):
            encoder_filters.append(int(current_filters*self.new_filter_factor))
            current_filters = int(self.progression_factor*current_filters)

        max_filters = encoder_filters[-1]
        encoder_filters.pop()
        decoder_filters = encoder_filters[::-1]
        
        return encoder_filters, decoder_filters, max_filters
    
    def forward(self, x):
        x, shortcuts = self.encoder(x)
 
        x = self.decoder(x, shortcuts)

        x = self.dense(x)

        x = self.classifier(x)

        return x
