import torch
from typing import Tuple, List
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        # Define CNN
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels[0], kernel_size=3, stride=2, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels[0], 128, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels[1], kernel_size=3, stride=2, padding=1),  # Conv6
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels[1], 256, kernel_size=3, stride=1, padding=1),  # Conv7
            nn.ReLU(inplace=True),
            nn.Conv2d(256, output_channels[2], kernel_size=3, stride=2, padding=1),  # Conv8
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels[2], 128, kernel_size=3, stride=1, padding=1),  # Conv9
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels[3], kernel_size=3, stride=2, padding=1),  # Conv10
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels[3], 128, kernel_size=3, stride=1, padding=1),  # Conv11
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),  # Conv12
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels[4], 128, kernel_size=3, stride=1, padding=1),  # Conv13
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels[5], kernel_size=3, stride=1, padding=0),  # Conv14
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        x = self.features(x)
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

