from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(CNNModel, self).__init__()
        self.config = config
        self.activation = self._get_activation(config["act_fn"])
        self.cl = self._build_conv_layers()
        self.fcl = self._build_fc_layers()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {"relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU(), "elu": nn.ELU(), "tanh": nn.Tanh()}
        return activations[name.lower()]

    def _build_conv_layers(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(
                in_channels=self.config["ch_nums"][0],
                out_channels=self.config["ch_nums"][1],
                kernel_size=self.config["conv_kernel_sizes"][0],
                padding="same",
            ),
            nn.BatchNorm1d(self.config["ch_nums"][1]),
            self.activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.config["dropout_conv"]),
            
            nn.Conv1d(
                in_channels=self.config["ch_nums"][1],
                out_channels=self.config["ch_nums"][2],
                kernel_size=self.config["conv_kernel_sizes"][1],
                padding="same",
            ),
            nn.BatchNorm1d(self.config["ch_nums"][2]),
            self.activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(
                in_channels=self.config["ch_nums"][2],
                out_channels=self.config["ch_nums"][3],
                kernel_size=self.config["conv_kernel_sizes"][2],
                padding="same",
            ),
            nn.BatchNorm1d(self.config["ch_nums"][3]),
            self.activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            
            
            

        )

    def _build_fc_layers(self) -> nn.Sequential:
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(self.config["adaptive_pool_size"]),
            nn.Flatten(),
            
            nn.Linear(self.config["adaptive_pool_size"] * self.config["ch_nums"][3], 
                self.config["fully_connected_layer"][0]),
            nn.BatchNorm1d(self.config["fully_connected_layer"][0]),
            self.activation,
            nn.Dropout(self.config["dropout_fcl"]),
            
            nn.Linear(self.config["fully_connected_layer"][0], self.config["fully_connected_layer"][1]),
            nn.BatchNorm1d(self.config["fully_connected_layer"][1]),
            self.activation,
            nn.Dropout(self.config["dropout_fcl"]),
            

            nn.Linear(self.config["fully_connected_layer"][1], self.config["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 0), "constant", 0)
        x_cl = self.cl(x)
        x_fcl = self.fcl(x_cl)
        return x_fcl


    def conv_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.cl(x)