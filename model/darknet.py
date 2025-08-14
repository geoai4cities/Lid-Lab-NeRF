import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import yaml

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                              stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out

class DarknetFeatureExtractor(nn.Module):
    def __init__(self):
        super(DarknetFeatureExtractor, self).__init__()
        self.blocks = [1, 2, 8, 8, 4]  # Darknet53 architecture
        self.strides = [2, 2, 2, 2, 2]
        self.bn_d = 0.1
        
        # input layer
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        # encoder layers
        self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                        stride=self.strides[0])
        self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                        stride=self.strides[1])
        self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                        stride=self.strides[2])
        self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                        stride=self.strides[3])
        self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                        stride=self.strides[4])
        
        self.dropout = nn.Dropout2d(0.1)

    def _make_enc_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                        kernel_size=3,
                                        stride=[1, stride], dilation=1,
                                        padding=1, bias=False)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=self.bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i),
                          block(inplanes, planes)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        features = []
        skips = {}
        os = 1

        # First layer
        x, skips, os = self.run_layer(x, self.conv1, skips, os)
        x, skips, os = self.run_layer(x, self.bn1, skips, os)
        x, skips, os = self.run_layer(x, self.relu1, skips, os)
        features.append(x)

        # Encoder blocks
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        features.append(x)
        
        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        features.append(x)
        
        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        features.append(x)
        
        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        features.append(x)
        
        x, skips, os = self.run_layer(x, self.enc5, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        features.append(x)
        
        return features, skips

class DarknetLoss(nn.Module):
    def __init__(self, device, weights_path=None):
        super(DarknetLoss, self).__init__()
        self.device = device
        self.feature_extractor = DarknetFeatureExtractor().to(device)
        
        # Load pretrained weights if provided
        if weights_path is not None:
            try:
                w_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
                self.feature_extractor.load_state_dict(w_dict, strict=True)
                print("Successfully loaded model backbone weights")
            except Exception as e:
                print("Couldn't load backbone, using random weights. Error: ", e)
            
        # Feature weights from original implementation
        self.weights = [1/32, 1/32, 1/16, 1/8, 1/4, 1/2]
        
        # Load sensor means and stds (you can modify these values based on your data)
        self.sensor_img_means = torch.tensor([0.0, 0.0, 0.0, 0.0], 
                                           dtype=torch.float).to(device)
        self.sensor_img_stds = torch.tensor([1.0, 1.0, 1.0, 1.0], 
                                          dtype=torch.float).to(device)
        
        # Freeze the network
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, range_data, points_data, mask, target_range, target_points):
        """
        Args:
            range_data: Range/depth data [batch_size, H, W]
            points_data: Point cloud data [batch_size, 3, H, W] (xyz coordinates)
            mask: Binary mask [batch_size, H, W]
            target_range: Target range data [batch_size, H, W]
            target_points: Target point cloud data [batch_size, 3, H, W]
        Returns:
            loss: Feature loss between input and target
        """
        # Prepare input data
        x = torch.cat([range_data.unsqueeze(1), points_data], dim=1)  # [B, 4, H, W]
        x = x * mask.unsqueeze(1)
        y = torch.cat([target_range.unsqueeze(1), target_points], dim=1)  # [B, 4, H, W]
        
        # Normalize inputs
        x = (x - self.sensor_img_means[None, :, None, None]) / self.sensor_img_stds[None, :, None, None]
        y = (y - self.sensor_img_means[None, :, None, None]) / self.sensor_img_stds[None, :, None, None]
        
        # Get features
        x_features, _ = self.feature_extractor(x)
        y_features, _ = self.feature_extractor(y)
        
        # Calculate loss
        loss = 0.0
        for i in range(len(x_features)):
            loss += self.weights[i] * ((x_features[i] - y_features[i]) ** 2).mean()
            
        return loss

    def set_sensor_stats(self, means, stds):
        """
        Set custom sensor statistics for normalization
        Args:
            means: Tensor of shape [4] for range and xyz means
            stds: Tensor of shape [4] for range and xyz standard deviations
        """
        self.sensor_img_means = means.to(self.device)
        self.sensor_img_stds = stds.to(self.device)

# Example usage:
"""
# Initialize
device = torch.device('cuda')
darknet_loss = DarknetLoss(device, weights_path='path/to/pretrained/weights')

# Optional: Set custom sensor statistics
means = torch.tensor([0.0, 0.0, 0.0, 0.0])  # your sensor means
stds = torch.tensor([1.0, 1.0, 1.0, 1.0])   # your sensor stds
darknet_loss.set_sensor_stats(means, stds)

# Use in training
range_data = torch.randn(batch_size, H, W)        # range/depth values
points_data = torch.randn(batch_size, 3, H, W)    # xyz coordinates
mask = torch.ones(batch_size, H, W)               # binary mask
target_range = torch.randn(batch_size, H, W)      # target range values
target_points = torch.randn(batch_size, 3, H, W)  # target xyz coordinates

loss = darknet_loss(range_data, points_data, mask, target_range, target_points)
"""