import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.conv1 = nn.Conv3d(2, self.num_features, kernel_size=4, stride=2, padding=1)
        
        self.conv2 = nn.Conv3d(self.num_features, self.num_features*2, kernel_size=4, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm3d(self.num_features*2)
        
        self.conv3 = nn.Conv3d(self.num_features*2, self.num_features*4, kernel_size=4, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm3d(self.num_features*4)
        
        self.conv4 = nn.Conv3d(self.num_features*4, self.num_features*8, kernel_size=4, stride=1)
        self.enc_bn4 = nn.BatchNorm3d(self.num_features*8)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # TODO: 2 Bottleneck layers
        self.fcn1 = nn.Linear(self.num_features*8, self.num_features*8)
        self.fcn2 = nn.Linear(self.num_features*8, self.num_features*8)
        
        self.bottleneck = nn.Sequential(
            self.fcn1,
            nn.ReLU(),
            self.fcn2,
            nn.ReLU()
        )
        
        
        # TODO: 4 Decoder layers
        self.deconv1 = nn.ConvTranspose3d(self.num_features*8*2, self.num_features*4, kernel_size=4, stride=1)
        self.dec_bn1 = nn.BatchNorm3d(self.num_features*4)
        
        self.deconv2 = nn.ConvTranspose3d(self.num_features*4*2, self.num_features*2, kernel_size=4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm3d(self.num_features*2)
        
        self.deconv3 = nn.ConvTranspose3d(self.num_features*2*2, self.num_features, kernel_size=4, stride=2, padding=1)
        self.dec_bn3 = nn.BatchNorm3d(self.num_features)
        
        self.deconv4 = nn.ConvTranspose3d(self.num_features*2, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = self.leaky_relu(self.conv1(x))
        x_e2 = self.leaky_relu(self.enc_bn2(self.conv2(x_e1)))
        x_e3 = self.leaky_relu(self.enc_bn3(self.conv3(x_e2)))
        x_e4 = self.leaky_relu(self.enc_bn4(self.conv4(x_e3)))
        
        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x = self.relu(self.dec_bn1(self.deconv1(torch.cat([x, x_e4], dim=1))))
        x = self.relu(self.dec_bn2(self.deconv2(torch.cat([x, x_e3], dim=1))))
        x = self.relu(self.dec_bn3(self.deconv3(torch.cat([x, x_e2], dim=1))))
        x = self.deconv4(torch.cat([x, x_e1], dim=1))

        x = torch.squeeze(x, dim=1)
        
        # TODO: Log scaling
        x = torch.log(torch.abs(x)+1)
        return x
