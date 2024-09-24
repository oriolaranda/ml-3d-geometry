import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.wnl1 = nn.utils.weight_norm(nn.Linear(latent_size+3,512))
        self.wnl2 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnl3 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnl4 = nn.utils.weight_norm(nn.Linear(512,512-latent_size-3))
        self.wnl5 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnl6 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnl7 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnl8 = nn.utils.weight_norm(nn.Linear(512,512))
        
        self.fcn = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.dropout(self.relu(self.wnl1(x_in)))
        x = self.dropout(self.relu(self.wnl2(x)))
        x = self.dropout(self.relu(self.wnl3(x)))
        x = self.dropout(self.relu(self.wnl4(x)))
        
        x = self.dropout(self.relu(self.wnl5(torch.cat((x, x_in),dim=1))))
        x = self.dropout(self.relu(self.wnl6(x)))
        x = self.dropout(self.relu(self.wnl7(x)))
        x = self.dropout(self.relu(self.wnl8(x)))
        
        x = self.fcn(x)
        
        return x
