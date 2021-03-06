import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 5, bilinear)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


class Conv_Block(nn.Module):

    def __init__(self, in_channels):
        super(Conv_Block, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv1_BN = nn.BatchNorm2d(in_channels)
        self.Conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv2_BN = nn.BatchNorm2d(in_channels)
        self.Conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = F.silu(self.Conv1_BN(self.Conv1(x))) + x
        x_2 = F.silu(self.Conv2_BN(self.Conv2(x_1))) + x_1
        x_3 = self.Conv3(x_2) + x_2 + x

        return x_3


class Conv_Block_Encoder(nn.Module):

    def __init__(self, in_channels):
        super(Conv_Block_Encoder, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv1_BN = nn.BatchNorm2d(in_channels)
        self.Conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv2_BN = nn.BatchNorm2d(in_channels)
        self.Conv3 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = F.silu(self.Conv1_BN(self.Conv1(x))) + x
        x_2 = F.silu(self.Conv2_BN(self.Conv2(x_1))) + x_1
        x_3 = self.Conv3(x_2)

        return x_3


class Conv_Block_Decoder(nn.Module):

    def __init__(self, in_channels):
        super(Conv_Block_Decoder, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv1_BN = nn.BatchNorm2d(in_channels)
        self.Conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv2_BN = nn.BatchNorm2d(in_channels)
        self.Conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = F.silu(self.Conv1_BN(self.Conv1(x))) + x
        x_2 = F.silu(self.Conv2_BN(self.Conv2(x_1))) + x_1
        x_3 = self.Conv3(x_2)

        return x_3


class ConvLSTMCell(nn.Module):
    """
    Basic CLSTM cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, b, h, w):
        return (torch.zeros(b, self.hidden_dim, h, w).cuda(),
                torch.zeros(b, self.hidden_dim, h, w).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(in_channels=cur_input_dim,
                                          hidden_channels=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ae_model(nn.Module):

    def __init__(self):
        super(ae_model, self).__init__()

        self.Encoder_Conv_Block1_1 = Conv_Block_Encoder(5)
        self.Encoder_Conv_Block1_2 = Conv_Block_Encoder(10)

        self.Encoder_Maxpool1 = nn.MaxPool2d(5)

        self.Encoder_Conv_Block2_1 = Conv_Block_Encoder(20)
        self.Encoder_Conv_Block2_2 = Conv_Block_Encoder(40)

        self.Encoder_Maxpool2 = nn.MaxPool2d(5)

        self.Middle_Block1 = Conv_Block(80)
        self.Middle_Block2 = Conv_Block(80)
        self.Middle_Block3 = Conv_Block(80)

        self.Transpose_1 = nn.ConvTranspose2d(80, 80, 5, stride=5)

        self.Decoder_Conv_Block1_1 = Conv_Block_Decoder(80)
        self.Decoder_Conv_Block1_2 = Conv_Block_Decoder(40)

        self.Transpose_2 = nn.ConvTranspose2d(20, 20, 5, stride=5)

        self.Decoder_Conv_Block2_1 = Conv_Block_Decoder(20)
        self.Decoder_Conv_Block2_2 = Conv_Block_Decoder(10)

    def forward(self, noisy):
        x = self.Encoder_Conv_Block1_1(noisy)
        x = self.Encoder_Conv_Block1_2(x)
        x = self.Encoder_Maxpool1(x)

        x = self.Encoder_Conv_Block2_1(x)
        x = self.Encoder_Conv_Block2_2(x)
        x = self.Encoder_Maxpool2(x)

        x = self.Middle_Block1(x)
        x = self.Middle_Block2(x)
        x = self.Middle_Block3(x)

        x = self.Transpose_1(x)
        x = self.Decoder_Conv_Block1_1(x)
        x = self.Decoder_Conv_Block1_2(x)

        x = self.Transpose_2(x)
        x = self.Decoder_Conv_Block2_1(x)
        x = self.Decoder_Conv_Block2_2(x)

        return x


class unet_model(nn.Module):

    def __init__(self):
        super(unet_model, self).__init__()

        self.unet = UNet(5)

    def forward(self, noisy):
        x = self.self.unet (noisy)


        return x


class Convlstm_model(nn.Module):

    def __init__(self):
        super(Convlstm_model, self).__init__()

        self.Convlstm = ConvLSTM(in_channels=5, hidden_channels=5, kernel_size=(3, 3), num_layers=2, \
                                 batch_first=True, bias=True, return_all_layers=False)

    def forward(self, noisy):
        noisy = noisy[None]
        x = self.Convlstm(noisy)[0][0]
        x = x.view(-1, 5, 1025, 200)


        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

unet_model = unet_model()
unet_model.load_state_dict(torch.load('model/unet_5channel.pth'))
unet_model.to(device)

ae_model = ae_model()
ae_model.load_state_dict(torch.load('model/ae_5channel.pth'))
ae_model.to(device)

cl_model = Convlstm_model()
cl_model.load_state_dict(torch.load('model/Convlstm_5channel.pth'))
cl_model.to(device)