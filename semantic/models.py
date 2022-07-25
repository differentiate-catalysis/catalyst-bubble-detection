import torch
import torch.nn as nn
from unet import UNet2D, UNet3D


class FCDenseNet3D(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(4,4,4), up_blocks=(4,4,4), first_conv_out_channels=48,
                bottleneck_layers=4, growth_rate=12, classes=2):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        channels_count = 0
        skip_connection_channel_counts = []

        # first convolution, prior to the dense blocks
        self.first_conv = nn.Conv3d(in_channels=in_channels, out_channels=first_conv_out_channels,
                    kernel_size=(3, 3, 3), padding=1, bias=True)
        channels_count = first_conv_out_channels

        # encoder side - downsampling
        self.dense_blocks_down = nn.ModuleList([])
        self.trans_down_blocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.dense_blocks_down.append(DenseBlock(channels_count, growth_rate, down_blocks[i]))
            channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,channels_count)
            self.trans_down_blocks.append(TransitionDown(channels_count))

        # bottleneck - no transition blocks
        self.bottleneck = DenseBlock(channels_count, growth_rate, bottleneck_layers, upsample=True)
        prev_block_channels = growth_rate*bottleneck_layers
        channels_count += prev_block_channels


        # decoder side - upsampling
        self.trans_up_blocks = nn.ModuleList([])
        self.dense_blocks_up = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.trans_up_blocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.dense_blocks_up.append(DenseBlock(channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            channels_count += prev_block_channels

        # last dense block doesn't upsample
        self.trans_up_blocks.append(TransitionUp(prev_block_channels, prev_block_channels))
        channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.dense_blocks_up.append(DenseBlock(channels_count, growth_rate, up_blocks[-1], upsample=False))
        channels_count += growth_rate*up_blocks[-1]

        # last convolution
        self.last_conv = nn.Conv3d(in_channels=channels_count, out_channels=classes,
                    kernel_size=(1, 1, 1), padding=0, bias=True)
        # softmax the output
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # apply the first convolution
        out = self.first_conv(x)

        # encoder side - downsample
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.dense_blocks_down[i](out)
            skip_connections.append(out)
            out = self.trans_down_blocks[i](out)

        # bottleneck
        out = self.bottleneck(out)

        # decoder side - upsample
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.trans_up_blocks[i](out, skip)
            out = self.dense_blocks_up[i](out)

        #last convolution
        out = self.last_conv(out)

        #softmax
        out = self.softmax(out)
        return out


def DenseLayer(in_channels, growth_rate):
    dense_layer = nn.Sequential(
        nn.BatchNorm3d(in_channels),
        nn.ReLU(True),
        nn.Conv3d(in_channels, growth_rate, kernel_size=(3, 3, 3),
                padding=1, bias=True),
        nn.Dropout3d(0.2)
    )
    return dense_layer

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x

def TransitionDown(in_channels):
    transition_down = nn.Sequential(
        nn.BatchNorm3d(num_features=in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1 ,1), padding=0, bias=True),
        nn.Dropout3d(0.2),
        nn.MaxPool3d(2)
    )
    return transition_down


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3, 3), stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        _, _, h, w, d = out.size()
        _, _, h_max, w_max, d_max = skip.size()
        h_low = (h - h_max) // 2
        w_low = (w - w_max) // 2
        d_low = (d - d_max) // 2
        out = out[:, :, h_low:(h_low + h_max), w_low:(w_low + w_max), d_low:(d_low + d_max)]
        out = torch.cat([out, skip], 1)
        return out

class CustomUNet2D(UNet2D):

    def forward(self, x):
        x = torch.squeeze(x, 4)
        if len(x.shape) != 4:
            raise ValueError("2D networks can only be used on datasets with slices = 1")
        x = super().forward(x)
        x = torch.unsqueeze(x, 4)
        return x

def get_model(model: str, encoding_blocks: int) -> nn.Module:
    if model == 'unet':
        return UNet3D(
            normalization='batch',
            in_channels=1,
            preactivation=True,
            residual=True,
            out_classes=2,
            num_encoding_blocks=encoding_blocks,
            upsampling_type='trilinear',
        )
    elif model == 'fcdensenet':
        return FCDenseNet3D(
            in_channels=1,
            down_blocks=[4] * encoding_blocks,
            up_blocks=[4] * encoding_blocks,
            first_conv_out_channels=48,
            bottleneck_layers=4,
            growth_rate=12,
            classes=2
        )
    elif model == 'unet2d':
        return CustomUNet2D(
            normalization='batch',
            in_channels=1,
            preactivation=True,
            residual=True,
            out_classes=2,
            num_encoding_blocks=encoding_blocks,
        )
    raise ValueError('Invalid model type supplied')

def run(model, shape, device):
    x = torch.rand(*shape, device=device)
    with torch.no_grad():
        y = model(x)
    return y

if __name__ == '__main__':
    device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
    model = FCDenseNet3D(
        in_channels=3, 
        down_blocks=(4, 4, 4),
        up_blocks=(4, 4, 4), 
        first_conv_out_channels=48, 
        bottleneck_layers=4,
        growth_rate=12, 
        classes=2
    ).to(device).eval()
    shape = (1, 3, 250, 250, 8)
    out_shape = (shape[0], 2, *shape[2:])
    y = run(model, shape, device)
    print(tuple(y.shape) == out_shape)

model_listing = ['unet', 'fcdensenet', 'unet2d']