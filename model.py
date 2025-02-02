import torch
import torch.nn as nn

# Constants
channel_rate = 64
image_shape = (256, 256, 3)
patch_shape = (channel_rate, channel_rate, 3)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation_factor=None):
        super(DenseBlock, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels, 4 * channel_rate, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(4 * channel_rate)

        if dilation_factor is not None:
            self.conv2 = nn.Conv2d(4 * channel_rate, channel_rate, 3,
                                   padding=dilation_factor, dilation=dilation_factor)
        else:
            self.conv2 = nn.Conv2d(4 * channel_rate, channel_rate, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(channel_rate)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.leaky_relu(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # The Head
        self.head = nn.Conv2d(3, 4 * channel_rate, 3, padding=1)

        # The Dense Field
        # Calculate input channels for each dense block
        self.dense1 = DenseBlock(4 * channel_rate)  # Input from head
        self.dense2 = DenseBlock(5 * channel_rate, dilation_factor=1)  # Previous + channel_rate
        self.dense3 = DenseBlock(6 * channel_rate)
        self.dense4 = DenseBlock(7 * channel_rate, dilation_factor=2)
        self.dense5 = DenseBlock(8 * channel_rate)
        self.dense6 = DenseBlock(9 * channel_rate, dilation_factor=3)
        self.dense7 = DenseBlock(10 * channel_rate)
        self.dense8 = DenseBlock(11 * channel_rate, dilation_factor=2)
        self.dense9 = DenseBlock(12 * channel_rate)
        self.dense10 = DenseBlock(13 * channel_rate, dilation_factor=1)

        # The Tail
        self.tail_leaky = nn.LeakyReLU(0.2)
        self.tail_conv1 = nn.Conv2d(14 * channel_rate, 4 * channel_rate, 1)
        self.tail_bn = nn.BatchNorm2d(4 * channel_rate)

        # Final layers
        self.final_conv = nn.Conv2d(8 * channel_rate, channel_rate, 3, padding=1)  # 4*channel_rate + 4*channel_rate
        self.final_leaky = nn.LeakyReLU(0.2)
        self.output_conv = nn.Conv2d(channel_rate, 3, 3, padding=1)

    def forward(self, x):
        # Head
        h = self.head(x)

        # Dense field with concatenations
        x1 = self.dense1(h)
        c1 = torch.cat([h, x1], dim=1)

        x2 = self.dense2(c1)
        c2 = torch.cat([c1, x2], dim=1)

        x3 = self.dense3(c2)
        c3 = torch.cat([c2, x3], dim=1)

        x4 = self.dense4(c3)
        c4 = torch.cat([c3, x4], dim=1)

        x5 = self.dense5(c4)
        c5 = torch.cat([c4, x5], dim=1)

        x6 = self.dense6(c5)
        c6 = torch.cat([c5, x6], dim=1)

        x7 = self.dense7(c6)
        c7 = torch.cat([c6, x7], dim=1)

        x8 = self.dense8(c7)
        c8 = torch.cat([c7, x8], dim=1)

        x9 = self.dense9(c8)
        c9 = torch.cat([c8, x9], dim=1)

        x10 = self.dense10(c9)
        c10 = torch.cat([c9, x10], dim=1)

        # Tail
        x = self.tail_leaky(c10)
        x = self.tail_conv1(x)
        x = self.tail_bn(x)

        # Global Skip Connection
        x = torch.cat([h, x], dim=1)
        x = self.final_conv(x)
        x = self.final_leaky(x)

        # Output
        x = self.output_conv(x)
        return torch.tanh(x)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, channel_rate, normalize=False),
            *discriminator_block(channel_rate, 2 * channel_rate),
            *discriminator_block(2 * channel_rate, 4 * channel_rate),
            *discriminator_block(4 * channel_rate, 4 * channel_rate),
            nn.Flatten(),
            nn.Linear(4 * channel_rate * (patch_shape[0] // 16) * (patch_shape[1] // 16), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.patch_discriminator = PatchDiscriminator()

    def extract_patches(self, x):
        B, C, H, W = x.shape
        patches = []
        for i in range(0, H - patch_shape[0] + 1, patch_shape[0]):
            for j in range(0, W - patch_shape[1] + 1, patch_shape[1]):
                patch = x[:, :, i:i + patch_shape[0], j:j + patch_shape[1]]
                patches.append(patch)
        return patches

    def forward(self, x):
        # Split image into patches
        patches = self.extract_patches(x)

        # Process each patch
        outputs = []
        for patch in patches:
            outputs.append(self.patch_discriminator(patch))

        # Average the outputs
        return torch.mean(torch.stack(outputs, dim=1), dim=1)


def print_network(net, name):
    """Print the network architecture."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(f'Network [{name}]')
    print(net)
    print(f'Total number of parameters: {num_params:,}')


if __name__ == '__main__':
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Print model architectures
    print_network(generator, 'Generator')
    print('\n' + '=' * 50 + '\n')
    print_network(discriminator, 'Discriminator')

    # Test forward pass
    test_input = torch.randn(1, 3, 256, 256).to(device)
    gen_output = generator(test_input)
    disc_output = discriminator(test_input)

    print(f'\nGenerator output shape: {gen_output.shape}')
    print(f'Discriminator output shape: {disc_output.shape}')
