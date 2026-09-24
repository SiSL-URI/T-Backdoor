import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate, neuron
from copy import deepcopy
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet50
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11


def get_model(dataname='gesture', T=16, init_tau=0.02, use_plif=False, use_max_pool=False, detach_reset=False):
    '''
    For a given dataset, return the model according to https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

    Parameters:

        dataname (str): name of the dataset.
        T (int): number of time steps.
        init_tau (float): initial tau of the neuron.
        use_plif (bool): whether to use PLIF.
        use_max_pool (bool): whether to use max pooling.
        alpha_learnable (bool): whether to learn the alpha.
        detach_reset (bool): whether to detach the reset.

    Returns:

        model (NeuromorphicNet): the model.
    '''

    if dataname == 'mnist':
        model = NMNISTNet(spiking_neuron=neuron.IFNode,
                          surrogate_function=surrogate.ATan(), detach_reset=True)
    elif dataname == 'gesture':
        model = DVSGestureNet(spiking_neuron=neuron.LIFNode,
                              surrogate_function=surrogate.ATan(), detach_reset=True)
    elif dataname == 'caltech':
        model = CaltechNet(spiking_neuron=neuron.LIFNode,
                           surrogate_function=surrogate.ATan(), detach_reset=True)
    elif dataname == 'cifar10':
        model = CIFAR10DVSNet(spiking_neuron=neuron.LIFNode,
                              surrogate_function=surrogate.ATan(), detach_reset=True)
    elif dataname == 'imagenet':
        model = spiking_resnet50(spiking_neuron=neuron.IFNode,
                                 surrogate_function=surrogate.ATan(), detach_reset=True)
    else:
        raise ValueError('Dataset {} is not supported'.format(dataname))

    return model


def get_model_adaptive(dataname='gesture', T=16, init_tau=0.02, use_plif=False, use_max_pool=False, detach_reset=False):
    if dataname == 'mnist':
        model = NMNISTNet(spiking_neuron=neuron.IFNode,
                          surrogate_function=surrogate.ATan(), detach_reset=True)
    elif dataname == 'gesture':
        model = DVSGestureNet_Adaptive(spiking_neuron=neuron.LIFNode,
                                       surrogate_function=surrogate.ATan(), detach_reset=True)
    elif dataname == 'cifar10':
        model = CIFAR10DVSNet(spiking_neuron=neuron.LIFNode,
                              surrogate_function=surrogate.ATan(), detach_reset=True)
    elif dataname == 'imagenet':
        model = spiking_resnet50(spiking_neuron=neuron.IFNode,
                                 surrogate_function=surrogate.ATan(), detach_reset=True)
    else:
        raise ValueError('Dataset {} is not supported'.format(dataname))

    return model


class Autoencoder(nn.Module):

    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            layer.Conv2d(2, channels,  kernel_size=4,
                         padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(channels, channels * 2,
                         kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 2),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(channels * 2, channels * 4,
                         kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 4),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(channels * 4, channels * 8,
                         kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 8),
            spiking_neuron(**deepcopy(kwargs)),
        )
        self.decoder = nn.Sequential(
            layer.ConvTranspose2d(
                channels * 8, channels * 4,  kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 4),
            spiking_neuron(**deepcopy(kwargs)),
            layer.ConvTranspose2d(
                channels * 4, channels * 2,  kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 2),
            spiking_neuron(**deepcopy(kwargs)),
            layer.ConvTranspose2d(channels * 2, channels,
                                  kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.ConvTranspose2d(
                channels, 2,  kernel_size=4, padding=1, stride=2, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class AutoencoderMNIST(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            layer.Conv2d(2, channels,  kernel_size=4,
                         padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(channels, channels * 2,
                         kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 2),
            spiking_neuron(**deepcopy(kwargs)),
        )
        self.decoder = nn.Sequential(
            layer.ConvTranspose2d(channels * 2, channels,
                                  kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.ConvTranspose2d(
                channels, 2,  kernel_size=4, padding=0, stride=2, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class AutoencoderCaltech(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            layer.Conv2d(2, channels,  kernel_size=4,
                         padding=1, stride=3, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(channels, channels * 2,
                         kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 2),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(channels * 2, channels * 4,
                         kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 4),
            spiking_neuron(**deepcopy(kwargs)),
        )
        self.decoder = nn.Sequential(
            layer.ConvTranspose2d(
                channels * 4, channels * 2,  kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels * 2),
            spiking_neuron(**deepcopy(kwargs)),
            layer.ConvTranspose2d(channels * 2, channels,
                                  kernel_size=4, padding=1, stride=2, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.ConvTranspose2d(
                channels, 2,  kernel_size=4, padding=1, stride=3, bias=False, output_padding=1),
            spiking_neuron(**deepcopy(kwargs)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


# =============================================================================
# CaltechNet — SEW ResNet-18 backbone
#   • Input : 2-channel neuromorphic events (N-Caltech101, 128×128)
#   • Output: 101 classes
#   • SEW-ResNet shortcuts operate on spike tensors directly, avoiding
#     membrane potential accumulation across residual blocks — better suited
#     for neuromorphic data than vanilla Spiking ResNet.
#   • `conv1` replaced with a 2-ch, stride-1, 3×3 stem to preserve spatial
#     resolution on the 128×128 DVS frames.
#   • `cnf` defaults to 'ADD'; can be set to 'AND' or 'IAND'.
# =============================================================================
class CaltechNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, cnf: str = 'ADD', **kwargs):
        super().__init__()

        self.model = sew_resnet18(
            spiking_neuron=spiking_neuron,
            num_classes=101,
            cnf=cnf,
            **kwargs,
        )

        self.model.conv1 = layer.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


# =============================================================================
# CIFAR10DVSNet — SEW ResNet-18 backbone
#   • Input : 2-channel neuromorphic events (CIFAR10-DVS, 128×128)
#   • Output: 10 classes
#   • SEW-ResNet (Spike-Element-Wise ResNet) uses element-wise spike addition
#     in residual shortcuts, which is more hardware-friendly and avoids the
#     membrane potential accumulation issue of vanilla Spiking ResNet.
#   • `conv1` is replaced with a 2-ch, stride-1, 3×3 variant so that the
#     smaller DVS spatial resolution is preserved through the stem.
#   • The default 7×7 / stride-2 stem is aggressive for 128-px inputs; the
#     smaller kernel keeps more spatial detail.
#   • `cnf` (connect function) defaults to 'ADD' (spike addition); can also
#     be set to 'AND' or 'IAND' depending on the experiment.
# =============================================================================
# class CIFAR10DVSNet(nn.Module):
#     def __init__(self, channels=128, spiking_neuron: callable = None, cnf: str = 'ADD', **kwargs):
#         super().__init__()

#         # Build SEW ResNet-18 with 10 output classes
#         self.model = sew_resnet18(
#             spiking_neuron=spiking_neuron,
#             num_classes=10,
#             cnf=cnf,
#             **kwargs,
#         )

#         # SEW ResNet-18 stem: conv1 is Conv2d(3, 64, kernel_size=7, stride=2, padding=3).
#         # Replace with Conv2d(2, 64, kernel_size=3, stride=1, padding=1) so that:
#         #   • 2-channel DVS polarity input is accepted.
#         #   • Spatial downsampling is gentler on the 128×128 DVS frames.
#         self.model.conv1 = layer.Conv2d(
#             in_channels=2,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False,
#         )

#     def forward(self, x: torch.Tensor):
#         return self.model(x)
    
class CIFAR10DVSNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(4):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels,
                        kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 100),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels,
                        kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(*args, **kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(*args, **kwargs),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class DVSGestureNet_Adaptive(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels,
                        kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(*args, **kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(*args, **kwargs),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        latent = self.conv_fc(x)
        return latent, self.fc(latent)


class MNISTNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, channels, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 7 * 7, 2048),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(2048, 100),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer()
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class NMNISTNet(MNISTNet):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__(channels, spiking_neuron, **kwargs)
        self.conv_fc[0] = layer.Conv2d(
            2, channels, kernel_size=3, padding=1, bias=False)
        self.conv_fc[-6] = layer.Linear(channels * 8 * 8, 2048)
