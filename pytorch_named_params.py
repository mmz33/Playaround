import torch.nn as nn


class AbstractModel(nn.Module):

    def __init__(self):
        super().__init__()

    def print_named_params(self):
        for n, p in self.named_parameters():
            print(n, p)


class Conv2dBlock(AbstractModel):

    def __init__(self):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(
                in_channels=1, out_channels=10, kernel_size=(3,3), stride=(1,1))


class ConvModel(AbstractModel):

    def __init__(self):
        super().__init__()
        conv2d_block = Conv2dBlock()


conv_block = Conv2dBlock()
conv_block.print_named_params()  # conv2d.weights and conv2d.bias

conv_model = ConvModel()
conv_model.print_named_params()  # nothing
