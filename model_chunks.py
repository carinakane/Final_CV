import torch.nn as nn
import torch

#Creates Classes of:
    #ResNet Block
    #Resnet (Generator)
    #PatchGAN (Discriminator)
    #GAN Loss Function

#Data Format = torch.Size([3, 256, 256])


class ResBlock(nn.Module):
    """
    Resnet block with passable arguments for in_channels, out_channels, kernel_size, padding, stride

    No downsampling implemented.
    """
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                   padding_mode='reflect', bias=False)
        
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,padding=1,
                                   padding_mode='reflect', bias=False)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x + identity)

        return x


#9 Block ResNet

class ResNet(nn.Module):
    """
    Resnet 9 residual blocks 

    """
    def __init__(self):
        super(ResNet, self).__init__()

        # Initial Conv, BN,
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding='same', bias=False, padding_mode= 'reflect') # in_channels = 3 , out_channels = 64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        #Downsample 1
        self.convds1 = nn.Conv2d(64, 128, kernel_size=3, stride = 2, padding=1, bias=False)
        self.bnds1 = nn.BatchNorm2d(128)
        self.reluds1 = nn.ReLU()

        #Downsample 2
        self.convds2 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding=1, bias=False)
        self.bnds2 = nn.BatchNorm2d(256) 
        self.reluds2 = nn.ReLU()

        #Res Block X9
        self.res = ResBlock(256)

        #Upsample 1
        self.convus1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bnus1 = nn.BatchNorm2d(128)
        self.reluus1 = nn.ReLU()

        #Upsample 2
        self.convus2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bnus2 = nn.BatchNorm2d(64)
        self.reluus2 = nn.ReLU()

        # FinalConv, TanH
        self.convf = nn.Conv2d(64, 3, kernel_size=3, padding='same', bias=False, padding_mode= 'reflect')
        self.bnf = nn.BatchNorm2d(3)
        self.tanhf = nn.Tanh()


    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # Initial Conv, BN, ReLU

        x_ds1 = self.reluds1(self.bnds1(self.convds1(x)))  # Downsample 1
        x_ds2 = self.reluds2(self.bnds2(self.convds2(x_ds1)))  # Downsample 2

        x_res1 = self.res(x_ds2)  # Res Blocks
        x_res2 = self.res(x_res1)
        x_res3 = self.res(x_res2)
        x_res4 = self.res(x_res3)
        x_res5 = self.res(x_res4)
        x_res6 = self.res(x_res5)
        x_res7 = self.res(x_res6)
        x_res8 = self.res(x_res7)
        x_res9 = self.res(x_res8)

        x_us1 = self.reluus1(self.bnus1(self.convus1(x_res9)))  # Upsample 1
        x_us2 = self.reluus2(self.bnus2(self.convus2(x_us1)))  # Upsample 2

        x_final = self.tanhf(self.bnf(self.convf(x_us2)))  # Final Conv, BN, TanH

        return x_final



class PatchGAN(nn.Module):
    '''
    three conv layers
    assumes input shape of: (3, 256, 256)

    '''
    def __init__(self):
        super(PatchGAN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)



def GANLoss(prediction, target_is_real):
    loss_function = nn.MSELoss()
    target_tensor = torch.full_like(prediction, 1.0) if target_is_real else torch.full_like(prediction, 0.0)
    loss = loss_function(prediction, target_tensor)
    return loss