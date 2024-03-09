import torch
import torch.nn as nn
from model_chunks import *
import itertools
from collections import OrderedDict
import torch.optim.lr_scheduler as lr_scheduler

from data_load_preprocess import *
from ootb import visualizer 
import sys
import os



# train params
n_epochs_decay = 100
n_epochs = 100
epoch_count = 1


class CycleGAN(nn.Module):
    def __init__(self):
        
        super(CycleGAN, self).__init__()

        ##################################### Out of the Box for Visualizing #################################################
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_AB', 'G_BA', 'D_A', 'D_B']
        ###########################################################################################################################
      
        # Initialize Device
        self.gpu_ids = 0,1,2,3
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

        # Initialize Generators: G_AB and G_BA
        self.G_AB = ResNet()
        self.G_BA = ResNet()

        # Initialize two discriminators: D_A and D_B
        self.D_A = PatchGAN()
        self.D_B = PatchGAN()
          

    def forward(self, real_A, real_B): 
        self.real_A = real_A
        self.real_B = real_B
        self.fake_B = self.G_AB(real_A) 
        self.rec_A = self.G_BA(self.fake_B)   
        self.fake_A = self.G_BA(real_B)  
        self.rec_B = self.G_AB(self.fake_A) 
        return self.fake_A, self.fake_B, self.rec_A, self.rec_B


    def backward_D_B(self, real_B, fake_A):
        """Loss for discriminator D_B"""

        pred_real_B = self.D_B(real_B)
        loss_D_real_B = GANLoss(pred_real_B, True)
        
        pred_fake_B = self.D_B(fake_A.detach())
        loss_D_fake_B = GANLoss(pred_fake_B, False)

        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        self.loss_D_B = loss_D_B

    def backward_D_A(self, real_A, fake_B):
        """Loss for discriminator D_A"""

        pred_real_A = self.D_A(real_A)
        loss_D_real_A = GANLoss(pred_real_A, True)
        
        pred_fake_A = self.D_A(fake_B.detach())
        loss_D_fake_A = GANLoss(pred_fake_A, False)

        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        self.loss_D_A = loss_D_A

    
    def backward_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        """Loss for both generators"""

        self.loss_G_A = GANLoss(self.D_A(fake_B), True)
        self.loss_G_B = GANLoss(self.D_B(fake_A), True)

        # || G_B(G_A(A)) - A||
        self.loss_cycle_A = torch.nn.L1Loss()(rec_A, real_A) * 10 #extra weight added to cycle loss
        # || G_A(G_B(B)) - B||
        self.loss_cycle_B = torch.nn.L1Loss()(rec_B, real_B) * 10 #extra weight added to cycle loss
        #Combine all generator losses
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B 
        self.loss_G.backward()


















    ########################################## OUT OF THE BOX VISUALIZATION IMPLEMENTATION FUNCTIONS #############################################
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    
    def get_current_visuals(self, real_A, real_B,fake_A, fake_B, rec_A, rec_B):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        visual_ret['real_A'] = real_A
        visual_ret['real_B'] = real_B
        visual_ret['fake_A'] = fake_A
        visual_ret['fake_B'] = fake_B
        visual_ret['rec_A'] = rec_A
        visual_ret['rec_B'] = rec_B
        return visual_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join('checkpoints', save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

         
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
        

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass






