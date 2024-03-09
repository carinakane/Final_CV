from data_load_preprocess import *
from ootb import visualizer as vis
import torch.nn.init as init
from scratch_model import *
import sys
from collections import OrderedDict

################# (Out of the Box)##########################################################################################
#Visdom and HTML params and Network Saving 
display_freq = 200 
display_ncols = 4 
display_id = 1
display_server = "http://localhost"
display_env = 'main'
display_port = 8097
update_html_freq = 1000
print_freq = 100 
save_latest_freq = 5000 
save_epoch_freq = 5 
epoch_count = 1
phase = 'train'
visualizer = vis.Visualizer()  
########################################################################################################################

# Create Dataset 
data_loader = AnimeDataLoader()
dataset = data_loader.load_data()
print('Dataset Created')
print('Length of Dataset = ', len(dataset))

# Device 
gpu_ids = [0, 1, 2, 3]
device = torch.device('cuda:{}'.format(gpu_ids[0]))

# IF LOAD 
generatorA = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/latest_net_G_AB.pth') 
generatorB = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/latest_net_G_BA.pth')
discriminatorA = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/latest_net_D_A.pth')
discriminatorB = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/latest_net_D_B.pth')

model = CycleGAN()
model.G_AB.load_state_dict(generatorA)
model.G_BA.load_state_dict(generatorB)
model.D_A.load_state_dict(discriminatorA)
model.D_B.load_state_dict(discriminatorB)

# Move model to device
model = model.to(device) 

# Optimizers Gen and Disc
optimizer_G = torch.optim.Adam(itertools.chain(model.G_AB.parameters(), model.G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(model.D_A.parameters(), model.D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

# LR Schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.9)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.9)
    
# Set Up Parallelization on Multiple GPUs
model = nn.DataParallel(model, device_ids=gpu_ids)

def test():
    model.eval()
    visualizer.reset()    

    for i, data in enumerate(dataset):  # per epoch
        real_A = data['A'].to(device)
        real_B = data['B'].to(device)

        image_dict = OrderedDict()
        for idx, img in enumerate(real_A[0:49]):
            with torch.no_grad():
                fake_B = model.module.G_AB(img.unsqueeze(0))  # Forward pass through generator
            image_dict[f"image{idx}"] = fake_B.squeeze(0).cpu()  # Add generated image to dictionary

        visualizer.display_current_results7(image_dict, 1, False)

test()
