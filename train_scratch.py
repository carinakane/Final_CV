from data_load_preprocess import *
from ootb import visualizer as vis
import torch.nn.init as init
from scratch_model import *
import sys

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

#Device 
gpu_ids = [0, 1, 2, 3]
device = torch.device('cuda:{}'.format(gpu_ids[0]))

# IF LOAD 

generatorA = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/20_net_G_AB.pth') 
generatorB = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/20_net_G_BA.pth')
discriminatorA = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/20_net_D_A.pth')
discriminatorB = torch.load('/home/carinakane/computer_vision/final_project/carina_CycleGAN/checkpoints/20_net_D_B.pth')

model = CycleGAN()
model.G_AB.load_state_dict(generatorA)
model.G_BA.load_state_dict(generatorB)
model.D_A.load_state_dict(discriminatorA)
model.D_B.load_state_dict(discriminatorB)

'''
# Init Model with Weights 
for param in model.parameters():
    if param.dim() > 1:
        init.normal_(param, std=0.02)
'''

model = model.to(device) 

# Optimizers Gen and Disc
optimizer_G = torch.optim.Adam(itertools.chain(model.G_AB.parameters(), model.G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(model.D_A.parameters(), model.D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

# LR Schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.9)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.9)
    
# Set Up Parallelization on Multiple GPUs
model = nn.DataParallel(model, device_ids=gpu_ids)

#State dict to device
model_cpu = model.module.cpu().to(device)

for param in model_cpu.parameters():
    param.data = param.data.to(device)
    if param.grad is not None:
        param.grad.data = param.grad.data.to(device)

model_cpu.train()
          
############################################# TRAINING ##################################
def train():
    
    total_iters = 0  
    for epoch in range(0,100):
        model.train()
        
        epoch_iter = 0                  
        visualizer.reset()    

        scheduler_G.step() 
        scheduler_D.step() 

        for i, data in enumerate(dataset):  #per epoch
            total_iters += 1
            epoch_iter += 1
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
 
            #Update Weights
            fake_A, fake_B, rec_A, rec_B = model.forward(real_A, real_B)
            optimizer_G.zero_grad() 
            model.module.backward_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)     
            optimizer_G.step() 
            optimizer_D.zero_grad()  
            model.module.backward_D_A(real_A, fake_B)     
            model.module.backward_D_B(real_B, fake_A)      
            optimizer_D.step() 

            #Get Losses
            loss_G = model.module.loss_G.item()
            loss_D = (model.module.loss_D_A + model.module.loss_D_B) * 0.5

            #Progress
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
                      .format(epoch+1, n_epochs, i+1, len(data_loader), loss_G, loss_D))
                
                print('LRs (G,D)= ', scheduler_G.get_last_lr()[0], scheduler_D.get_last_lr()[0])










####################################### EVERYTHING BELOW IS VISUALIZATION / CHECKPOINT Saving (Out of the Box)##########################################
            
            if total_iters % display_freq == 0:   
                save_result = total_iters % 1000 == 0
                model.module.compute_visuals()
                visualizer.display_current_results(model.module.get_current_visuals(real_A, fake_B, rec_A, real_B, fake_A, rec_B), epoch, save_result)


            if total_iters % print_freq == 0:   
                losses = model.module.get_current_losses() 
                visualizer.print_current_losses(epoch, epoch_iter, losses) 
                if display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), losses)       
            
            if total_iters % save_latest_freq == 0:   
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                model.module.save_networks(save_suffix)


            if epoch % save_epoch_freq == 0 and epoch_iter == 100:             
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.module.save_networks('latest')
                model.module.save_networks(epoch)

####################################################################################################################################



train()
