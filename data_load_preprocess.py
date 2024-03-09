import torchvision.transforms as transforms
import os
import torch
from PIL import Image
import random
from torch.utils.data import DataLoader

dataroot = 'anime_faces'
batch_size = 4

# Data Pre-Processing Functions (out of the box)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def get_transform(params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
   
    crop_size = 256
    load_size = 286
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    osize = [load_size, load_size]
    transform_list.append(transforms.Resize(osize, method))

    if params is None:
        transform_list.append(transforms.RandomCrop(crop_size))
    else:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images[:len(images)]



#Preprocess Data Class

class PreprocessDataset():

    def __init__(self):

        self.dir_A = os.path.join(dataroot, 'testA')  #create paths 
        self.dir_B = os.path.join(dataroot, 'testB')  

        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from paths
        self.B_paths = sorted(make_dataset(self.dir_B))    

        self.A_size = len(self.A_paths)  # get dataset sizes
        self.B_size = len(self.B_paths)  
        
        self.transform_A = get_transform()
        self.transform_B = get_transform()

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size] 
        index_B = index % self.B_size
        
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # transform image
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)


#Data Loader 

class AnimeDataLoader():

    def __init__(self):
   
        self.dataset = PreprocessDataset()

        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle= False,
            num_workers=4)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

