import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms.functional as F
import torchvision as t
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
from torch.nn import init
import pdb
def get_img_norm_cfg(dataset_name, dataset_dir):
    if  dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'IRDST-real':   
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        # with open(dataset_dir + '/' + dataset_name +'/img_idx/train'  + '.txt', 'r') as f:
        #     train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name +'/img_idx/test'  + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = test_list#train_list #+ test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        print(len(img_list))
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    print('img_pth',img_pth)
                    pdb.set_trace()
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
                    
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg
def Denormalization(img, img_norm_cfg):
    return img*img_norm_cfg['std']+img_norm_cfg['mean']
def Normalized(img, img_norm_cfg):
    return (img-img_norm_cfg['mean'])/img_norm_cfg['std']
class BasicTestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None,fold=0):
        super(BasicTestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.fold = fold
        with open(self.dataset_dir+'/img_idx/test'  + '.txt', 'r') as f:
        # with open(f'/home/dww/OD/BasicIRSTD/val_fold{fold}.txt', 'w') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            #img_path=(self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')
            #pdb.set_trace()
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/')).convert('L')
            ###resize下
            # img = img.resize((512, 512))
            # mask = mask.resize((512, 512))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))
            #img_path=(self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask,[h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img   

class BasicInferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(BasicInferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test' '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 