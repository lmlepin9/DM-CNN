import numpy as np
from larcv import larcv
import random

import ROOT
from ROOT import TChain

import torch
from torch.utils.data import Dataset

def image_modify(img):
    img_mod = np.where(img<10,    0,img)
    img_mod = np.where(img>500, 500,img_mod)
    return img_mod


#Plane 2 is the only one present in the cropped dataset, therefore we use plane = 0 
class MPID_Dataset(Dataset):
    def __init__(self, input_file, image_tree, device, plane=0,augment=False, verbose=False):
        self.plane=plane
        self.augment=augment
        self.verbose=verbose
         

        self.particle_image_chain = TChain(image_tree)
        self.particle_image_chain.AddFile(input_file)
        if (device):
            self.device=device
        else:
            self.device="cpu"
        
    def __getitem__(self, ENTRY):
        # Reading Image

        #print ("open ENTRY @ {}".format(ENTRY))

        self.particle_image_chain.GetEntry(ENTRY)
        self.this_image_cpp_object = self.particle_image_chain.image2d_image2d_binary_branch
        
        self.event_info = torch.zeros([3])
        self.event_info[0] = self.this_image_cpp_object.run()
        self.event_info[1] = self.this_image_cpp_object.subrun()
        self.event_info[2] = self.this_image_cpp_object.event()
        self.this_image=larcv.as_ndarray(self.this_image_cpp_object.as_vector()[self.plane])
        # Image Thresholding
        self.this_image=image_modify(self.this_image)

        #print (self.this_image)
        #print ("sum, ")
        #if (np.sum(self.this_image) < 9000):
        #    ENTRY+
        
        if self.augment:
            if random.randint(0, 1):
            #if True:
                #if (self.verbose): print ("flipped")
                self.this_image = np.fliplr(self.this_image)
            if random.randint(0, 1):
            #if True:
                #if (self.verbose): print ("transposed")
                self.this_image = self.this_image.transpose(1,0)
        self.this_image = torch.from_numpy(self.this_image.copy())
        # self.this_image=torch.tensor(self.this_image, device=self.device).float()

        self.this_image=self.this_image.clone().detach()
        
        # Creating labels 
        self.event_label = torch.zeros([2])
        # Signal events are labeled with run = 100 
        if(self.this_image_cpp_object.run() == 100):
            self.event_label[0] = 1
        
        else:
            self.event_label[1] = 1
                
                              
        return (self.this_image, self.event_label, self.event_info)
        #return (self.this_image, self.event_label)

    def __len__(self):
        return self.particle_image_chain.GetEntries()

