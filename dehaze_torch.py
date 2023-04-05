import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
from GuideFilter.guided_filter import GuidedFilter2d,FastGuidedFilter2d


# Dark Channel Piro
class DarkChannelPrior(nn.Module):
    def __init__(self,kernel_size,top_candidates_ratio,omega,
                 radius, eps,
                 open_threshold=True,
                 depth_est=False):
        super().__init__()
        
        # dark channel piror
        self.kernel_size = kernel_size
        self.pad = nn.ReflectionPad2d(padding=kernel_size//2)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=0)
        
        # airlight estimation.
        self.top_candidates_ratio = top_candidates_ratio
        self.open_threshold = open_threshold
        
        # raw transmission estimation 
        self.omega = omega
        
        # image guided filtering
        self.radius = radius
        self.eps = eps
        self.guide_filter = GuidedFilter2d(radius=self.radius,eps= self.eps)
        
        self.depth_est = depth_est
        
    def forward(self,image):
        
        # compute the dark channel piror of given image.
        b,c,h,w = image.shape
        image_pad = self.pad(image)
        local_patches = self.unfold(image_pad)
        dc,dc_index = torch.min(local_patches,dim=1,keepdim=True)
        dc = dc.view(b,1,h,w)
        dc_vis = dc
        # airlight estimation.
        top_candidates_nums = int(h*w*self.top_candidates_ratio)
        dc = dc.view(b,1,-1) # dark channels
        searchidx = torch.argsort(-dc,dim=-1)[:,:,:top_candidates_nums]
        searchidx = searchidx.repeat(1,3,1)
        image_ravel = image.view(b,3,-1)
        value = torch.gather(image_ravel,dim=2,index=searchidx)
        airlight,image_index = torch.max(value,dim =-1,keepdim=True)
        airlight = airlight.squeeze(-1)
        if self.open_threshold:
            airlight = torch.clamp(airlight,max=220)
        
        # get the raw transmission
        airlight = airlight.unsqueeze(-1).unsqueeze(-1)
        processed = image/airlight
        
        processed_pad = self.pad(processed)
        local_patches_processed = self.unfold(processed_pad)
        dc_processed, dc_index_processed = torch.min(local_patches_processed,dim=1,keepdim=True)
        dc_processed = dc_processed.view(b,1,h,w)
        
        raw_t = 1.0 - self.omega * dc_processed
        if self.open_threshold:
            raw_t = torch.clamp(raw_t,min=0.2)
            
        # raw transmission guided filtering.
        # refined_tranmission = soft_matting(image_data_tensor,raw_transmission,r=40,eps=1e-3)
        normalized_img = simple_image_normalization(image)
        refined_transmission = self.guide_filter(raw_t,normalized_img)
        
        
        # recover image: get radiance.
        image = image.float()
        tiledt = refined_transmission.repeat(1,3,1,1)
        
        dehaze_images = (image - airlight)*1.0/tiledt + airlight
        
        # recover scaled depth or not
        if self.depth_est:
            depth = recover_depth(refined_transmission)
            return dehaze_images, dc_vis,airlight,raw_t,refined_transmission,depth
        
        return dehaze_images, dc_vis,airlight,raw_t,refined_transmission



def simple_image_normalization(tensor):
    b,c,h,w = tensor.shape
    tensor_ravel = tensor.view(b,3,-1)
    image_min,_ = torch.min(tensor_ravel,dim=-1,keepdim=True)
    image_max,_ = torch.max(tensor_ravel,dim=-1,keepdim=True)
    image_min = image_min.unsqueeze(-1)
    image_max = image_max.unsqueeze(-1)
    
    normalized_image = (tensor - image_min) /(image_max-image_min)
    return normalized_image

def recover_depth(transmission,beta=0.001):
    negative_depth = torch.log(transmission)
    return (-negative_depth)/beta


