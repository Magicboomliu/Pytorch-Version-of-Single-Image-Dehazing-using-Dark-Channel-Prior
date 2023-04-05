from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../..")
from dehaze_torch import DarkChannelPrior
import torch


def image_numpy_to_tensor(image_np):
    image_data = torch.from_numpy(image_np) # [H,W,3]
    image_data = image_data.permute(2,0,1).unsqueeze(0)
    return image_data



if __name__=="__main__":
    
    sample_image_path = "fogg.png"
    image_data = np.array(Image.open(sample_image_path))
    image_data = np.asarray(image_data,dtype=np.float64)
    
    image_data_tensor = image_numpy_to_tensor(image_data)
    
    # image_data_tensor = torch.cat((image_data_tensor,image_data_tensor),dim=0)
 
    
    
    dark_channel_piror = DarkChannelPrior(kernel_size=15, top_candidates_ratio=0.0001,
                                          omega=0.95,radius=40,eps=1e-3,open_threshold=True,depth_est=True)
    
    dehaze_images, dc,airlight,raw_t,refined_transmission,depth= dark_channel_piror(image_data_tensor)
    
    # dehaze_images = dehaze_images[:1,:,:,:]
    # raw_t = raw_t[1:2,:,:,:]
    # refined_transmission = refined_transmission[0:1,:,:,:]
    # depth = depth[0:1,:,:,:]
    # dc = dc[0:1,:,:,:]
    # print(dehaze_images.shape)
    # print(dc.shape)
    

    plt.figure(figsize=(5,7))
    plt.subplots_adjust(left=None, bottom=None,right=None,top=None,wspace=None,hspace=None)
    plt.subplot(3,2,1)
    plt.axis("off")
    plt.title("Original_Images")
    plt.imshow(image_data_tensor.squeeze(0).permute(1,2,0).cpu().numpy()/255)
    plt.subplot(3,2,2)
    plt.axis("off")
    plt.title("Dark Channels")
    plt.imshow(dc.squeeze(0).squeeze(0).cpu().numpy(),cmap='gray')
    plt.subplot(3,2,3)
    plt.axis("off")
    plt.title("Raw Transmissions")
    plt.imshow(raw_t.squeeze(0).squeeze(0).cpu().numpy(),cmap='gray')

    plt.subplot(3,2,4)
    plt.axis("off")
    plt.title("softmatting transmissions")
    plt.imshow(refined_transmission.squeeze(0).squeeze(0).cpu().numpy(),cmap='gray')
    plt.subplot(3,2,5)
    plt.axis("off")
    plt.title("recovered de-haze images")
    plt.imshow(dehaze_images.squeeze(0).permute(1,2,0).cpu().numpy()/255)
    plt.subplot(3,2,6)
    plt.axis("off")
    plt.title("scaled depth map")
    plt.imshow(depth.squeeze(0).squeeze(0).cpu().numpy(),cmap='jet')
    plt.savefig("examples.png",bbox_inches='tight')
    plt.show()

