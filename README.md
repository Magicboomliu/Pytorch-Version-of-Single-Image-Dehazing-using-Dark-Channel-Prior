# Pytorch-Version-of-Single-Image-Dehazing-using-Dark-Channel-Prior
An Pytorch  implementent of CVPR2009 "single image dehazing using dark channel prior." Support Batch Level Operation.


## Dependencies
* pytorch
* numpy
* PIL


## Reference:   
[1] https://github.com/He-Zhang/image_dehaze  
[2] https://github.com/anjali-chadha/dark-channel-prior-dehazing  
[3] Single Image Haze Removal Using Dark Channel Prior, Kaiming He, Jian Sun, and Xiaoou Tang", in CVPR 2009  
[4] Guided Image Filtering, Kaiming He, Jian Sun, and Xiaoou Tang", in ECCV 2010.

## Usage
```python
from dehaze_torch import DarkChannelPrior

dark_channel_piror = DarkChannelPrior(kernel_size=15,top_candidates_ratio=0.0001,omega=0.95,radius=40,eps=1e-3,open_threshold=True,depth_est=True)
    
dehaze_images, dc,airlight,raw_t,refined_transmission,depth = dark_channel_piror(image_data_tensor)

```
## Visualization of Dehazing Images and Intermediate Results 

![](examples.png)

## Citation 

If you find this implementation is helpful, please consider to cite: 
```
@misc{zihualiu2023DCP,
  title={Pytorch-Version-of-Single-Image-Dehazing-using-Dark-Channel-Prior},
  author={Zihua Liu},
  howpublished={\url{https://github.com/Magicboomliu/Pytorch-Version-of-Single-Image-Dehazing-using-Dark-Channel-Prior}},
  year={2023}
}
```