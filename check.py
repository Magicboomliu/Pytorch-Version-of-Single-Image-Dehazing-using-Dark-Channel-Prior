import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    refine_transmission_from_numpy = np.load("refined_transmission_numpy.npy")
    refined_transmission_from_torch = np.load("refined_tranmission_torch.npy")
    
    print(refine_transmission_from_numpy.mean())
    print(refined_transmission_from_torch.mean())
    error = refine_transmission_from_numpy - refined_transmission_from_torch
    
    print(error.mean())
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(refine_transmission_from_numpy)
    plt.subplot(1,3,2)
    plt.axis("off")
    plt.imshow(refined_transmission_from_torch)
    plt.subplot(1,3,3)
    plt.axis("off")
    plt.imshow(error)
    
    
    plt.show()
    
    
    
    pass