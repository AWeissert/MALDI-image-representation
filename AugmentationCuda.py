###### AugmentationCuda
### Author: Annalena Weissert 
## Last Edited: 02 July 2026


import torch
from torchvision.transforms import v2   # For image augmentations
from torchvision import models          # Assess Resnet model (for preprocessing)
import numpy as np

from perlin_numpy import generate_fractal_noise_2d # https://github.com/pvigier/perlin-numpy  (Last Accessed: 08 July 2026)
import skimage.morphology               # To generate ellipse
import cv2                              # For Guillotine Partition


##################################### Global Variables #####################################
input_size = 232 

model = models.resnet50(weights='DEFAULT').cuda() # Initialize model
weights = models.ResNet50_Weights.IMAGENET1K_V2   # Load weights
preprocess = weights.transforms()                 # Load ResNet's preprocessing pipeline

    
###################################### MALDI Augmentations ######################################
## RandomEllipse: Add an Ellipse of random size between 5-25% of image size to zero-pixels at a random position.
# Dependecy: package skimage.morphology
# Input: img_res - input image, c - constant value that zero-values within the ellipse are set to
def RandomEllipse(img_res, c):
    s = input_size
    min_radius = int(s*0.05) 
    max_radius = int(s*0.2) 

    r1, r2 = np.random.choice(np.arange(min_radius, max_radius+1), 2)
    x = np.random.choice(np.arange(s-r2*2-1))
    y = np.random.choice(np.arange(s-r1*2-1))
    shape_template = shape_template = skimage.morphology.ellipse(r1, r2) 
    shape_template = torch.from_numpy(shape_template).to(dtype=torch.float32).cuda() 
    shape_template = v2.RandomRotation(180, expand=True)(shape_template.reshape(1,2*r2+1,2*r1+1)) 
    shape_template = v2.Resize((2*r2+1,2*r1+1))(shape_template)

    add_shape = img_res[:, :, x:(x+2*r2+1), y:(y+2*r1+1)]
    add_shape[torch.where((add_shape <= 0) & (shape_template > 0))] = c
    
    img_res[:,:, x:(x+2*r2+1), y:(y+2*r1+1)] =  add_shape 
    return img_res

## AddTissueShape: Adds two Random Ellipses, to simulate tissue flaps that may have been altered/lost compared to the reference image
# Note: It's important to apply this Augmentation at the beginning, so all following noise functions are applied to this artificial "Tissue"
# Note2: The intensity of these Ellipses are sampled between zero and the median of all non-zero pixels of the input batch to simulate realistic non-tumor tissue
# Dependency: function RandomEllipse
class AddTissueShape(torch.nn.Module):
    def forward(self, img): 
        img_res = img.clone()
        c = np.random.sample() * torch.mean(img) 
        img_res = RandomEllipse(img_res, c)
        img_res = RandomEllipse(img_res, c)
        return img_res
    
    
## Perlin Noise: Simulate smooth intensity gradients between Ref & ion images. Multiplicative noise.
# Dependency: package "perlin_numpy" (https://github.com/pvigier/perlin-numpy, last checked: 17th April 2026)  
# Note: The function originally creates output between -1 and 1; here we calculate *(1/3) and +1 to scale between 2/3 and 4/3 and have less extrem noise, 
# Note2: Output may be larger than 1      
class PerlinNoise(torch.nn.Module):
    def forward(self, img):  
        noise = generate_fractal_noise_2d((input_size, input_size), (4, 4))/3 +1 
        noise = torch.from_numpy(noise).to(dtype=torch.float32).cuda() 
        img = img * noise 
        return img

## Poisson Noise: Simulates noise specifially for MALDI-MSI / conut intensities. 
# Source: Verbececk et.al, 2019, Unsupervised machine learning for exploratory data... S.252:
#           "This means that both the noise and variability of the signal are likely to be governed by Poisson
#             statistics (Keenan & Kotula, 2004a) and will not necessarily approximate a Gaussian distribution"
# Note: torch.poisson doesn't compute for negative values and the output are integers. 
#       That's why the image is converted to integer color space first and the signs are reapplied afterwards.
class PoissonNoise(torch.nn.Module):
    def forward(self, img):  
        img_sign = torch.sign(img)
        img = torch.poisson(img_sign * img * 255)
        img = (img * img_sign)/255 
        return img    
        
## Adjust Background: Adds or substracts a baseline value to all non-zero pixels. Trains network to ignore baseline-intensity changes.
# Note: Chooses random constant between -0.1 and 0.1, and adds it to all non-zero values
# Input: img >= 0
# Output: can be lower than 0, or larger than 1
class AdjustBackground(torch.nn.Module):
    def forward(self, img): 
        add = torch.zeros((img.shape)).to(dtype=torch.float32).cuda() 
        constant_add = (torch.rand(1)*0.8 - 0.4).to(dtype=torch.float32).cuda() 
        add[img > 0] = constant_add
        img = img + add 
        img[img < 0] = 0

        img = img / (1+constant_add)
        
        return img  
           
## Zero Pixels: Sets random Pixels to Zero, similar to MALDI behaviour. The amount of pixels set to zero is chosen randomly between 0-20 % of all pixels
# Source of Idea: Guo et.al., 2024, DeepION: A Deep Learning-Based Low-Dimensional Represetation Model...
# Note: Draw random integer between 0 and 20% of all pixels as random amount (amount0). Then draw amount0 random x and y values for the missing pixels.
# Note (23.09): SetPixel Function: Only set non-zero pixels to a 1, up to 10%, and up to 20% to 0.
def SetPixels(img, set_to = 0, max_percent = 0.15):
    s = input_size
    amount0 = np.random.randint(0, s**2 * max_percent)
    x = np.random.randint(0, s, amount0)
    y = np.random.randint(0, s, amount0)
    img = img.clone()
    img[:,:,x,y] = torch.where(img[:,:,x,y] > 0, set_to , 0).to(dtype=torch.float32).cuda() 
    return img 

class SaltPepperNoise(torch.nn.Module):
    def forward(self, img): 
        img_res = img.clone()
        img_res = SetPixels(img, 1, 0.2)
        img_res = SetPixels(img_res, 0, 0.2) 
        return img_res      
      
## !
# mean c between -0.3 and 0.3, sigma s between 0.05 and 0.15 (default is 0.1)      
class RanodmGaussianNoise(torch.nn.Module):
    def forward(self, img): 
        img_res0 = img.clone()
        c = ((torch.rand(1)*0.6) - 0.3 ).to(dtype=torch.float32).cuda()
        s = ((torch.rand(1)*0.1) + 0.05 ).to(dtype=torch.float32).cuda()
        img_res = v2.GaussianNoise(mean=c, sigma=s)(img_res0)
        img_res = torch.where(img_res0 > 0, img_res, 0) 
        img_max = torch.amax(img,dim=(1,2,3)).reshape(-1,1,1,1).expand([-1,3,232,232]).to(dtype=torch.float32).cuda()
        img_res = img_res / img_max
        
        return img_res   
 
#################################### Reference Augmentations ####################################
## Function to Add "Guillotine Partition" around an object
# Dependecy: package "cv2"
# Input: single img with all three colour channels. The pixel-size of the rectangles and the fill value. 
#        img background has to be zero, otherwise thresholding at 1 wouldn't make sense
# Note: I didn't find a way to do this batchwise. May be ineffective to convert to numpy, then back for each image independently
# Idea: Create rectangular bounding box around all non-zero pixels. Partition it in smaller rectangles of fixed size. 
#       Fill those rectangles that contain at least one non-zero pixel with fill_value by choosing maximum of fill_value and original intensity
def GuillotineBackground(img_input, rect_size = 10, fill_value = 20): 
    img = img_input.cpu()
    img = np.array(img * 255, dtype=np.uint8) 
    _, binary_mask = cv2.threshold(img[0], 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    rectangles = []

    for i in range(x, x + w, rect_size):
        for j in range(y, y + h, rect_size):
            sub_w = min(rect_size, x + w - i)  
            sub_h = min(rect_size, y + h - j)  
            if np.any(img[:, j:j+sub_h, i:i+sub_w] > 0):  
                rectangles.append((i, j, sub_w, sub_h))
     
    for rect in rectangles:
        x, y, w, h = rect
        img[:,y:y+h, x:x+w] = np.maximum(img[:,y:y+h, x:x+w], fill_value)
    
    img =  torch.from_numpy(img/255).to(dtype=torch.float32).cuda()
    return img

## Function to Add "Guillotine Partition" around an object to simulate the cutout that appears only in the References
# Dependecy: function GuillotineBackground, package "cv2"
# Note: Apply GuillotineBackground Functions to each image in the batch. Draw random rect size (5-23=232*10%)and fill value (0-26=255*10%) in the beginning.
class GuillotineBackground_batch(torch.nn.Module):
    def forward(self, img): 
        img_res = img.clone()
        batch_size = img.shape[0]
        rd_rect_size = np.random.randint(5,23) 
        rd_fill_value = np.random.randint(0,26)
        for b in range(batch_size):
            img_res[b] = GuillotineBackground(img[b], rect_size = rd_rect_size, fill_value = rd_fill_value)  
        return img_res
    
##################################### Augmentation Pipeline #####################################
# General Note: For my own Augmentation no random Apply is applied because they intrinsically contain randomness

## Augmentation pipeline to Simulate the Reference
# apply rotation, affine trafo and guillotine partition
# Note: Rotation is not applied within RandomAffine, because they don't have option "expand" to avoid cutting tissue edges.
#        After slight Random Rotation img is resized to input size.
augment_ref = torch.nn.Sequential(
    v2.RandomRotation(20, expand=True),
    v2.Resize(input_size),
    v2.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1), shear=2),
    v2.GaussianBlur((1, 7), (0.1, 2.0)),
    
    v2.RandomApply(torch.nn.ModuleList([
        GuillotineBackground_batch(),
    ]), p=0.5),
    
    preprocess
)


# color jitter of default byol: ColorJitter(brightness=(0.19999999999999996, 1.8), contrast=(0.19999999999999996, 1.8), saturation=(0.19999999999999996, 1.8), hue=(-0.2, 0.2))
# NOTE ColorJitter Brightness: effectively image * brightness_factor (deep in documentation: Pil enhance = PIL blend of 0-image with original and alpha=1+/-brightness (alpha=0 is black image, 1 is original))
augment_maldi = torch.nn.Sequential(
    AddTissueShape(),
    
    v2.RandomApply(torch.nn.ModuleList([ 
        SaltPepperNoise(), 
    ]), p=0.8),

    AdjustBackground(),
    
    v2.RandomApply(torch.nn.ModuleList([
        v2.GaussianBlur((1, 7), (0.1, 2.0)),
    ]), p=0.8),
    
                
    v2.RandomApply(torch.nn.ModuleList([
        PerlinNoise(),
    ]), p=0.8),

    PoissonNoise(),

    v2.RandomApply(torch.nn.ModuleList([
        RanodmGaussianNoise(),
    ]), p=0.5),
    
    v2.RandomApply(torch.nn.ModuleList([
        v2.ColorJitter(brightness=0.8, contrast=0.8),
    ]), p=0.5),
    
    preprocess
)



