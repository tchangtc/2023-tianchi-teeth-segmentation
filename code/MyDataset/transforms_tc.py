from typing import Any
import cv2
import pdb
import torch
import torch.nn as nn
from PIL import Image
import math
import random
import numpy as np
from torchvision import transforms
# import torch.nn.functional as F
import torchvision.transforms.functional as TTF
from scipy.ndimage import gaussian_filter


# transforms.RandomHorizontalFlip
# transforms.RandomVerticalFlip
# transforms.RandomErasing
# transforms.ColorJitter
# transforms.GaussianBlur(kernel_size=[3, 3], sigma=(0.2, 1.5))
# transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
# transforms.GaussianBlur(kernel_size=[3, 3], sigma=(0.2, 1.5)),
# transforms.LinearTransformation
# transforms.FiveCrop
# transforms.TenCrop

def make_transforms(image_set):
    if image_set == 'Train':
        
        # return None
        return transforms.Compose([
            
            RandomHVFlip(p_per_sample=0.5),
            RandomRotate90(p_per_sample=0.5),
            
            RandomHVFlip(p_per_sample=0.5),

            RandomNoise(p_per_sample=0.5, mag=0.1),                
            
            # RandomSelect(
            #     # GaussianBlur(p_per_sample=0.3, kernel_size=[3, 3]),
            #     # ColorJitter(p_per_sample=0.3),
            #     p_per_sample=0.5,
            # ),
            
            # RandomSelect(
            RandomContrast(p_per_sample=0.5, mag=0.40),  
            # RandomHSV(p_per_sample=0.5, mag=[0.40, 0.40, 0]),
            #     p_per_sample=0.5
            # ),

            RandomRotateScale(p_per_sample=0.5, angle=30, scale=[0.5, 2.0]),

            # RandomSelect(
            RandomCutout(p_per_sample=0.5),
            # RandomErasing(p_per_sample=0.5),
                # p_per_sample=0.5
            # ),

            # ElasticTransforms(p_per_sample=0.25),

        ])
    else:

        return None
    


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p_per_sample):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p_per_sample

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() < self.p:
            return self.transforms1(sample)
        return self.transforms2(sample)
    
# def do_random_hflip(image, ):
#     if np.random.rand()<0.5:
#         image = cv2.flip(image,1)
#     return image
# class RandomHFlip(object):

#     def __init__(self, p_per_sample) -> None:
#         self.p = p_per_sample
        
#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
#         if np.random.rand() < self.p:
#             image = cv2.flip(image, 1)
#             mask  = cv2.flip(mask, 1)
#         return {'image': image, 'mask': mask}

class RandomHVFlip(object):
    def __init__(self, p_per_sample) -> None:
        self.p = p_per_sample
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            image = cv2.flip(image, 1)  # horizontal
            mask  = cv2.flip(mask, 1)
        if np.random.rand() < self.p:   # vertical
            image = cv2.flip(image, 0)
            mask  = cv2.flip(mask, 0)
        if np.random.rand() < self.p:   # both
            image = cv2.flip(image, -1)
            mask  = cv2.flip(mask, -1)
        return {'image': image, 'mask': mask}

# class RandomHVFlip(object):
#     def __init__(self, p_per_sample) -> None:
#         self.p = p_per_sample

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
        
#         if np.random.rand() < self.p:
#             image = TTF.hflip(image)
#             mask  = TTF.hflip(mask)

#         if np.random.rand() < self.p:
#             image = TTF.vflip(image)
#             mask  = TTF.vflip(mask)

#         return {'image': image, 'mask': mask}

class RandomErasing(object):
    def __init__(self, p_per_sample=0.5, *args, **kwargs):
        self.p = p_per_sample
        self.eraser = transforms.RandomErasing(*args, **kwargs)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            image = self.eraser(image)
        return {'image': image, 'mask': mask}

class GaussianBlur(object):
    def __init__(self, p_per_sample, *args, **kwargs):
        self.p = p_per_sample
        self.gaussianblur = transforms.GaussianBlur(*args, **kwargs)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            # image = 
            image = self.gaussianblur(image)

        return {'image': image, 'mask': mask}

class ColorJitter(object):
    def __init__(self, p_per_sample, *args, **kwargs):
        self.p = p_per_sample
        self.colorjitter = transforms.ColorJitter(*args, **kwargs)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            image = self.colorjitter(image)

        return {'image': image, 'mask': mask}


# def do_random_rot90(image, mask):
#     r = np.random.choice([
#         0,
#         cv2.ROTATE_90_CLOCKWISE,
#         cv2.ROTATE_90_COUNTERCLOCKWISE,
#         cv2.ROTATE_180,
#     ])
#     if r==0:
#         return image, mask
#     else:
#         image = cv2.rotate(image, r)
#         mask = cv2.rotate(mask, r)
#         return image, mask

class RandomRotate90(object):
    def __init__(self, p_per_sample) -> None:
        self.p = p_per_sample
        
    def __call__(self, sample):
        # pdb.set_trace()
        image, mask = sample['image'], sample['mask']

        if np.random.rand() < self.p:
            r = np.random.choice([
                0,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                cv2.ROTATE_180,
            ])
            if r==0:
                return {'image': image, 'mask': mask}

            else:
                image = cv2.rotate(image, r)
                mask = cv2.rotate(mask, r)
                return {'image': image, 'mask': mask}

            
        return {'image': image, 'mask': mask}
    


# def do_random_noise(image, mask, mag=0.1):
# 	height, width = image.shape[:2]
# 	noise = np.random.uniform(-1,1, (height, width,1))*mag
# 	image = image + noise
# 	image = np.clip(image,0,1)
# 	return image, mask


class RandomNoise(object):

    def __init__(self, p_per_sample, mag=0.1) -> None:
        self.p = p_per_sample
        self.mag = mag
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']


        if np.random.rand() < self.p:

            height, width = image.shape[:2]
            noise = np.random.uniform(-1,1, (height, width,1)) * self.mag
            image = image + noise
            image = np.clip(image, 0, 1)

        return {'image': image, 'mask': mask}
    
#intensity
# def do_random_contrast(image, mask, mag=0.3):
# 	alpha = 1 + random.uniform(-1,1)*mag
# 	image = image * alpha
# 	image = np.clip(image,0,1)
# 	return image, mask


class RandomContrast(object):

    def __init__(self, p_per_sample, mag=0.1) -> None:
        self.p = p_per_sample
        self.mag = mag
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.rand() < self.p:
            alpha = 1 + random.uniform(-1, 1) * self.mag
            image = image * alpha
            image = np.clip(image, 0, 1)

        return {'image': image, 'mask': mask}



# def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
# 	image = (image*255).astype(np.uint8)
# 	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
# 	h = hsv[:, :, 0].astype(np.float32)  # hue
# 	s = hsv[:, :, 1].astype(np.float32)  # saturation
# 	v = hsv[:, :, 2].astype(np.float32)  # value
# 	h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
# 	s =  s*(1 + random.uniform(-1,1)*mag[1])
# 	v =  v*(1 + random.uniform(-1,1)*mag[2])
	
# 	hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
# 	hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
# 	hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
# 	image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# 	image = image.astype(np.float32)/255
# 	return image, mask


class RandomHSV(object):

    def __init__(self, p_per_sample, mag=0.1) -> None:
        self.p = p_per_sample
        self.mag = mag
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            image = (image*255).astype(np.uint8)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            h = hsv[:, :, 0].astype(np.float32)  # hue
            s = hsv[:, :, 1].astype(np.float32)  # saturation
            v = hsv[:, :, 2].astype(np.float32)  # value
            h = (h*(1 + random.uniform(-1,1) * self.mag[0])) % 180
            s =  s*(1 + random.uniform(-1,1) * self.mag[1])
            v =  v*(1 + random.uniform(-1,1) * self.mag[2])
            
            hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
            hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
            hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            image = image.astype(np.float32)/255
        return {'image': image, 'mask': mask}
    

# def do_random_rotate_scale(image, mask, angle=30, scale=[0.8,1.2] ):
#     angle = np.random.uniform(-angle, angle)
#     scale = np.random.uniform(*scale) if scale is not None else 1
    
#     height, width = image.shape[:2]
#     center = (height // 2, width // 2)
    
#     transform = cv2.getRotationMatrix2D(center, angle, scale)
#     image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
#                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#     mask  = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR,
#                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#     return image, mask

class RandomRotateScale(object):

    def __init__(self, p_per_sample, angle=30, scale=[0.8,1.2]) -> None:
        self.p = p_per_sample
        self.angle = angle
        self.scale = scale
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            angle = np.random.uniform(-self.angle, self.angle)
            scale = np.random.uniform(*self.scale) if self.scale is not None else 1
            
            height, width = image.shape[:2]
            center = (height // 2, width // 2)
            
            transform = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            mask  = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return {'image': image, 'mask': mask}
    

# def do_random_crop(image, mask, size):
#     height, width = image.shape[:2]
#     x = np.random.choice(width -size) if width>size else 0
#     y = np.random.choice(height-size) if height>size else 0
#     image = image[y:y+size,x:x+size]
#     mask  = mask[y:y+size,x:x+size]
#     return image, mask


# class RandomCrop(object):

#     def __init__(self, p_per_sample, size) -> None:
#         self.p = p_per_sample
#         self.size = size
    
#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
#         if np.random.rand() < self.p:
#             height, width = image.shape[:2]
#             x = np.random.choice(width - self.size) if width > self.size else 0
#             y = np.random.choice(height - self.size) if height > self.size else 0
#             image = image[y : y + self.size, x : x + self.size]
#             mask  = mask[y : y + self.size, x : x + self.size]
#         return image, mask
    




# def do_random_cutout(image, mask, num_block=5, block_size=[0.1,0.3], fill='constant'):
#     assert image.shape[:2] == mask.shape
#     assert image.shape[0] == mask.shape[1]
#     assert image.shape[0] == mask.shape[1]

#     height, width = image.shape[:2]

#     num_block = np.random.randint(1,num_block+1)
#     for n in range(num_block):
#         h = np.random.uniform(*block_size)
#         w = np.random.uniform(*block_size)
#         h = int(h*height)
#         w = int(w*width)
#         x = np.random.randint(0,width-w)
#         y = np.random.randint(0,height-h)
#         if fill=='constant':
#             image[y:y+h,x:x+w]=0
#             mask[y:y+h,x:x+w]=0
#         else:
#             raise NotImplementedError
#         return {'image': image, 'mask': mask}

class RandomCutout(object):

    def __init__(self, p_per_sample, num_block=5, block_size=[0.1,0.3], fill='constant') -> None:
        self.p = p_per_sample
        self.num_block = num_block
        self.block_size = block_size
        self.fill = fill
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # pdb.set_trace()
        assert image.shape[:2] == mask.shape
        assert image.shape[0] == mask.shape[0]
        assert image.shape[1] == mask.shape[1]
        if np.random.rand() < self.p:

            height, width = image.shape[:2]

            num_block = np.random.randint(1, self.num_block + 1)
            for n in range(num_block):
                h = np.random.uniform(*self.block_size)
                w = np.random.uniform(*self.block_size)
                h = int(h * height)
                w = int(w * width)
                x = np.random.randint(0, width - w)
                y = np.random.randint(0, height - h)
                if self.fill=='constant':
                    image[y : y + h, x :x + w] = 0
                    mask[y : y + h, x : x + w] = 0
                else:
                    raise NotImplementedError
        return {'image': image, 'mask': mask}
    


# def do_elastic_transform(image, mask, alpha=120, sigma=120* 0.05, alpha_affine=120* 0.03):
#     """Elastic deformation of images as described in [Simard2003]_ (with modifications).
#     Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#          Convolutional Neural Networks applied to Visual Document Analysis", in
#          Proc. of the International Conference on Document Analysis and
#          Recognition, 2003.
#     """
#     assert image.shape[:2] == mask.shape
#     assert image.shape[0] == mask.shape[1]
#     assert image.shape[0] == mask.shape[1]
#     height, width = image.shape[:2]

#     # Random affine
#     center_square = np.array((height, width), dtype=np.float32) // 2
#     square_size = min((height, width)) // 3
#     alpha = float(alpha)
#     sigma = float(sigma)
#     alpha_affine = float(alpha_affine)

#     pts1 = np.array(
#         [
#             center_square + square_size,
#             [center_square[0] + square_size, center_square[1] - square_size],
#             center_square - square_size,
#         ],
#         dtype=np.float32,
#     )
#     pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(
#         np.float32
#     )
#     matrix = cv2.getAffineTransform(pts1, pts2)

#     image = cv2.warpAffine(image, M=matrix, dsize=(width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#     mask = cv2.warpAffine(mask, M=matrix, dsize=(width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


#     if 1:
#         dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
#         dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha


#     x, y = np.meshgrid(np.arange(width), np.arange(height))
#     map_x = np.float32(x + dx)
#     map_y = np.float32(y + dy)
#     image = cv2.remap( image, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

#     mask = cv2.remap( mask, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

#     return {'image': image, 'mask': mask}


class ElasticTransforms(object):

    def __init__(self, p_per_sample, alpha=120, sigma=120* 0.05, alpha_affine=120* 0.03):
        self.p = p_per_sample
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
    

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        assert image.shape[:2] == mask.shape
        assert image.shape[0] == mask.shape[0]
        assert image.shape[1] == mask.shape[1]
        height, width = image.shape[:2]

        # Random affine
        center_square = np.array((height, width), dtype=np.float32) // 2
        square_size = min((height, width)) // 3
        alpha = float(self.alpha)
        sigma = float(self.sigma)
        alpha_affine = float(self.alpha_affine)

        pts1 = np.array(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ],
            dtype=np.float32,
        )
        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(
            np.float32
        )
        matrix = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image, M=matrix, dsize=(width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M=matrix, dsize=(width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


        if 1:
            dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha


        x, y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)
        image = cv2.remap( image, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        mask = cv2.remap( mask, map1=map_x, map2=map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return {'image': image, 'mask': mask}
    







class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob
 
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)
 
    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
 
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))
 
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
 
        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)
 
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
 
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
 
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
 
        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
 
        mask = mask.expand_as(img)
        img = img * mask
 
        return img
 
 
class GridMask(nn.Module):
    def __init__(self, d1=96, d2=224, rotate=360, ratio=0.4, mode=1, prob=0.8):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)
 
    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)
 
    def forward(self, img):
        if not self.training:
            return img
 
        n, c, h, w = img.size()
        y = []
        for i in range(n):
            y.append(self.grid(img[i]))
 
        y = torch.cat(y).view(n, c, h, w)
 
        return y