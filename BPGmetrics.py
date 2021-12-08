# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:36:42 2021

@author: vipvi
"""
import cv2
import math
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import imageio
import matplotlib.pyplot as plt
import os
from PIL import Image

#Image class to analyze image quality metrics
class Metrics:
    def __init__(self,img1,img2):
        self.img1 = img1
        self.img2 = img2
        
    def CR(self,bit_depth):
        input_size = (self.img1.shape[0]*self.img1.shape[1]*bit_depth)/8
        output_size = (self.img2.shape[0]*self.img2.shape[1]*bit_depth)/8
        compression_ratio = input_size/output_size
        return compression_ratio
        
    def CF(self,compression_ratio):
        return 1/compression_ratio
    
    def Convert(self):
        shape1 = self.img1.shape
        shape2 = self.img2.shape
        dict_dim1 = {}
        dict_dim2 = {}
        dimensions = ['Height','Width','Channels']
        for idx,dim in enumerate(dimensions):
            dict_dim1[dim] = shape1[idx]
            dict_dim2[dim] = shape2[idx]
        for key,value in dict_dim1.items():
            print(f'{key},{value}\n')
        for key,value in dict_dim2.items():
            print(f'{key},{value}\n')
            
    def SSIM(self):
        return ssim(self.img1,self.img2,multichannel=False)
        
    def PSNR(self,mse,bit_depth):
        if bit_depth==8:
            PIXEL_MAX = 255.0
        elif bit_depth==16:
            PIXEL_MAX = 65535.0
        elif bit_depth==24:
            PIXEL_MAX = 16777215.0    
        return 100 if mse == 0 else 20*math.log10(PIXEL_MAX/math.sqrt(mse))
    
    def MSE(self):
        err = np.sum((self.img1.astype("float") - self.img2.astype("float")) ** 2)
        err/=float(self.img1.shape[0]*self.img1.shape[1])
        return err
    
    def RMSE(self,mse):
        return math.sqrt(mse)
    
    def BPP(self,bit_depth):
        return bit_depth/8
    
    def __del__(self):
        del self.img1
        del self.img2  

metrics = []
img1 = cv2.imread('C:/Users/vipvi/Desktop/Research/BPG Research/JPEG compression/BPG/BPG/rgb16bitlinear/flower.png')
img2 = cv2.imread('C:/Users/vipvi/Desktop/Research/BPG Research/JPEG compression/BPG/BPG/rgb16bitlinear/flower_decoded.png')
IMG = Metrics(img1,img2)
IMG.Convert()
mse = IMG.MSE()
rmse = IMG.RMSE(mse)
bpp = IMG.BPP(16)
psnr = IMG.PSNR(mse,16)
#sim = IMG.SSIM()
#compression ratio calculated based on file sizes
cr = IMG.CR(16)
cf = 1/cr
metrics.extend([mse,rmse,bpp,psnr,cr,cf])
for val in metrics:
    print(val)




        
        
        
        
    
    