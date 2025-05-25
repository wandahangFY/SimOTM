# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


def receptiveField(img, R=3, r=1, fac_r=-1, fac_R=6):
    # img1 = np.float32(img)

    x, y = np.meshgrid(np.arange(1, R * 2 + 2), np.arange(1, R * 2 + 2))
    dis = np.sqrt((x - (R + 1)) ** 2 + (y - (R + 1)) ** 2)
    flag1 = (dis <= r)
    flag2 = np.logical_and(dis > r, dis <= R)
    kernal = flag1 * fac_r + flag2 * fac_R
    # kernal /= kernal.sum()
    kernal = kernal / kernal.sum()
    out = cv2.filter2D(img, -1, kernal)
    return out


def SimOTM(img):
    blur = cv2.blur(img, (3, 3))
    rec = receptiveField(img)
    result = cv2.merge([img, blur, rec])
    return result

def SimOTMBBS(img):
    blur = cv2.blur(img, (3, 3))
    result = cv2.merge([img, blur, blur])
    return result

def SimOTMSSS(img):
    #  TIF  16 bit
    result = cv2.merge([img, img, img])
    return result

def enhance_brightness_or_contrast(image, target_gray_value, brightness_alpha=1.5, contrast_alpha=1.0, beta=0):
    gray_value = np.mean(image)
    if gray_value >= target_gray_value:
        enhanced_image = cv2.convertScaleAbs(image, alpha=contrast_alpha, beta=beta)
    else:
        avg_diff = target_gray_value - gray_value
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.0, beta=avg_diff)
    return enhanced_image

def SimOTMBrights(img):
    blur = cv2.blur(img, (3, 3))
    rec = receptiveField(img)
    result = cv2.merge([img, blur, rec])
    return result






def OneToThreeEqualizeHist(img):
    equ = cv2.equalizeHist(img)
    result = cv2.merge([equ, equ, equ])
    return result

def OneToThreeCLAHE(img):
    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)
    result = cv2.merge([dst, dst, dst])
    return result

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    # dst = clahe.apply(img)



def OneToThreeEqualizeHist_ADD_CLAHE(img):
    equ = cv2.equalizeHist(img)

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)

    result = cv2.merge([0.5 * equ + 0.5 * dst, 0.5 * equ + 0.5 * dst, 0.5 * equ + 0.5 * dst])
    return result


def OneToThreeEqualizeHist_Muti_CLAHE(img):
    equ = cv2.equalizeHist(img)

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)

    result = cv2.merge([img,dst, equ])
    return result


def OneToThreeEqualizeHist_THEN_CLAHE(img):
    equ = cv2.equalizeHist(img)

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(equ)

    result = cv2.merge([ dst, dst,  dst])
    return result


def OneToThree_EqualizeHist(img):
    equ = cv2.equalizeHist(img)

    # # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # # 限制对比度的自适应阈值
    # dst = clahe.apply(equ)

    result = cv2.merge([ equ, equ,  equ])
    return result

def OneToThree_CLAHE(img):
    # equ = cv2.equalizeHist(img)

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)

    result = cv2.merge([ dst, dst,  dst])
    return result


def OneToThreeCLAHE_THEN_EqualizeHist(img):


    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)
    equ = cv2.equalizeHist(dst)

    result = cv2.merge([equ, equ, equ])
    return result

def OneToThree_Sharpen(img):

    # # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # # 限制对比度的自适应阈值
    # dst = clahe.apply(img)
    # equ = cv2.equalizeHist(dst)

    # sharpen_op = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]], dtype=np.float32)
    sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
    sharpen_image = cv2.convertScaleAbs(sharpen_image)

    result = cv2.merge([sharpen_image, sharpen_image, sharpen_image])
    return result
# ————————————————
# 版权声明：本文为CSDN博主「lovefive55」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_41709536/article/details/100889298


def OneToThree_receptiveField(img):
    # blur = cv2.blur(img, (3, 3))
    # blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    # img_d=(np.uint8)(img * 0.7)
    # image = np.power(img, 0.8)  # 对像素值指数变换
    # dst = clahe.apply(img)

    # # 线性变换
    # a = 1.4
    # O = (np.uint8)(float(a) * img)
    # O[O > 255] = 255  # 大于255要截断为255

    rec = receptiveField(img)
    result = cv2.merge([rec, rec, rec])
    # # 图像归一化
    # fI = img / 255.0
    # # 伽马变化
    # gamma = 0.4
    # O = np.power(fI, gamma)
    # img_re = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # img_re[:, :, 0] = img
    # img_re[:, :, 1] = blur
    # img_re[:, :, 2] = dst

    # cv2.imwrite()
    return result



def OneToThreeORDERCLAHEUSM(img):
    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)
    blur_img = cv2.GaussianBlur(dst, (0, 0), 5)
    usm = cv2.addWeighted(dst, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([usm, usm, usm])
    return result

def OneToThree_CLAHE_USM(img):
    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)
    blur_img = cv2.GaussianBlur(dst, (0, 0), 5)
    usm = cv2.addWeighted(dst, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([usm, usm, usm])
    return result

def OneToThree_CLAHE_USM_P(img):
    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    dst = clahe.apply(img)

    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([dst, img, usm])
    return result

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    # dst = clahe.apply(img)

def OneToThreeUSM(img):
    # USM锐化增强方法(Unsharpen Mask)
    # 先对原图高斯模糊，用原图减去系数x高斯模糊的图像
    # 再把值Scale到0~255的RGB像素范围
    # 优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
    # （原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
    # src = cv.imread("ma.jpg")
    # cv.imshow("input", src)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([usm, usm, usm])
    return result
    # cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    # cv.imshow("mask image", usm)

def OneToThreeGaussianBlur(img):
    # USM锐化增强方法(Unsharpen Mask)
    # 先对原图高斯模糊，用原图减去系数x高斯模糊的图像
    # 再把值Scale到0~255的RGB像素范围
    # 优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
    # （原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
    # src = cv.imread("ma.jpg")
    # cv.imshow("input", src)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    # usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([blur_img, blur_img, blur_img])
    return result
    # cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    # cv.imshow("mask image", usm)


def OneToThree2_ADD_RB(img):
    blur = cv2.blur(img, (3, 3))
    # blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    # dst = clahe.apply(img)

    # img_d=(np.uint8)(img * 0.7)
    # image = np.power(img, 0.8)  # 对像素值指数变换

    rec=receptiveField(img)

    result = cv2.merge([0.5*blur+0.5*rec, 0.5*blur+0.5*rec, 0.5*blur+0.5*rec])
    # # 图像归一化
    # fI = img / 255.0
    # # 伽马变化
    # gamma = 0.4
    # O = np.power(fI, gamma)
    # img_re = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # img_re[:, :, 0] = img
    # img_re[:, :, 1] = blur
    # img_re[:, :, 2] = dst


    # cv2.imwrite()
    return result


def OneToThree2_ORDER_BR(img):
    blur = cv2.blur(img, (3, 3))
    # blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    # dst = clahe.apply(img)

    # img_d=(np.uint8)(img * 0.7)
    # image = np.power(img, 0.8)  # 对像素值指数变换

    rec=receptiveField(blur)
    result = cv2.merge([rec, rec, rec])
    # # 图像归一化
    # fI = img / 255.0
    # # 伽马变化
    # gamma = 0.4
    # O = np.power(fI, gamma)
    # img_re = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # img_re[:, :, 0] = img
    # img_re[:, :, 1] = blur
    # img_re[:, :, 2] = dst


    # cv2.imwrite()
    return result

def OneToThree2_ORDER_RB(img):
    # blur = cv2.blur(img, (3, 3))
    # blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    # 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值
    # dst = clahe.apply(img)

    # img_d=(np.uint8)(img * 0.7)
    # image = np.power(img, 0.8)  # 对像素值指数变换

    rec=receptiveField(img)
    blur = cv2.blur(rec, (3, 3))
    result = cv2.merge([blur, blur, blur])
    # # 图像归一化
    # fI = img / 255.0
    # # 伽马变化
    # gamma = 0.4
    # O = np.power(fI, gamma)
    # img_re = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # img_re[:, :, 0] = img
    # img_re[:, :, 1] = blur
    # img_re[:, :, 2] = dst


    # cv2.imwrite()
    return result





def OneToThreeGaussianBlur3_3(img):
    # USM锐化增强方法(Unsharpen Mask)
    # 先对原图高斯模糊，用原图减去系数x高斯模糊的图像
    # 再把值Scale到0~255的RGB像素范围
    # 优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
    # （原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
    # src = cv.imread("ma.jpg")
    # cv.imshow("input", src)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), 3)
    # usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([blur_img, blur_img, blur_img])
    return result
    # cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    # cv.imshow("mask image", usm)

def OneToThreeGaussianBlur5_5(img):
    # USM锐化增强方法(Unsharpen Mask)
    # 先对原图高斯模糊，用原图减去系数x高斯模糊的图像
    # 再把值Scale到0~255的RGB像素范围
    # 优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
    # （原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
    # src = cv.imread("ma.jpg")
    # cv.imshow("input", src)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), 5)
    # usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([blur_img, blur_img, blur_img])
    return result
    # cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    # cv.imshow("mask image", usm)

def OneToThreeGaussianBlur7_7(img):
    # USM锐化增强方法(Unsharpen Mask)
    # 先对原图高斯模糊，用原图减去系数x高斯模糊的图像
    # 再把值Scale到0~255的RGB像素范围
    # 优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
    # （原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
    # src = cv.imread("ma.jpg")
    # cv.imshow("input", src)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), 7)
    # usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    result = cv2.merge([blur_img, blur_img, blur_img])
    return result
    # cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    # cv.imshow("mask image", usm)


def OneToThree2_MeanBlur3_3(img):
    blur = cv2.blur(img, (3, 3))
    # blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    result = cv2.merge([blur, blur, blur])

    return result

def OneToThree2_MeanBlur5_5(img):
    blur = cv2.blur(img, (5, 5))
    # blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    result = cv2.merge([blur, blur, blur])

    return result


def OneToThree2_MeanBlur7_7(img):
    blur = cv2.blur(img, (7, 7))
    # blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    result = cv2.merge([blur, blur, blur])

    return result

def OneToThree2_MedianBlur3_3(img):
    # blur = cv2.blur(img, (7, 7))
    blur = cv2.medianBlur(img, 3)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    result = cv2.merge([blur, blur, blur])

    return result

def OneToThree2_MedianBlur5_5(img):
    # blur = cv2.blur(img, (7, 7))
    blur = cv2.medianBlur(img, 5)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    result = cv2.merge([blur, blur, blur])

    return result

def OneToThree2_MedianBlur7_7(img):
    # blur = cv2.blur(img, (7, 7))
    blur = cv2.medianBlur(img, 7)
    # blur = cv2.medianBlur(blur, 3)
    # medianBlur

    result = cv2.merge([blur, blur, blur])

    return result

def OneToThree2_BilateralFilterBlur(img):
    # blur = cv2.blur(img, (7, 7))
    # blur = cv2.medianBlur(img, 7)
    # blur = cv2.medianBlur(blur, 3)
    blur = cv2.bilateralFilter(src=img, d=0, sigmaColor=40, sigmaSpace=10)
    # medianBlur

    result = cv2.merge([blur, blur, blur])

    return result


#伽玛变换
def OneToThree2_Gamma(img, c=0.00000005, v=4.0):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut) #像素灰度值的映射
    output_img = np.uint8(output_img+0.5)
    result = cv2.merge([output_img, output_img, output_img])

    return result

#对数变换
def OneToThree2_Log(img, c=42):
    output = c * np.log(1.0 + img)
    output_img = np.uint8(output + 0.5)
    result = cv2.merge([output_img, output_img, output_img])
    return result








class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
            use_simotm="RGB"
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.use_simotm = use_simotm
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    # def load_image(self, i, rect_mode=True):
    #     """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             try:
    #                 im = np.load(fn)
    #             except Exception as e:
    #                 LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
    #                 Path(fn).unlink(missing_ok=True)
    #                 im = cv2.imread(f)  # BGR
    #         else:  # read image
    #             im = cv2.imread(f)  # BGR
    #         if im is None:
    #             raise FileNotFoundError(f"Image Not Found {f}")
    #
    #         h0, w0 = im.shape[:2]  # orig hw
    #         if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
    #             r = self.imgsz / max(h0, w0)  # ratio
    #             if r != 1:  # if sizes are not equal
    #                 w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
    #                 im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    #         elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
    #             im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
    #
    #         # Add to buffer if training with augmentations
    #         if self.augment:
    #             self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #             self.buffer.append(i)
    #             if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
    #                 j = self.buffer.pop(0)
    #                 if self.cache != "ram":
    #                     self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
    #
    #         return im, (h0, w0), im.shape[:2]
    #
    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                # im = cv2.imread(f)  # BGR
                if self.use_simotm ==  'Gray2BGR':
                    im = cv2.imread(f)  # BGR
                elif self.use_simotm == 'SimOTM':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = SimOTM(im)
                elif self.use_simotm == 'SimOTMBBS':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = SimOTMBBS(im)

                elif self.use_simotm == 'OneToThree2_Log':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThree2_Log(im)

                elif self.use_simotm == 'OneToThree2_Gamma':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThree2_Gamma(im)
                elif self.use_simotm == 'OneToThree2_MedianBlur3_3':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThree2_MedianBlur3_3(im)

                elif self.use_simotm == 'OneToThree2_MeanBlur3_3':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThree2_MeanBlur3_3(im)
                elif self.use_simotm == 'OneToThreeGaussianBlur3_3':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThreeGaussianBlur3_3(im)
                elif self.use_simotm == 'OneToThreeUSM':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThreeUSM(im)

                elif self.use_simotm == 'OneToThree_receptiveField':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThree_receptiveField(im)

                elif self.use_simotm == 'OneToThreeCLAHE':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThreeCLAHE(im)
                elif self.use_simotm == 'OneToThreeEqualizeHist':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                    im = OneToThreeEqualizeHist(im)
                elif self.use_simotm == 'Gray':
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # GRAY
                elif self.use_simotm == 'Gray16bit':
                    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)  # GRAY
                    im = im.astype(np.float32)
                elif self.use_simotm == 'SimOTMSSS':
                    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)  # TIF 16bit
                    im=im.astype(np.float32)
                    im = SimOTMSSS(im)
                elif self.use_simotm == 'RGBT':
                    im_visible = cv2.imread(f)  # BGR
                    im_infrared = cv2.imread(f.replace('visible', 'infrared'), cv2.IMREAD_GRAYSCALE)  # BGR
                    if im_visible is None or im_infrared is None:
                        raise FileNotFoundError(f"Image Not Found {f}")
                    h_vis, w_vis = im_visible.shape[:2]  # orig hw
                    h_inf, w_inf = im_infrared.shape[:2]  # orig hw

                    if h_vis!=h_inf or w_vis!=w_inf:

                        r_vis = self.imgsz / max(h_vis, w_vis)  # ratio
                        r_inf = self.imgsz / max( h_inf, w_inf )  # ratio
                        if r_vis != 1:  # if sizes are not equal
                            interp = cv2.INTER_LINEAR if (self.augment or r_vis > 1) else cv2.INTER_AREA
                            im_visible = cv2.resize(im_visible, (min(math.ceil(w_vis * r_vis), self.imgsz), min(math.ceil(h_vis * r_vis), self.imgsz)),
                                            interpolation=interp)
                        if r_inf != 1:  # if sizes are not equal
                            interp = cv2.INTER_LINEAR if (self.augment or r_inf > 1) else cv2.INTER_AREA
                            im_infrared = cv2.resize(im_infrared, (
                            min(math.ceil(w_inf * r_inf), self.imgsz), min(math.ceil(h_inf * r_inf), self.imgsz)),
                                                    interpolation=interp)

                    # 将彩色图像的三个通道分离
                    b, g, r = cv2.split(im_visible)
                    # 合并成四通道图像
                    im = cv2.merge((b, g, r, im_infrared))
                elif self.use_simotm == 'RGBRGB6C':
                    im_visible = cv2.imread(f)  # BGR
                    im_infrared = cv2.imread(f.replace('visible', 'infrared'))  # BGR
                    if im_visible is None or im_infrared is None:
                        raise FileNotFoundError(f"Image Not Found {f}")
                    h_vis, w_vis = im_visible.shape[:2]  # orig hw
                    h_inf, w_inf = im_infrared.shape[:2]  # orig hw

                    if h_vis != h_inf or w_vis != w_inf:

                        r_vis = self.imgsz / max(h_vis, w_vis)  # ratio
                        r_inf = self.imgsz / max(h_inf, w_inf)  # ratio
                        if r_vis != 1:  # if sizes are not equal
                            interp = cv2.INTER_LINEAR if (self.augment or r_vis > 1) else cv2.INTER_AREA
                            im_visible = cv2.resize(im_visible, (
                            min(math.ceil(w_vis * r_vis), self.imgsz), min(math.ceil(h_vis * r_vis), self.imgsz)),
                                                    interpolation=interp)
                        if r_inf != 1:  # if sizes are not equal
                            interp = cv2.INTER_LINEAR if (self.augment or r_inf > 1) else cv2.INTER_AREA
                            im_infrared = cv2.resize(im_infrared, (
                                min(math.ceil(w_inf * r_inf), self.imgsz), min(math.ceil(h_inf * r_inf), self.imgsz)),
                                                     interpolation=interp)

                    # 将彩色图像的三个通道分离
                    b, g, r = cv2.split(im_visible)
                    b2, g2, r2 = cv2.split(im_infrared)
                    # 合并成6通道图像
                    im = cv2.merge((b, g, r, b2, g2, r2))
                else:
                    im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin=0.5):
        """Check image caching requirements vs available disk space."""
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = cv2.imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
            return False
        return True

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
