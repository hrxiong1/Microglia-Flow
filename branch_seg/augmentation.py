# -*- coding: utf-8 -*-
# 180 - 90 - LEFT_RIGHT - TOP_BOTTOM - 270 - invert

from PIL import Image, ImageChops
import os
import cv2 as cv

file_dir = 'F:/data-microglia/unet-origin/SegmentationClass/'  # 原始图片路径
rotate_180 = 'F:/data-microglia/unet-origin/SegmentationClass1/'  # 保存路径

for img_name in os.listdir(file_dir):
    img_path = file_dir + img_name  # 批量读取图片
    # print(img_path)
    img = Image.open(img_path)
    LEFT_RIGHT = img.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
    TOP_BOTTOM = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转
    rotated_90 = img.transpose(Image.ROTATE_90)  # 旋转90度
    rotated_180 = img.transpose(Image.ROTATE_180)  # 旋转180度
    rotated_270 = img.transpose(Image.ROTATE_270)  # 旋转270度
    # invert = ImageChops.invert(img)
    rotated_270.save(rotate_180 + '270_' + img_name)  # 保存旋转后的图片

print('finish')

