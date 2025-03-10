
"""
同态滤波 —— gamma变换 -- hessian变换
"""
from tongtai_filter import homomorphic_filter, put
from gamma import gamma_bianhuan
from hessian_filter import hessian
from ostu import ostu
import cv2
from PIL import Image
import numpy as np
import os

def ske(img):
    img_tongtai = put(img)

    img_tongtai = cv2.split(img_tongtai)[0]
    img_gamma = gamma_bianhuan(img_tongtai, 0.5)

    img_gamma = img_gamma * 255
    img_hessian = hessian(img_gamma)

    img_hessian = img_hessian * 255
    img_hessian = Image.fromarray(img_hessian).convert("RGB")
    img_hessian = np.array(img_hessian)
    img_binary = ostu(img_hessian)

    return img_binary


if __name__ == '__main__':
    filepath = ".\\images"
    filename = os.listdir(filepath)  # 图像名列表
    base_dir = filepath + "\\"
    new_dir = ".\\output"  # 以\\结尾
    for img in filename:
        im = Image.open(base_dir + img).convert("RGB")
        im = np.array(im)
        out = ske(im)
        cv2.imwrite(new_dir + "\\" + img, out)

