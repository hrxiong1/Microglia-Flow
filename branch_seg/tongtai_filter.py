import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 同态滤波器
def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst

def put(image):
    # image = cv2.imread(path, 1)
    # image = cv2.imread(os.path.join(base, path), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 同态滤波器
    h_image = homomorphic_filter(image, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5)
    # plt.imshow(h_image, 'gray')
    # plt.axis('off')
    # # cv2.imwrite("../images/1-tongtai.jpg", h_image)
    # plt.show()

    return h_image

# 图像处理函数，要传入路径
# put(r'../images/1-gray.tif')
