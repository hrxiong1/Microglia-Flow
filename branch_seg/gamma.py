import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


# 伽马变换，有参数,利用幂函数运算

def gamma_bianhuan(image, gamma):
    image = image / 255.0
    New = np.power(image, gamma)
    return New


if __name__ == '__main__':  ##启动语句
    a = cv2.imread('../images/1-tongtai.jpg')  # 路径名中不能有中文，会出错,cv2.
    image1 = cv2.split(a)[0]  # 蓝
    # image2 = cv2.split(a)[1]  # 绿
    # image3 = cv2.split(a)[2]  # 红

    # image_1 = gamma_bianhuan(image1, 1.5)
    image_1_2 = gamma_bianhuan(image1, 0.5)
    # image_2 = gamma_bianhuan(image2, 0.5)
    # image_3 = gamma_bianhuan(image3, 0.5)
    # merged = cv2.merge([image1, image2, image3])  # 合并三通道
    cv2.imshow("1", image_1_2)  # 处理后的蓝色通道

    # cv2.imshow("2", image_1)  # 处理后的蓝色通道
    # cv2.imshow("3", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("../images/1-tongtai_gamma.jpg", image_1_2*255)  #  输出处理后的蓝色通道
