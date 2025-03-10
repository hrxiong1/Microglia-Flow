
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vesselness2d import *


def hessian(img):
    # reading image
    # image = Image.open(img).convert("RGB")
    image = Image.fromarray(img).convert("RGB")
    image = np.array(image)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image, cmap='gray')

    # convert forgeground to background and vice-versa
    # image = 255 - image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(image[(image > 0)], 1) * 0.9
    image[(image <= thr)] = thr
    image = image - np.min(image)
    image = image / np.max(image)

    sigma = [1.5, 2, 2, 3, 2.5]
    spacing = [1, 1]
    tau = 2

    output = vesselness2d(image, sigma, spacing, tau)
    output = output.vesselness2d()

    return output


if __name__ == '__main__':
    img_dir = '../images/1-tongtai_gamma.jpg'  # 路径写自己的
    output = hessian(img_dir)

    cv2.imwrite("../images/1-tongtai_gamma_hessian.jpg", output*255)
    plt.imshow(output, cmap='gray')
    plt.show()

