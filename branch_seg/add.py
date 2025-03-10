# import cv2
#
# img1 = cv2.imread('C:/Users/LX/Desktop/1.tif')
# img2 = cv2.imread('C:/Users/LX/Desktop/Figure_2.tif')
# print(img1.shape)
# print(img2.shape)
# # 输出： (1039, 750, 3)
# # (1050, 700, 3)
#
# img2.resize((img1.shape[0], img1.shape[1], 3))
# print(img2.shape)
# # 输出：(1039, 750, 3)
#
# res = cv2.add(img1, img2)  # 或者res=cv2.add(img1,10)
# cv2.imwrite('C:/Users/LX/Desktop/111111.tif', res)


# 文件夹批量加
import cv2
import os

file_namelist1 = 'F:/data-microglia/frangi'  # 文件夹1路径
file_namelist2 = 'F:/data-microglia/frangi-soma'  # 文件夹2路径
path = 'F:/data-microglia/frangi-total'  # 保存路径
image1 = os.listdir(file_namelist1)  # 读取文件夹名称
print(image1)

for i in image1:  # 遍历文件夹中文件名
    # print(str(i))
    img1 = cv2.imread(file_namelist1 + '/' + str(i))

    img2 = cv2.imread(file_namelist2 + '/' + str(i))
    des = cv2.add(img1, img2)
    cv2.imwrite(path + '/' + str(i), des)
