import cv2
import matplotlib.pyplot as plt


def ostu(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    return threshold

if __name__ == '__main__':
    img = cv2.imread('../images/1-tongtai_gamma_hessian.jpg')
    threshold = ostu(img)
    cv2.imshow("res", threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

