
import cv2

def read_image(path):
    return cv2.imread(path)

def save_image(img, path):
    cv2.imwrite(path, img)
