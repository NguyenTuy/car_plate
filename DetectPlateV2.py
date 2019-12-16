import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single

# Dinh nghia cac ky tu tren bien so
char_list = '0123456789ABCDEFIGHKLMNPRSTUVXYZ'


# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
img_path = "testdata/1.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

if (len(LpImg)):
    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    # cv2.imshow("normal", LpImg[0])

    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    # h, w = gray.shape
    # height = np.int (h)
    # width = np.int (w)
    # print("w-h: %f - %f" %(w, h))
    # if w < h * 1.5:
    #     top_image = gray[0: np.int(height / 2), 0: width]
    #     bottom_image = gray[np.int (height / 2): height, 0: width]
    #     gray = np.concatenate((top_image, bottom_image), axis=1)

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 127, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    # cv2.imshow("threshold", binary)
    # kernel = np.ones((5,5),np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("close", binary)
