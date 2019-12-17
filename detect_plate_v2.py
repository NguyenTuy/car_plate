import cv2
from lib_detection import detect_lp, im2single
import config

# Dinh nghia cac ky tu tren bien so
char_list = '0123456789ABCDEFIGHKLMNPRSTUVXYZ'


# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    new_string = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            new_string += lp[i]
    return new_string

def get_plate_area(path, wpod_net):

    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(path)

    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    if config.DEBUG:
        cv2.imshow("Origin", Ivehicle)

    _, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

    if (len(LpImg)):
        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        if config.DEBUG:
            cv2.imshow("Detect plate", LpImg[0])

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
        return binary
    return LpImg[0]