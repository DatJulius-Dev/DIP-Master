"""
VIRTUAL GLASSES ASSET GENERATOR
Tự động sinh 4 mẫu kính thực tế ảo chuẩn kênh BGRA:
"Kính Phi Công Aviator Mạ Vàng"
"Kính Mắt Tròn Cổ Điển Retro"
"Kính Râm Thể Thao Cyber Neon"
"Kính Cận Gọng Vuông Hiện Đại"
"""

import os
import cv2
import numpy as np

def generate_aviator_glasses(width=600, height=220):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    # Mau ma vang: BGR (30, 215, 255)
    gold = (30, 215, 255, 255)
    lens_color = (40, 40, 50, 190)

    # Mat kinh trai & phai dang gio le (Aviator Teardrop)
    pts_left = np.array([
        [100, 50], [240, 50], [260, 110], [230, 180], [130, 180], [80, 110]
    ], dtype=np.int32)
    pts_right = np.array([
        [360, 50], [500, 50], [520, 110], [470, 180], [370, 180], [340, 110]
    ], dtype=np.int32)

    # Trong mat kinh
    cv2.fillPoly(img, [pts_left], lens_color)
    cv2.fillPoly(img, [pts_right], lens_color)

    # Vien ma vang
    cv2.polylines(img, [pts_left], True, gold, 6)
    cv2.polylines(img, [pts_right], True, gold, 6)

    # Cau noi song mui tren va duoi
    cv2.line(img, (240, 50), (360, 50), gold, 5)
    cv2.line(img, (255, 75), (345, 75), gold, 4)

    # Cang kinh 2 ben
    cv2.line(img, (80, 100), (10, 100), gold, 5)
    cv2.line(img, (520, 100), (590, 100), gold, 5)

    return img

def generate_retro_round_glasses(width=600, height=220):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    amber = (20, 120, 200, 255)
    lens = (80, 70, 60, 160)

    # Mat tron trai & phai
    cv2.circle(img, (180, 110), 75, lens, -1)
    cv2.circle(img, (420, 110), 75, lens, -1)
    cv2.circle(img, (180, 110), 75, amber, 8)
    cv2.circle(img, (420, 110), 75, amber, 8)

    # Cau noi song mui cong
    cv2.ellipse(img, (300, 110), (60, 25), 0, 180, 360, amber, 6)

    # Cang kinh
    cv2.line(img, (105, 110), (10, 110), amber, 6)
    cv2.line(img, (495, 110), (590, 110), amber, 6)

    return img

def generate_cyber_sport_glasses(width=600, height=220):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    cyan_neon = (255, 240, 0, 255)
    magenta_neon = (255, 0, 255, 255)
    lens_neon = (180, 50, 80, 210)

    # Kinh the thao lien khoi (Visor Shield)
    pts_shield = np.array([
        [40, 40], [560, 40], [540, 170], [300, 130], [60, 170]
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts_shield], lens_neon)
    cv2.polylines(img, [pts_shield], True, cyan_neon, 5)
    cv2.line(img, (60, 170), (300, 130), magenta_neon, 4)
    cv2.line(img, (300, 130), (540, 170), magenta_neon, 4)

    return img

def generate_square_glasses(width=600, height=220):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    black = (30, 30, 35, 255)
    lens = (120, 110, 100, 130)

    # Khung vuong trai
    cv2.rectangle(img, (100, 50), (260, 170), lens, -1)
    cv2.rectangle(img, (100, 50), (260, 170), black, 8)

    # Khung vuong phai
    cv2.rectangle(img, (340, 50), (500, 170), lens, -1)
    cv2.rectangle(img, (340, 50), (500, 170), black, 8)

    # Cau noi song mui
    cv2.line(img, (260, 95), (340, 95), black, 7)

    # Cang kinh
    cv2.line(img, (100, 95), (10, 95), black, 7)
    cv2.line(img, (500, 95), (590, 95), black, 7)

    return img

def get_glasses_dict():
    return {
        "Kính Phi Công Aviator Mạ Vàng": generate_aviator_glasses(),
        "Kính Mắt Tròn Cổ Điển Retro": generate_retro_round_glasses(),
        "Kính Râm Thể Thao Cyber Neon": generate_cyber_sport_glasses(),
        "Kính Cận Gọng Vuông Hiện Đại": generate_square_glasses()
    }