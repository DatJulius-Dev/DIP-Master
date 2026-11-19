"""
CÔNG CỤ KIỂM TRA THIẾT BỊ CAMERA TRÊN WINDOWS
Quét qua các cổng camera 0, 1, 2, 3, 4 với cả hai backend DirectShow và Default
để xác định máy tính có nhận diện được camera vật lý hoặc camera ảo (Iriun / DroidCam) hay không.
"""

import sys
import cv2

def main():
    print("=" * 60)
    print(" KIỂM TRA THIẾT BỊ CAMERA TRÊN HỆ ĐIỀU HÀNH")
    print("=" * 60)
    
    found_any = False
    
    for idx in range(5):
        # 1. Thử với DirectShow (chuẩn Windows)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"[HOẠT ĐỘNG TỐT] Cổng Camera {idx} (DirectShow) -> Độ phân giải: {w}x{h}")
                found_any = True
                cap.release()
                continue
            cap.release()
            
        # 2. Thử với backend mặc định (MSMF)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"[HOẠT ĐỘNG TỐT] Cổng Camera {idx} (Default backend) -> Độ phân giải: {w}x{h}")
                found_any = True
                cap.release()
                continue
            cap.release()
            
    print("=" * 60)
    if found_any:
        print("[KẾT QUẢ] Máy tính CÓ ít nhất 1 Camera đang hoạt động bình thường!")
        print("          Hãy chọn đúng số cổng Camera tương ứng trên giao diện Web App.")
    else:
        print("[KẾT QUẢ] Máy tính của bạn HIỆN TẠI CHƯA CÓ bất kỳ thiết bị Camera nào:")
        print("          - Chưa có Webcam vật lý cắm vào cổng USB.")
        print("          - Hoặc phần mềm DroidCam / Iriun trên điện thoại chưa được bấm kết nối với PC.")
        print("          -> Giải pháp: Dùng tính năng 'Tải lên video kiểm thử (.mp4)' trên Web App để test ngay!")
    print("=" * 60)

if __name__ == "__main__":
    main()
