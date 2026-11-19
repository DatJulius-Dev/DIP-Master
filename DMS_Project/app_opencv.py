"""
DESKTOP HUD ADAS & MULTI-MODE COMPUTER VISION APPLICATION (OPENCV RUNTIME)
Ứng dụng tương tác thời gian thực tích hợp đầy đủ 6 phân hệ trên nền OpenCV tốc độ cao (30 - 60 FPS):
- Phím '1': Phân hệ 1 - Giám sát An toàn Tài xế (DMS EAR, MAR, PERCLOS, solvePnP 3D, Đồ thị sóng EAR)
- Phím '2': Phân hệ 2 - Chuẩn hóa Khuôn mặt Affine 256x256 & Xác thực Liveness 3 bước
- Phím '3': Phân hệ 3 - Thử kính Thực tế ảo 3D (6-Point Perspective Homography AR) -> Phím 'k' đổi mẫu kính
- Phím '4': Phân hệ 4 - Trang điểm Kỹ thuật số (CIELAB-HLS Lip & 3D Blush) -> Phím 's' đổi màu son, 'b' bật/tắt má hồng
- Phím '5': Phân hệ 5 - Theo dõi Ánh mắt & Tương tác Mống mắt (Iris Gaze Tracking & Virtual Click)
- Phím '6': Phân hệ 6 - Biến hình & Hoán đổi Khuôn mặt (Delaunay Triangulation Face Morphing / Swapping) -> Phím 'f' đổi mẫu, 'g' lưới Delaunay
Tuân thủ chặt chẽ tiêu chuẩn kỹ thuật: không sử dụng ký tự icon hay emoji.
"""

import os
import sys
import time
import argparse
import collections

# Chặn cảnh báo rác
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_SEVERITY"] = "fatal"
os.environ["ABSL_LOG_LEVEL"] = "fatal"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.FATAL)
    absl.logging.set_stderrthreshold(absl.logging.FATAL)
except Exception:
    pass

for _mod in ['tensorflow', 'tensorflow.tools', 'tensorflow.tools.docs']:
    if _mod not in sys.modules:
        sys.modules[_mod] = None

import cv2
import numpy as np
from dms_engine import DMSEngine
from audio_manager import AudioManager
from glasses_generator import get_glasses_dict


def main():
    parser = argparse.ArgumentParser(description="DMS & 6-Mode Interactive Computer Vision Desktop Application")
    parser.add_argument("--camera", type=int, default=0, help="Chi so camera (mac dinh: 0)")
    parser.add_argument("--video", type=str, default="", help="Duong dan file video thu nghiem (tuy chon)")
    parser.add_argument("--mode", type=int, default=1, help="Che do khoi dong (1..6, mac dinh: 1)")
    args = parser.parse_args()

    print("[INFO] Khoi tao camera va Engine xu ly anh so 6 phan he...")
    engine = DMSEngine()
    audio = AudioManager()
    glasses_db = get_glasses_dict()
    glasses_names = list(glasses_db.keys())
    glasses_idx = 0

    lip_palette = [
        ("Do Ruby Co Dien", (20, 15, 235)),
        ("Hong Cherry Tuoi Tre", (150, 30, 255)),
        ("Cam Dat Thoi Thuong", (30, 110, 250)),
        ("Do Man Quyen Ru", (55, 10, 160)),
        ("Hong San Ho Ngot Ngao", (80, 80, 255))
    ]
    lip_idx = 0
    enable_blush = True

    morph_presets = engine.get_morph_preset_names()
    morph_idx = 0
    morph_mode = "swap"  # "swap" hoac "morph"
    morph_alpha = 0.65
    show_delaunay = True

    source = args.video if args.video else args.camera
    if not args.video and sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[LOI] Khong the mo nguon hinh anh: {source}")
        return

    if not args.video:
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

    max_history = 100
    ear_history = collections.deque(maxlen=max_history)
    prev_time = time.time()
    frame_count = 0
    active_mode = max(1, min(6, int(args.mode)))
    show_visuals = True

    print("[INFO] He thong DMS & Multi-Application Desktop HUD dang hoat dong.")
    print("      - Nhan phim '1' -> '6' de chuyen doi tuc thoi giua 6 Phan he ung dung.")
    print("      - Nhan 'c' de hieu chuan tu the 0 chuan (Zero-Ref & EAR ca nhan hoa).")
    print("      - Nhan '[' hoac ']' de giam/tang nguong EAR.")
    print("      - Nhan '-' hoac '=' de giam/tang nguong MAR.")
    print("      - Nhan 'k' (Mode 3) de doi mau kinh AR 3D.")
    print("      - Nhan 's' / 'b' (Mode 4) de doi mau son / bat tat phan ma hong.")
    print("      - Nhan 'f' / 't' / 'g' (Mode 6) de doi mau khuon mat / doi Swap-Morph / bat tat luoi Delaunay.")
    print("      - Nhan 'm' de bat/tat am thanh canh bao.")
    print("      - Nhan 'r' de reset bo dem.")
    print("      - Nhan 'v' de bat/tat hien thi Khung nhan dien & Luoi da giac.")
    print("      - Nhan 'q' hoac 'ESC' de thoat.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1

        if not args.video:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        curr_time = time.time()
        fps = 1.0 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time

        need_dms_hud = (active_mode == 1)
        metrics, annotated = engine.process_frame(
            frame,
            draw_box=(show_visuals and active_mode == 1),
            draw_contours=(show_visuals and active_mode == 1),
            draw_vectors=(show_visuals and active_mode == 1),
            render_hud=need_dms_hud
        )
        landmarks_2d = metrics.get("landmarks_2d")

        # =====================================================================
        # MODE 1: DMS GIAM SAT AN TOAN TAI XE
        # =====================================================================
        if active_mode == 1:
            if metrics["is_yawning"]:
                audio.trigger_alert("yawn")
            elif metrics["is_drowsy"]:
                audio.trigger_alert("drowsy")
            elif metrics["is_distracted"]:
                audio.trigger_alert("distracted")

            ear_history.append(metrics["ear"])

            hud_overlay = annotated.copy()
            cv2.rectangle(hud_overlay, (15, 15), (415, 272), (15, 20, 30), -1)
            cv2.addWeighted(hud_overlay, 0.75, annotated, 0.25, 0, annotated)
            cv2.rectangle(annotated, (15, 15), (415, 272), (0, 242, 254), 1)

            cv2.putText(annotated, "MODE 1: DMS TELEMETRY [PHIM 1-6 DOI MODE]", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 242, 254), 2)
            cv2.line(annotated, (25, 46), (400, 46), (100, 100, 100), 1)

            cv2.putText(annotated, f"FPS: {fps:.1f}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(annotated, f"EAR: {metrics['ear']:.3f} (Nguong: {engine.ear_thresh:.2f})", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(annotated, f"MAR: {metrics['mar']:.3f} (Nguong: {engine.mar_thresh:.2f})", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(annotated, f"PERCLOS: {metrics['perclos']:.1f}% (Nguong: {engine.perclos_thresh:.0f}%)", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(annotated, f"Pitch: {metrics['pitch']:+.1f} deg", (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
            cv2.putText(annotated, f"Yaw:   {metrics['yaw']:+.1f} deg", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
            cv2.putText(annotated, f"Roll:  {metrics['roll']:+.1f} deg", (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

            calib_txt = "ZERO-REF: DA BAT" if metrics["is_calibrated"] else "ZERO-REF: MAC DINH"
            cv2.putText(annotated, calib_txt, (25, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            gx, gy, gw, gh = 20, 285, 390, 75
            cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), (20, 25, 35), -1)
            cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), (80, 80, 80), 1)

            thresh_y = int(gy + gh - (engine.ear_thresh / 0.45) * gh)
            thresh_y = int(np.clip(thresh_y, gy, gy + gh))
            cv2.line(annotated, (gx, thresh_y), (gx + gw, thresh_y), (0, 0, 255), 1)
            cv2.putText(annotated, f"Nguong EAR: {engine.ear_thresh:.2f}", (gx + 5, thresh_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

            for i in range(1, len(ear_history)):
                v1 = np.clip(ear_history[i - 1] / 0.45, 0, 1)
                v2 = np.clip(ear_history[i] / 0.45, 0, 1)
                pt1 = (int(gx + (i - 1) * (gw / max_history)), int(gy + gh - v1 * gh))
                pt2 = (int(gx + i * (gw / max_history)), int(gy + gh - v2 * gh))
                cv2.line(annotated, pt1, pt2, (0, 242, 254), 2, cv2.LINE_AA)

            if metrics["is_yawning"]:
                cv2.rectangle(annotated, (w // 2 - 270, 30), (w // 2 + 270, 95), (0, 140, 255), -1)
                cv2.putText(annotated, "CANH BAO: BIEU HIEN NGU GAT (NGAP)", (w // 2 - 260, 75), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
            elif metrics["is_drowsy"]:
                cv2.rectangle(annotated, (0, 0), (w, h), (0, 0, 255), 12)
                cv2.rectangle(annotated, (w // 2 - 280, 30), (w // 2 + 280, 95), (0, 0, 255), -1)
                cv2.putText(annotated, "! NGUY HIEM: TAI XE NGU GAT !", (w // 2 - 260, 75), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            elif metrics["is_distracted"]:
                cv2.rectangle(annotated, (w // 2 - 310, 30), (w // 2 + 310, 95), (0, 69, 255), -1)
                cv2.putText(annotated, f"MAT TAP TRUNG: {metrics['distraction_type']}", (w // 2 - 280, 75), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
            else:
                cv2.putText(annotated, "TRANG THAI: TINH TAO", (w - 270, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # =====================================================================
        # MODE 2: FACE ALIGNMENT (AFFINE 256x256) & LIVENESS
        # =====================================================================
        elif active_mode == 2:
            annotated = frame.copy()
            cv2.putText(annotated, "MODE 2: FACE ALIGNMENT (AFFINE 256x256)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
            if landmarks_2d:
                pt_l = landmarks_2d[33]
                pt_r = landmarks_2d[263]
                cv2.line(annotated, pt_l, pt_r, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(annotated, pt_l, 5, (0, 255, 0), -1)
                cv2.circle(annotated, pt_r, 5, (0, 255, 0), -1)
                aligned_crop, angle = engine.align_face(frame, landmarks_2d, 180, 180)
                if aligned_crop is not None and aligned_crop.size > 0:
                    ah, aw = aligned_crop.shape[:2]
                    x_pos = max(0, w - aw - 16)
                    y_pos = 16
                    annotated[y_pos:y_pos+ah, x_pos:x_pos+aw] = aligned_crop
                    cv2.rectangle(annotated, (x_pos, y_pos), (x_pos + aw, y_pos + ah), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Goc xoay: {angle:+.1f} deg", (x_pos + 10, y_pos + ah + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)

        # =====================================================================
        # MODE 3: THU KINH THUC TE AO 3D (HOMOGRAPHY 6 DIEM)
        # =====================================================================
        elif active_mode == 3:
            annotated = frame.copy()
            g_name = glasses_names[glasses_idx % len(glasses_names)]
            if landmarks_2d:
                annotated = engine.overlay_glasses_homography(annotated, landmarks_2d, glasses_db[g_name])
            cv2.putText(annotated, f"MODE 3: KINH 3D - {g_name} ('k' doi kinh)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 200, 0), 2, cv2.LINE_AA)

        # =====================================================================
        # MODE 4: TRANG DIEM KY THUAT SO (CIELAB-HLS LIP & 3D BLUSH)
        # =====================================================================
        elif active_mode == 4:
            annotated = frame.copy()
            lip_name, lip_bgr = lip_palette[lip_idx % len(lip_palette)]
            blush_bgr = (160, 110, 255) if enable_blush else None
            if landmarks_2d:
                annotated = engine.apply_makeup(
                    annotated,
                    landmarks_2d,
                    lip_bgr,
                    lip_alpha=0.60,
                    blush_bgr=blush_bgr,
                    blush_alpha=0.35,
                    color_space="HYBRID"
                )
            cv2.putText(annotated, f"MODE 4: SON [{lip_name}] ('s' doi son, 'b' ma hong)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 50, 255), 2, cv2.LINE_AA)

        # =====================================================================
        # MODE 5: GAZE TRACKING & IRIS INTERACTION
        # =====================================================================
        elif active_mode == 5:
            annotated = frame.copy()
            cv2.putText(annotated, "MODE 5: GAZE TRACKING & IRIS INTERACTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 242, 254), 2, cv2.LINE_AA)
            if landmarks_2d:
                gaze_data = engine.analyze_gaze(landmarks_2d)
                annotated = engine.draw_iris_landmarks(annotated, landmarks_2d, gaze_data)
                cv2.putText(
                    annotated,
                    f"Huong nhin: {gaze_data['h_direction']} & {gaze_data['v_direction']}",
                    (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # =====================================================================
        # MODE 6: FACE MORPHING & FACE SWAPPING (DELAUNAY TRIANGULATION)
        # =====================================================================
        elif active_mode == 6:
            annotated = frame.copy()
            preset_name = morph_presets[morph_idx % len(morph_presets)]
            src_img = engine.get_morph_preset_image(preset_name)
            if landmarks_2d and src_img is not None:
                annotated, info_m = engine.apply_face_morph_or_swap(
                    annotated,
                    landmarks_2d,
                    source_bgr=src_img,
                    mode=morph_mode,
                    alpha=morph_alpha,
                    blend_mode="reinhard",
                    draw_delaunay=show_delaunay
                )
            cv2.putText(
                annotated,
                f"MODE 6: FACE {morph_mode.upper()} [{preset_name}] ('f' doi mau, 't' Swap/Morph, 'g' luoi)",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 180),
                2,
                cv2.LINE_AA
            )

        if fps < 15.0 and frame_count > 10:
            annotated = engine.draw_lag_loading_hud(annotated, fps=fps)
        cv2.imshow("DMS & Computer Vision Multi-Mode Application [TDTU - DIP 2026]", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')):
            active_mode = int(chr(key))
            print(f"[INFO] Chuyen sang Phan he {active_mode}")
        elif key == ord('k'):
            glasses_idx = (glasses_idx + 1) % len(glasses_names)
            print(f"[INFO] Doi mau kinh AR: {glasses_names[glasses_idx]}")
        elif key == ord('s'):
            lip_idx = (lip_idx + 1) % len(lip_palette)
            print(f"[INFO] Doi mau son: {lip_palette[lip_idx][0]}")
        elif key == ord('b'):
            enable_blush = not enable_blush
            print(f"[INFO] Phan ma hong: {'BAT' if enable_blush else 'TAT'}")
        elif key == ord('f'):
            morph_idx = (morph_idx + 1) % len(morph_presets)
            print(f"[INFO] Doi mau khuon mat Mode 6: {morph_presets[morph_idx]}")
        elif key == ord('t'):
            morph_mode = "morph" if morph_mode == "swap" else "swap"
            print(f"[INFO] Che do Mode 6: {morph_mode.upper()}")
        elif key == ord('g'):
            show_delaunay = not show_delaunay
            print(f"[INFO] Hien thi luoi tam giac Delaunay: {'BAT' if show_delaunay else 'TAT'}")
        elif key == ord('c'):
            engine.calibrate_neutral_pose(metrics["pitch_raw"], metrics["yaw_raw"], metrics["roll_raw"], metrics.get("ear", 0.28))
            print("[INFO] Da hieu chuan goc moc 0 thanh cong.")
        elif key == ord('r'):
            engine.reset_states()
            engine.reset_calibration()
            print("[INFO] Da reset toan bo bo dem.")
        elif key == ord('m'):
            audio.set_enabled(not audio.is_enabled)
            print(f"[INFO] Am thanh canh bao: {'BAT' if audio.is_enabled else 'TAT'}")
        elif key == ord('['):
            engine.ear_thresh = max(0.12, engine.ear_thresh - 0.01)
            print(f"[INFO] Nguong EAR giam xuong: {engine.ear_thresh:.2f}")
        elif key == ord(']'):
            engine.ear_thresh = min(0.35, engine.ear_thresh + 0.01)
            print(f"[INFO] Nguong EAR tang len: {engine.ear_thresh:.2f}")
        elif key == ord('-'):
            engine.mar_thresh = max(0.40, engine.mar_thresh - 0.02)
            print(f"[INFO] Nguong MAR giam xuong: {engine.mar_thresh:.2f}")
        elif key == ord('v'):
            show_visuals = not show_visuals
            print(f"[INFO] Hien thi truc quan AI: {'BAT' if show_visuals else 'TAT'}")
        elif key == ord('='):
            engine.mar_thresh = min(0.90, engine.mar_thresh + 0.02)
            print(f"[INFO] Nguong MAR tang len: {engine.mar_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()