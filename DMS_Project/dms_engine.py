"""
CORE DMS & FACE LANDMARK ANALYTICS ENGINE
Xử lý toàn bộ mô hình giải tích hình học, ước lượng tư thế 3D, Face Alignment,
Homography Glasses Try-On, Makeup HLS/LAB và Gaze Vector Tracking.
Hỗ trợ tương thích đa nền tảng: cả MediaPipe Legacy Solutions API (Python 3.9 - 3.11)
lẫn MediaPipe Tasks API FaceLandmarker mới (Python 3.12 - 3.14+).
Phiên bản V17 bổ sung 6 lớp phòng vệ thực tế (Real-World DMS Edge-Case Defenses):
1. Tiền xử lý CLAHE trên kênh L* (CIELAB) tự động kích hoạt khi thiếu sáng / lái xe ban đêm.
2. Bù trừ cơ vòng mắt khi cười tươi (Smile-Squint MWR Compensation) chống báo giả ngủ gật.
3. Bù góc chiếu phối cảnh 3D theo cos(Pitch) chống méo EAR khi hơi cúi đầu nhìn đồng hồ.
4. Khóa vị trí ghế ngồi không gian (Spatial Seat Locking) trong Cabin 2 người: khi tài xế khuất mặt,
   không bao giờ gán nhầm phụ xe (ngồi bên phải) thành tài xế.
5. Hiệu chuẩn cá nhân hóa kép (Zero-Pose + EAR Baseline) cho người mắt một mí / mắt nhỏ tự nhiên.
6. Đánh giá phương sai toán tử đạo hàm bậc hai Laplacian (Texture Sharpness) hỗ trợ chống giả mạo.
Tuân thủ chặt chẽ tiêu chuẩn kỹ thuật: không sử dụng ký tự icon hay emoji.
"""

import os
import sys
import time
import math
import threading
import collections
import numpy as np
import cv2
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")

# Cấu hình môi trường triệt tiêu log C++ của TensorFlow, MediaPipe, OpenCV
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

@contextmanager
def suppress_c_stderr():
    """
    Chặn triệt để toàn bộ log/cảnh báo ở tầng C++ (như XNNPACK, inference_feedback_manager, absl STDERR)
    ghi trực tiếp vào file descriptor 2 (C stderr).
    """
    saved_fd2 = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd2 = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
    except Exception:
        saved_fd2 = None
    try:
        yield
    finally:
        if saved_fd2 is not None:
            try:
                os.dup2(saved_fd2, 2)
                os.close(saved_fd2)
            except Exception:
                pass

with suppress_c_stderr():
    import mediapipe as mp


# -----------------------------------------------------------------
# BỘ CHUYỂN TIẾP TƯƠNG THÍCH ĐA PHIÊN BẢN (UNIVERSAL MESH ADAPTER)
# Tự động chọn Solutions API (Python 3.9 - 3.11) hoặc Tasks API (Python 3.12 - 3.14+)
# -----------------------------------------------------------------
class _FaceLandmarksWrapper:
    def __init__(self, landmarks_list):
        self.landmark = landmarks_list


class _FaceMeshResultWrapper:
    def __init__(self, list_of_landmarks):
        self.multi_face_landmarks = (
            [_FaceLandmarksWrapper(lm_list) for lm_list in list_of_landmarks]
            if list_of_landmarks
            else None
        )


class _FaceLandmarkerTasksAdapter:
    def __init__(self, model_path=None, num_faces=2):
        import urllib.request
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "face_landmarker.task")

            if not os.path.isfile(model_path) or os.path.getsize(model_path) < 10000:
                print("[INFO] Phát hiện MediaPipe Tasks API (Python 3.12+). Đang tải mô hình face_landmarker.task (3.6 MB)...")
                model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    print("[INFO] Tải mô hình face_landmarker.task thành công.")
                except Exception as e:
                    print(f"[CẢNH BÁO] Không thể tự động tải mô hình: {e}")
                    print(f"[HƯỚNG DẪN] Vui lòng tải file từ link sau và lưu vào {model_path}:\n{model_url}")

        base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
        options = mp_tasks_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = mp_tasks_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        detection_result = self.detector.detect(mp_image)
        if detection_result and detection_result.face_landmarks:
            return _FaceMeshResultWrapper(detection_result.face_landmarks)
        return _FaceMeshResultWrapper([])


def create_face_mesh_detector(max_faces=2):
    """Khởi tạo Face Mesh thích ứng tự động theo phiên bản MediaPipe / Python (hỗ trợ Cabin 2 người)."""
    with suppress_c_stderr():
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                return mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                pass

        try:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
            return mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            pass

        return _FaceLandmarkerTasksAdapter(num_faces=max_faces)


def apply_camera_scale(frame, scale=1.0, pan_x=0.0):
    """
    Phóng to (Digital Zoom > 1.0) hoặc Thu nhỏ toàn cảnh (Scale Out < 1.0 với viền đệm letterbox).
    Hỗ trợ dịch chuyển góc nhìn ngang (Pan Left / Right) để tập trung vào từng ghế ngồi trong cabin.
    """
    if frame is None:
        return None
    if abs(scale - 1.0) < 1e-3 and abs(pan_x) < 1e-3:
        return frame

    h, w = frame.shape[:2]
    if scale > 1.0:
        crop_w = int(w / scale)
        crop_h = int(h / scale)
        max_shift_x = (w - crop_w) // 2
        shift_x = int(pan_x * max_shift_x)
        cx = w // 2 + shift_x
        cy = h // 2
        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        cropped = frame[y1:y1+crop_h, x1:x1+crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        new_w = max(10, int(w * scale))
        new_h = max(10, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        max_shift_x = (w - new_w) // 2
        shift_x = int(pan_x * max_shift_x)
        x_offset = max(0, min(w - new_w, (w - new_w) // 2 + shift_x))
        y_offset = (h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas


class PersonTracker:
    def __init__(self, person_id=0, role="TÀI XẾ"):
        self.person_id = person_id
        self.role = role
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.face_lost_counter = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.eye_open_recovery_counter = 0
        self.roll_tilt_counter = 0
        self.last_centroid = None
        self.perclos_queue = collections.deque(maxlen=90)
        self.pitch_history = collections.deque(maxlen=8)
        self.smoothed_landmarks = None
        self.post_yawn_cooldown = 0

    def reset(self):
        self.eye_closed_counter = 0
        self.eye_open_recovery_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.roll_tilt_counter = 0
        self.face_lost_counter = 0
        self.last_centroid = None
        self.is_drowsy = False
        self.is_yawning = False
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.perclos_queue.clear()
        self.pitch_history.clear()
        self.smoothed_landmarks = None
        self.post_yawn_cooldown = 0


class DMSEngine:
    def __init__(
        self,
        ear_thresh=0.21,
        ear_consec_frames=10,
        mar_thresh=0.50,
        mar_consec_frames=5,
        pitch_down_thresh=14.0,
        pitch_up_thresh=20.0,
        yaw_turn_thresh=20.0,
        perclos_window=90,
        perclos_thresh=25.0,
        ema_alpha=0.65
    ):
        self.ear_thresh = float(ear_thresh)
        self.ear_consec_frames = int(ear_consec_frames)
        self.mar_thresh = float(mar_thresh)
        self.mar_consec_frames = int(mar_consec_frames)
        self.pitch_down_thresh = float(pitch_down_thresh)
        self.pitch_up_thresh = float(pitch_up_thresh)
        self.yaw_turn_thresh = float(yaw_turn_thresh)
        self.roll_tilt_thresh = 22.0
        self.perclos_thresh = float(perclos_thresh)
        self.ema_alpha = float(ema_alpha)

        # Công tắc tiền xử lý CLAHE ban đêm / chống chói ngược nắng (Dual-Mode Illumination)
        self.enable_clahe_night = False
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self._clahe_glare = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        self._gamma_glare_lut = np.array(
            [((i / 255.0) ** 1.38) * 255.0 for i in range(256)], dtype=np.uint8
        )

        # Khóa bảo vệ luồng khi Streamlit Rerun
        self._infer_lock = threading.Lock()

        # Bộ đếm trạng thái
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.face_lost_counter = 0

        # Cờ trạng thái cảnh báo
        self.is_drowsy = False
        self.is_yawning = False
        self.is_distracted = False
        self.distraction_type = "NORMAL"

        # Cửa sổ trượt tính PERCLOS
        self.perclos_queue = collections.deque(maxlen=int(perclos_window))

        # Bộ lọc làm mượt mốc điểm EMA (Exponential Moving Average) chống rung nhiễu
        self._smoothed_landmarks = None

        # Hiệu chuẩn góc chuẩn (Zero-Reference Calibration) & EAR cá nhân hóa
        self.zero_pitch = 0.0
        self.zero_yaw = 0.0
        self.zero_roll = 0.0
        self.baseline_ear = 0.0
        self.is_calibrated = False

        # Mô hình nhân học 3D Canonical Face Model (6 điểm chốt)
        self.model_points_68 = np.array([
            (0.0, 0.0, 0.0),             # 1: Đỉnh mũi (Nose tip)
            (0.0, 330.0, 65.0),          # 199: Cằm (Chin)
            (-225.0, -170.0, 135.0),     # 33: Khóe mắt ngoài trái ảnh
            (225.0, -170.0, 135.0),      # 263: Khóe mắt ngoài phải ảnh
            (-150.0, 150.0, 125.0),      # 61: Khóe miệng trái ảnh
            (150.0, 150.0, 125.0)        # 291: Khóe miệng phải ảnh
        ], dtype=np.float64)

        # Bộ theo dõi độc lập cho từng người (Hỗ trợ Cabin 2 người: Tài xế & Phụ xe)
        self.trackers = [
            PersonTracker(0, "TÀI XẾ"),
            PersonTracker(1, "PHỤ XE")
        ]

        # Khởi tạo Face Mesh detector (hỗ trợ tối đa 2 người)
        self.face_mesh = create_face_mesh_detector(max_faces=2)

    @staticmethod
    def _normalize_angle_diff(angle, ref_angle=0.0):
        """Chuẩn hóa hiệu hai góc Euler về đoạn [-180.0, +180.0] chống nhảy vọt qua ranh giới +-180 độ."""
        return (float(angle) - float(ref_angle) + 180.0) % 360.0 - 180.0

    def calibrate_neutral_pose(self, pitch, yaw, roll, current_ear=None):
        """
        Ghi nhận góc nhìn hiện tại làm gốc tọa độ 0 chuẩn, đồng thời cá nhân hóa ngưỡng EAR
        theo kích thước mắt thực tế của tài xế (hỗ trợ mắt một mí / mắt nhỏ tự nhiên).
        """
        self.zero_pitch = float(pitch)
        self.zero_yaw = float(yaw)
        self.zero_roll = float(roll)
        if current_ear is not None and float(current_ear) > 0.16:
            self.baseline_ear = float(current_ear)
            self.ear_thresh = round(max(0.15, min(0.26, self.baseline_ear * 0.72)), 2)
        self.is_calibrated = True

    def reset_calibration(self):
        """Xóa hiệu chuẩn về mặc định."""
        self.zero_pitch = 0.0
        self.zero_yaw = 0.0
        self.zero_roll = 0.0
        self.baseline_ear = 0.0
        self.is_calibrated = False

    def reset_states(self):
        """Reset toàn bộ biến đếm và cờ cảnh báo."""
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.face_lost_counter = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.perclos_queue.clear()
        self._smoothed_landmarks = None
        self.post_yawn_cooldown = 0
        for trk in self.trackers:
            trk.reset()

    def preprocess_low_light_clahe(self, frame_bgr):
        """
        Chuẩn hóa quang độ kép (Dual-Mode Illumination Normalization) trên không gian màu CIELAB:
        - Chế độ 1 (Low-Light / Night-Vision khi mean_luma < 72): Cân bằng biểu đồ xám cục bộ CLAHE trên kênh L*.
        - Chế độ 2 (Anti-Glare / Highlight Recovery khi ngược nắng chói lóa mean_luma > 192 hoặc cháy sáng cục bộ):
          Nén Gamma phi tuyến (gamma = 1.38) kết hợp CLAHE dịu để khôi phục chi tiết viền mí mắt bị lóa nắng.
        """
        mean_luma = float(np.mean(frame_bgr))
        glare_ratio = float(np.mean(frame_bgr > 242))
        is_low_light = self.enable_clahe_night or (mean_luma < 72.0)
        is_glare_overexposed = (mean_luma > 192.0) or (glare_ratio > 0.16 and mean_luma > 155.0)

        if not is_low_light and not is_glare_overexposed:
            return frame_bgr, mean_luma, False

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        if is_glare_overexposed and not (mean_luma < 72.0):
            l_compressed = cv2.LUT(l_ch, self._gamma_glare_lut)
            l_eq = self._clahe_glare.apply(l_compressed)
        else:
            l_eq = self._clahe.apply(l_ch)
        enhanced_bgr = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return enhanced_bgr, mean_luma, True

    @staticmethod
    def compute_face_laplacian_sharpness(frame_bgr, landmarks_2d):
        """
        Tính phương sai toán tử đạo hàm bậc hai Laplacian trên vùng ROI khuôn mặt
        phục vụ đánh giá độ sắc nét bề mặt thực (Texture Sharpness) hỗ trợ chống giả mạo.
        """
        if frame_bgr is None or not landmarks_2d:
            return 0.0
        h, w = frame_bgr.shape[:2]
        pts = np.array(landmarks_2d, dtype=np.int32)
        x1 = max(0, int(np.min(pts[:, 0])))
        y1 = max(0, int(np.min(pts[:, 1])))
        x2 = min(w, int(np.max(pts[:, 0])))
        y2 = min(h, int(np.max(pts[:, 1])))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return 0.0
        roi_gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
        return lap_var

    @staticmethod
    def _euclidean_dist(pt1, pt2):
        dx = float(pt1[0]) - float(pt2[0])
        dy = float(pt1[1]) - float(pt2[1])
        return math.hypot(dx, dy)

    def calculate_ear(self, landmarks, eye_indices):
        """
        Công thức Eye Aspect Ratio (Soukupova & Cech, 2016):
        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
        """
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        dist_v1 = self._euclidean_dist(p2, p6)
        dist_v2 = self._euclidean_dist(p3, p5)
        dist_h = self._euclidean_dist(p1, p4)

        if dist_h < 1e-6:
            return 0.0
        return (dist_v1 + dist_v2) / (2.0 * dist_h)

    def calculate_mar(self, landmarks):
        """
        Công thức Mouth Aspect Ratio (MAR):
        Đo tỷ lệ giãn cách chiều dọc trên bề ngang khóe miệng.
        """
        p_top1, p_bot1 = landmarks[13], landmarks[14]
        p_top2, p_bot2 = landmarks[81], landmarks[178]
        p_top3, p_bot3 = landmarks[311], landmarks[402]
        p_left, p_right = landmarks[61], landmarks[291]

        v1 = self._euclidean_dist(p_top1, p_bot1)
        v2 = self._euclidean_dist(p_top2, p_bot2)
        v3 = self._euclidean_dist(p_top3, p_bot3)
        h = self._euclidean_dist(p_left, p_right)

        if h < 1e-6:
            return 0.0
        return (v1 + v2 + v3) / (3.0 * h)

    def is_smiling_squint(self, landmarks, mar):
        """
        Phát hiện trạng thái cười tươi (Smile-Squint Detection):
        Khi cười tươi, bề ngang khóe miệng |p61 - p291| giãn rộng so với bề ngang gò má |p234 - p454|
        nhưng miệng không há to như khi ngáp (MAR < 0.42). Giúp chống báo giả ngủ gật khi híp mắt cười.
        """
        mouth_w = self._euclidean_dist(landmarks[61], landmarks[291])
        face_w = max(1.0, self._euclidean_dist(landmarks[234], landmarks[454]))
        mwr = mouth_w / face_w
        return (mwr > 0.44) and (mar < 0.42)

    def estimate_head_pose(self, landmarks_2d, img_w, img_h):
        """
        Giải bài toán Perspective-n-Point (solvePnP) và biến đổi Rodrigues để phân rã 3 góc Euler (Pitch, Yaw, Roll).
        """
        idx_list = [1, 199, 33, 263, 61, 291]
        image_points = np.array([landmarks_2d[i] for i in idx_list], dtype=np.float64)

        focal_length = float(img_w)
        cam_center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, cam_center[0]],
            [0.0, focal_length, cam_center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        rvec_init = np.zeros((3, 1), dtype=np.float64)
        tvec_init = np.array([[0.0], [0.0], [1500.0]], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points_68,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, None, None, camera_matrix

        if tvec[2, 0] < 0:
            rvec, tvec = -rvec, -tvec

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0.0

        pitch = self._normalize_angle_diff(-math.degrees(x), 4.0)
        yaw = self._normalize_angle_diff(math.degrees(y), 0.0)
        roll = self._normalize_angle_diff(math.degrees(z), 0.0)

        return pitch, yaw, roll, rvec, tvec, camera_matrix

    def process_frame(
        self,
        frame_bgr,
        is_static_image=False,
        cabin_mode="auto",
        draw_box=True,
        draw_contours=True,
        draw_vectors=True,
        render_hud=True
    ):
        t_start_process = time.perf_counter()
        """
        Xử lý khung hình Camera cho Phân hệ 1 (DMS Cốt lõi) và trích xuất mốc mặt cho các phân hệ.
        Tích hợp đầy đủ các lớp phòng vệ thực tế: CLAHE Night-Vision, Smile-Squint Compensation,
        Cosine Pitch Compensation, Micro-Nodding Detection và Dual-Cabin Spatial Seat Locking.
        """
        h, w = frame_bgr.shape[:2]
        proc_bgr, mean_luma, clahe_active = self.preprocess_low_light_clahe(frame_bgr)
        frame_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        with self._infer_lock:
            with suppress_c_stderr():
                results = self.face_mesh.process(frame_rgb)
        annotated = proc_bgr.copy() if render_hud else frame_bgr

        metrics = {
            "face_detected": False,
            "person_count": 0,
            "is_dual_mode": False,
            "persons": [],
            "driver": None,
            "codriver": None,
            "dual_alert_type": "none",
            "dual_alert_text": "",
            "ear": 0.0,
            "ear_left": 0.0,
            "ear_right": 0.0,
            "mar": 0.0,
            "perclos": 0.0,
            "pitch_raw": 0.0,
            "yaw_raw": 0.0,
            "roll_raw": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "is_drowsy": False,
            "is_yawning": False,
            "is_distracted": False,
            "is_smiling": False,
            "clahe_active": clahe_active,
            "mean_luma": mean_luma,
            "distraction_type": "NORMAL",
            "is_calibrated": self.is_calibrated,
            "landmarks_2d": None
        }

        if not results or not results.multi_face_landmarks:
            self._smoothed_landmarks = None
            self.face_lost_counter += 1
            for trk in self.trackers:
                trk.face_lost_counter += 1
                trk.smoothed_landmarks = None

            if self.face_lost_counter > 30:
                self.is_distracted = True
                self.distraction_type = "FACE LOST / OCCLUDED"
            else:
                self.is_distracted = False
                self.distraction_type = "SEARCHING FACE"

            metrics["is_distracted"] = self.is_distracted
            metrics["distraction_type"] = self.distraction_type
            t_elapsed = (time.perf_counter() - t_start_process) * 1000.0
            metrics["proc_latency_ms"] = t_elapsed
            metrics["is_lagging"] = (t_elapsed > 65.0)
            if render_hud and metrics["is_lagging"]:
                annotated = self.draw_lag_loading_hud(annotated, latency_ms=t_elapsed)
            return metrics, annotated

        self.face_lost_counter = 0

        raw_faces = results.multi_face_landmarks
        if cabin_mode == "single":
            faces = [raw_faces[0]]
        else:
            faces = raw_faces[:2]

        # Sắp xếp khuôn mặt theo tọa độ X trung bình từ trái sang phải
        faces = sorted(faces, key=lambda f: float(np.mean([lm.x for lm in f.landmark])))

        # Khóa không gian & tâm quỹ đạo ghế ngồi (Spatial-Temporal Seat Locking) cho Cabin 2 người:
        # Nếu đang chọn cố định "dual" mà chỉ thấy 1 khuôn mặt ở nửa bên phải (X > 0.56) hoặc gần tâm Phụ xe cũ hơn Tài xế,
        # đánh dấu Tài xế bị khuất mặt thay vì gán nhầm Phụ xe thành Tài xế!
        single_codriver_only = False
        if cabin_mode == "dual" and len(faces) == 1:
            mean_x_norm = float(np.mean([lm.x for lm in faces[0].landmark]))
            mean_y_norm = float(np.mean([lm.y for lm in faces[0].landmark]))
            if mean_x_norm > 0.56:
                single_codriver_only = True
            elif (
                mean_x_norm > 0.48
                and self.trackers[0].last_centroid is not None
                and self.trackers[1].last_centroid is not None
                and self.trackers[1].face_lost_counter < 10
            ):
                d_to_driver = math.hypot(mean_x_norm - self.trackers[0].last_centroid[0], mean_y_norm - self.trackers[0].last_centroid[1])
                d_to_codriver = math.hypot(mean_x_norm - self.trackers[1].last_centroid[0], mean_y_norm - self.trackers[1].last_centroid[1])
                if d_to_codriver + 0.04 < d_to_driver:
                    single_codriver_only = True

        is_dual = (len(faces) >= 2) or single_codriver_only

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        persons_metrics = []

        for idx, face_landmarks in enumerate(faces):
            actual_slot = 1 if single_codriver_only else idx
            role_name = "TÀI XẾ" if actual_slot == 0 else ("PHỤ XE" if is_dual else f"NGƯỜI #{idx+1}")
            tracker = self.trackers[actual_slot] if actual_slot < len(self.trackers) else self.trackers[-1]
            tracker.face_lost_counter = 0
            tracker.last_centroid = (
                float(np.mean([lm.x for lm in face_landmarks.landmark])),
                float(np.mean([lm.y for lm in face_landmarks.landmark]))
            )

            raw_pts = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.float32)

            if tracker.smoothed_landmarks is None or tracker.smoothed_landmarks.shape != raw_pts.shape:
                tracker.smoothed_landmarks = raw_pts.copy()
            else:
                motion_mag = float(np.mean(np.linalg.norm(raw_pts - tracker.smoothed_landmarks, axis=1)))
                adaptive_alpha = min(0.92, self.ema_alpha + min(0.25, motion_mag / 20.0))
                tracker.smoothed_landmarks = adaptive_alpha * raw_pts + (1.0 - adaptive_alpha) * tracker.smoothed_landmarks

            raw_landmarks_2d = [(int(round(pt[0])), int(round(pt[1]))) for pt in raw_pts]
            landmarks_2d = [(int(round(pt[0])), int(round(pt[1]))) for pt in tracker.smoothed_landmarks]

            # 1. Tính EAR mắt trái & mắt phải
            ear_l = self.calculate_ear(raw_pts, left_eye_indices)
            ear_r = self.calculate_ear(raw_pts, right_eye_indices)
            avg_ear_raw = (ear_l + ear_r) / 2.0

            # 2. Tính MAR miệng & phát hiện cười tươi (Smile-Squint)
            mar = self.calculate_mar(tracker.smoothed_landmarks)
            smiling = self.is_smiling_squint(tracker.smoothed_landmarks, mar)

            # 3. Ước lượng Head Pose 3D với chuẩn hóa góc
            p_raw, y_raw, r_raw, rvec, tvec, cam_mat = self.estimate_head_pose(tracker.smoothed_landmarks, w, h)
            p_rel = self._normalize_angle_diff(p_raw, self.zero_pitch if actual_slot == 0 else 0.0)
            y_rel = self._normalize_angle_diff(y_raw, self.zero_yaw if actual_slot == 0 else 0.0)
            r_rel = self._normalize_angle_diff(r_raw, self.zero_roll if actual_slot == 0 else 0.0)

            # Bù góc chiếu phối cảnh 3D theo cos(Pitch) để khi hơi cúi nhẹ nhìn đồng hồ không bị sụt giả EAR
            pitch_rad = math.radians(min(30.0, abs(p_rel)))
            cos_comp = max(0.86, math.cos(pitch_rad))
            avg_ear = avg_ear_raw / cos_comp

            # Nhận diện pha ngáp (Yawn Phase): khi miệng mở to hoặc đang trong quá trình ngáp
            is_in_yawn_phase = (mar > self.mar_thresh * 0.82) or (tracker.yawn_counter > 0) or tracker.is_yawning

            # Bù trừ nhắm/nheo mắt tự nhiên do phản xạ cơ mặt khi ngáp (Yawn-Squint Reflex)
            # hoặc cười tươi (Smile-Squint). Khi ngáp to, cơ mặt co kéo khiến mắt tự nhiên nheo lại,
            # KHÔNG ĐƯỢC tính vào nhắm mắt do ngủ gật!
            effective_ear_thresh = max(0.12, self.ear_thresh - (0.025 if smiling else 0.0))
            if is_in_yawn_phase:
                effective_ear_thresh = min(effective_ear_thresh, 0.12)

            # Theo dõi vận tốc gục đầu đột ngột (Micro-Nodding) hỗ trợ khi đeo kính râm/lóa kính
            tracker.pitch_history.append(p_rel)
            nod_drop = (tracker.pitch_history[0] - p_rel) if len(tracker.pitch_history) >= 5 else 0.0
            is_head_nodding_fast = (nod_drop > 11.5) and (p_rel < -10.0)

            # 4. Sanity Gate chống góc quay mặt lớn
            is_head_turned = abs(y_rel) > self.yaw_turn_thresh

            # 5. Tính PERCLOS & Sanity Filter
            # Nếu đang ngáp hoặc trong thời gian giảm xóc hậu-ngáp (post_yawn_cooldown), mắt nheo lại là do cơ học,
            # TUYỆT ĐỐI không tính vào PERCLOS để tránh tích lũy làm kích hoạt còi ngủ gật nhầm!
            is_closed_instant = (avg_ear < effective_ear_thresh) and (not is_head_turned) and (not is_in_yawn_phase) and (tracker.post_yawn_cooldown == 0)

            perclos_entry = 0.0 if (is_in_yawn_phase or tracker.post_yawn_cooldown > 0) else (1.0 if is_closed_instant else 0.0)
            tracker.perclos_queue.append(perclos_entry)
            perclos = (sum(tracker.perclos_queue) / max(1, len(tracker.perclos_queue))) * 100.0

            # 6. Ước lượng Mất tập trung & Gục nghiêng đầu sang vai (Distraction & Head Roll Tilt)
            distracted_detected = False
            distraction_reason = "NORMAL"
            is_head_tilted_roll = abs(r_rel) > self.roll_tilt_thresh

            if p_rel < -self.pitch_down_thresh:
                distracted_detected = True
                distraction_reason = "HEAD DOWN"
            elif p_rel > self.pitch_up_thresh:
                distracted_detected = True
                distraction_reason = "HEAD UP"
            elif y_rel < -self.yaw_turn_thresh:
                distracted_detected = True
                distraction_reason = "LOOKING RIGHT"
            elif y_rel > self.yaw_turn_thresh:
                distracted_detected = True
                distraction_reason = "LOOKING LEFT"
            elif is_head_tilted_roll:
                distracted_detected = True
                distraction_reason = "HEAD TILT (ROLL)"

            # Phát hiện ngủ gật nghiêng đầu qua vai (mắt lim dim kèm đầu nghiêng sâu > 22 độ)
            is_sideways_nod = is_head_tilted_roll and (avg_ear < effective_ear_thresh + 0.025)

            # 7. Logic phân loại Trạng thái: Tách bạch tuyệt đối giữa NGÁP (Cảnh báo biểu hiện) và NGỦ GẬT (Nguy hiểm)
            if is_static_image:
                p_is_yawning = (mar > self.mar_thresh) and (not is_head_turned)
                # Đang ngáp thì ưu tiên cảnh báo ngáp / biểu hiện buồn ngủ, KHÔNG báo ngủ gật chồng chéo!
                p_is_drowsy = (is_closed_instant or is_sideways_nod) and (not p_is_yawning)
                p_is_distracted = distracted_detected and (not p_is_yawning) and (not p_is_drowsy)
                p_distraction_type = distraction_reason if p_is_distracted else "NORMAL"
            else:
                # 7.1. Cập nhật bộ đếm ngáp (Yawning state tracking)
                was_yawning = tracker.is_yawning
                if mar > self.mar_thresh and not is_head_turned:
                    tracker.yawn_counter += 1
                    if tracker.yawn_counter >= self.mar_consec_frames:
                        tracker.is_yawning = True
                else:
                    decay_step = 3 if mar < (self.mar_thresh * 0.72) else 2
                    tracker.yawn_counter = max(0, tracker.yawn_counter - decay_step)
                    if tracker.yawn_counter == 0:
                        tracker.is_yawning = False

                # 7.2. Xử lý chuyển tiếp hậu-ngáp (Post-Yawn Transition Cooldown & Buffer Flush)
                # Khi vừa kết thúc ngáp, tài xế thường chớp mắt hoặc cơ mặt chưa giãn hết,
                # kích hoạt cooldown 20 frames và xả bớt bộ đệm PERCLOS để triệt tiêu dư chấn
                if was_yawning and not tracker.is_yawning:
                    tracker.post_yawn_cooldown = 20
                    for _ in range(min(18, len(tracker.perclos_queue))):
                        tracker.perclos_queue.append(0.0)
                    perclos = (sum(tracker.perclos_queue) / max(1, len(tracker.perclos_queue))) * 100.0

                if tracker.post_yawn_cooldown > 0:
                    tracker.post_yawn_cooldown -= 1

                # 7.3. Phân định rạch ròi giữa NGÁP (Cảnh báo biểu hiện) và NGỦ GẬT (Nguy hiểm thực sự)
                if is_in_yawn_phase or tracker.is_yawning:
                    # ĐANG TRONG PHA NGÁP: Khóa chặt cờ ngủ gật, reset bộ đếm nhắm mắt
                    tracker.is_drowsy = False
                    tracker.eye_closed_counter = 0
                    tracker.eye_open_recovery_counter = 0
                elif tracker.post_yawn_cooldown > 0:
                    # ĐANG COOLDOWN HẬU NGÁP: Không kích hoạt ngủ gật đột ngột
                    tracker.is_drowsy = False
                    tracker.eye_closed_counter = 0
                else:
                    # ĐÁNH GIÁ TRẠNG THÁI NGỦ GẬT (CHỈ KHI KHÔNG NGÁP)
                    if is_closed_instant or is_head_nodding_fast or is_sideways_nod:
                        tracker.eye_closed_counter += 1
                        tracker.eye_open_recovery_counter = 0
                    else:
                        tracker.eye_closed_counter = max(0, tracker.eye_closed_counter - 2)
                        if avg_ear > (effective_ear_thresh + 0.04):
                            tracker.eye_open_recovery_counter += 1
                        else:
                            tracker.eye_open_recovery_counter = 0

                    if tracker.eye_open_recovery_counter >= 2:
                        tracker.is_drowsy = False
                        tracker.eye_closed_counter = 0
                        for _ in range(min(4, len(tracker.perclos_queue))):
                            tracker.perclos_queue.append(0.0)
                        perclos = (sum(tracker.perclos_queue) / max(1, len(tracker.perclos_queue))) * 100.0
                    elif (tracker.eye_closed_counter >= self.ear_consec_frames) or (
                        len(tracker.perclos_queue) >= 30 and perclos >= self.perclos_thresh
                    ):
                        tracker.is_drowsy = True
                    else:
                        tracker.is_drowsy = False

                if distracted_detected and not tracker.is_yawning and not tracker.is_drowsy:
                    tracker.distraction_counter += 1
                    if tracker.distraction_counter >= 8:
                        tracker.is_distracted = True
                        tracker.distraction_type = distraction_reason
                else:
                    tracker.distraction_counter = max(0, tracker.distraction_counter - 2)
                    if tracker.distraction_counter == 0:
                        tracker.is_distracted = False
                        tracker.distraction_type = "NORMAL"

                p_is_yawning = tracker.is_yawning
                p_is_drowsy = tracker.is_drowsy
                p_is_distracted = tracker.is_distracted
                p_distraction_type = tracker.distraction_type

            p_info = {
                "person_id": actual_slot,
                "role": role_name,
                "ear": avg_ear,
                "ear_left": ear_l,
                "ear_right": ear_r,
                "mar": mar,
                "perclos": perclos,
                "pitch_raw": p_raw,
                "yaw_raw": y_raw,
                "roll_raw": r_raw,
                "pitch": p_rel,
                "yaw": y_rel,
                "roll": r_rel,
                "is_drowsy": p_is_drowsy,
                "is_yawning": p_is_yawning,
                "is_distracted": p_is_distracted,
                "is_smiling": smiling,
                "distraction_type": p_distraction_type,
                "landmarks_2d": landmarks_2d,
                "raw_landmarks_2d": raw_landmarks_2d,
                "rvec": rvec,
                "tvec": tvec,
                "cam_mat": cam_mat
            }
            persons_metrics.append(p_info)

        # Nếu ở chế độ Cabin 2 người mà Tài xế bị khuất mặt (chỉ thấy Phụ xe bên phải):
        if single_codriver_only:
            p_codriver = persons_metrics[0]
            p_driver_missing = {
                "person_id": 0,
                "role": "TÀI XẾ",
                "ear": 0.0,
                "ear_left": 0.0,
                "ear_right": 0.0,
                "mar": 0.0,
                "perclos": 0.0,
                "pitch_raw": 0.0,
                "yaw_raw": 0.0,
                "roll_raw": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "is_drowsy": False,
                "is_yawning": False,
                "is_distracted": True,
                "is_smiling": False,
                "distraction_type": "DRIVER OCCLUDED / OUT OF SEAT",
                "landmarks_2d": p_codriver["landmarks_2d"],
                "raw_landmarks_2d": p_codriver["raw_landmarks_2d"],
                "rvec": None,
                "tvec": None,
                "cam_mat": None
            }
            persons_metrics = [p_driver_missing, p_codriver]

        # Tổng hợp dữ liệu trả về
        metrics["face_detected"] = True
        metrics["person_count"] = len(persons_metrics)
        metrics["is_dual_mode"] = is_dual
        metrics["persons"] = persons_metrics

        p0 = persons_metrics[0]
        metrics.update({
            "ear": p0["ear"],
            "ear_left": p0["ear_left"],
            "ear_right": p0["ear_right"],
            "mar": p0["mar"],
            "perclos": p0["perclos"],
            "pitch_raw": p0["pitch_raw"],
            "yaw_raw": p0["yaw_raw"],
            "roll_raw": p0["roll_raw"],
            "pitch": p0["pitch"],
            "yaw": p0["yaw"],
            "roll": p0["roll"],
            "is_smiling": p0.get("is_smiling", False),
            "landmarks_2d": p0["landmarks_2d"],
            "driver": p0
        })
        self._smoothed_landmarks = np.array(p0["landmarks_2d"], dtype=np.float32)

        if not is_dual:
            metrics["is_drowsy"] = p0["is_drowsy"]
            metrics["is_yawning"] = p0["is_yawning"]
            metrics["is_distracted"] = p0["is_distracted"]
            metrics["distraction_type"] = p0["distraction_type"]
            self.is_drowsy = p0["is_drowsy"]
            self.is_yawning = p0["is_yawning"]
            self.is_distracted = p0["is_distracted"]
            self.distraction_type = p0["distraction_type"]

            if render_hud:
                annotated = self.draw_face_annotations(
                    annotated,
                    p0["landmarks_2d"],
                    p0,
                    draw_box=draw_box,
                    draw_contours=draw_contours,
                    draw_vectors=draw_vectors,
                    badge_mode="driver",
                    role_title="DRIVER #01"
                )
        else:
            p1 = persons_metrics[1]
            metrics["codriver"] = p1

            if p0["is_drowsy"]:
                metrics["dual_alert_type"] = "driver_drowsy"
                metrics["dual_alert_text"] = "TAI XE DANG NGU GAT! PHU XE HAY CAN THIEP XU LY!"
                metrics["is_drowsy"] = True
                metrics["is_yawning"] = False
                metrics["is_distracted"] = False
            elif p0["is_yawning"]:
                metrics["dual_alert_type"] = "driver_yawn"
                metrics["dual_alert_text"] = "TAI XE CO BIEU HIEN NGU GAT (NGAP)! PHU XE HAY NHAC NHO!"
                metrics["is_drowsy"] = False
                metrics["is_yawning"] = True
                metrics["is_distracted"] = False
            elif p0["is_distracted"]:
                metrics["dual_alert_type"] = "driver_distracted"
                metrics["dual_alert_text"] = "TAI XE DANG MAT TAP TRUNG / KHUAT MAT! PHU XE HAY CHU Y!"
                metrics["is_drowsy"] = False
                metrics["is_yawning"] = False
                metrics["is_distracted"] = True
                metrics["distraction_type"] = p0["distraction_type"]
            elif p1["is_drowsy"]:
                metrics["dual_alert_type"] = "codriver_drowsy"
                metrics["dual_alert_text"] = "PHU XE DANG NGU GAT (TAI XE VAN TINH TAO)"
                metrics["is_drowsy"] = False
                metrics["is_yawning"] = False
                metrics["is_distracted"] = False
            elif p1["is_yawning"]:
                metrics["dual_alert_type"] = "codriver_yawn"
                metrics["dual_alert_text"] = "PHU XE DANG NGAP NGHI"
                metrics["is_drowsy"] = False
                metrics["is_yawning"] = False
                metrics["is_distracted"] = False
            else:
                metrics["dual_alert_type"] = "normal"
                metrics["dual_alert_text"] = "CABIN 2 NGUOI: AN TOAN - TINH TAO"
                metrics["is_drowsy"] = False
                metrics["is_yawning"] = False
                metrics["is_distracted"] = False

            self.is_drowsy = metrics["is_drowsy"]
            self.is_yawning = metrics["is_yawning"]
            self.is_distracted = metrics["is_distracted"]
            self.distraction_type = p0["distraction_type"]

            if render_hud:
                if not single_codriver_only:
                    annotated = self.draw_face_annotations(
                        annotated,
                        p0["landmarks_2d"],
                        p0,
                        draw_box=draw_box,
                        draw_contours=draw_contours,
                        draw_vectors=draw_vectors,
                        badge_mode="driver",
                        role_title="TAI XE #01"
                    )
                annotated = self.draw_face_annotations(
                    annotated,
                    p1["landmarks_2d"],
                    p1,
                    draw_box=draw_box,
                    draw_contours=draw_contours,
                    draw_vectors=draw_vectors,
                    badge_mode="codriver",
                    role_title="PHU XE #02"
                )

                if metrics["dual_alert_type"] in ["driver_drowsy", "driver_yawn", "driver_distracted"]:
                    overlay = annotated.copy()
                    banner_color = (0, 0, 200) if metrics["dual_alert_type"] == "driver_drowsy" else (0, 120, 220)
                    cv2.rectangle(overlay, (0, 0), (w, 42), banner_color, -1)
                    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
                    cv2.putText(
                        annotated,
                        f"[CANH BAO 2 NGUOI] {metrics['dual_alert_text']}",
                        (18, 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        t_elapsed = (time.perf_counter() - t_start_process) * 1000.0
        metrics["proc_latency_ms"] = t_elapsed
        metrics["is_lagging"] = (t_elapsed > 65.0)
        if render_hud and metrics["is_lagging"]:
            annotated = self.draw_lag_loading_hud(annotated, latency_ms=t_elapsed)

        return metrics, annotated

    def draw_lag_loading_hud(self, img, latency_ms=None, fps=None):
        """
        Vẽ hiệu ứng Loading Spinner công nghệ cao và thông báo tình trạng tải/độ trễ
        khi hệ thống hoặc luồng camera bị chậm/lag.
        """
        h, w = img.shape[:2]
        center = (w - 32, 24)
        angle = int((time.time() * 320.0) % 360)

        box_w, box_h = 240, 32
        bx1, by1 = max(0, w - box_w - 12), 8
        bx2, by2 = w - 12, 8 + box_h

        overlay = img.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 180, 255), 1, cv2.LINE_AA)

        cv2.circle(img, center, 10, (40, 48, 56), 2, cv2.LINE_AA)
        cv2.ellipse(img, center, (10, 10), 0, angle, angle + 110, (0, 225, 255), 2, cv2.LINE_AA)
        cv2.ellipse(img, center, (10, 10), 0, angle + 180, angle + 290, (0, 255, 128), 2, cv2.LINE_AA)

        if latency_ms is not None and latency_ms > 0:
            info_txt = f"DANG XU LY ({latency_ms:.0f}ms)"
        elif fps is not None:
            info_txt = f"FPS THAP ({fps:.1f} FPS)"
        else:
            info_txt = "DANG TAI DU LIEU..."

        cv2.putText(img, info_txt, (bx1 + 10, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 225, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "[OPTIMIZING BUFFER]", (bx1 + 10, by1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 200, 220), 1, cv2.LINE_AA)
        return img

    def _draw_pose_axes(self, img, rvec, tvec, cam_mat, nose_pt):
        axis_len = 80.0
        axis_3d = np.array([
            (axis_len, 0.0, 0.0),
            (0.0, -axis_len, 0.0),
            (0.0, 0.0, -axis_len)
        ], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_mat, np.zeros((4, 1)))
        p_nose = (int(nose_pt[0]), int(nose_pt[1]))
        p_x = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
        p_y = (int(img_pts[1][0][0]), int(img_pts[1][0][1]))
        p_z = (int(img_pts[2][0][0]), int(img_pts[2][0][1]))

        cv2.line(img, p_nose, p_x, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, p_nose, p_y, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, p_nose, p_z, (255, 140, 0), 2, cv2.LINE_AA)

    def draw_face_annotations(
        self,
        img,
        landmarks_2d,
        metrics,
        draw_box=True,
        draw_contours=True,
        draw_vectors=True,
        badge_mode="driver",
        role_title=None
    ):
        if not landmarks_2d or len(landmarks_2d) < 468:
            return img

        h, w = img.shape[:2]
        pts = np.array(landmarks_2d, dtype=np.int32)
        min_x = int(np.min(pts[:, 0]))
        min_y = int(np.min(pts[:, 1]))
        max_x = int(np.max(pts[:, 0]))
        max_y = int(np.max(pts[:, 1]))
        bw = max_x - min_x
        bh = max_y - min_y

        if draw_contours:
            jaw_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
            pts_jaw = np.array([landmarks_2d[i] for i in jaw_idx], dtype=np.int32)
            cv2.polylines(img, [pts_jaw], False, (140, 160, 180), 1, cv2.LINE_AA)

            eb_l = [70, 63, 105, 66, 107]
            eb_r = [336, 296, 334, 293, 300]
            pts_eb_l = np.array([landmarks_2d[i] for i in eb_l], dtype=np.int32)
            pts_eb_r = np.array([landmarks_2d[i] for i in eb_r], dtype=np.int32)
            cv2.polylines(img, [pts_eb_l], False, (0, 242, 254), 1, cv2.LINE_AA)
            cv2.polylines(img, [pts_eb_r], False, (0, 242, 254), 1, cv2.LINE_AA)

            nose_idx = [168, 6, 197, 195, 5, 4, 1]
            pts_nose = np.array([landmarks_2d[i] for i in nose_idx], dtype=np.int32)
            cv2.polylines(img, [pts_nose], False, (0, 242, 254), 1, cv2.LINE_AA)

        if draw_vectors:
            cv2.line(img, landmarks_2d[159], landmarks_2d[145], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(img, landmarks_2d[386], landmarks_2d[374], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(img, landmarks_2d[33], landmarks_2d[133], (0, 200, 255), 1, cv2.LINE_AA)
            cv2.line(img, landmarks_2d[362], landmarks_2d[263], (0, 200, 255), 1, cv2.LINE_AA)
            cv2.line(img, landmarks_2d[13], landmarks_2d[14], (0, 140, 255), 2, cv2.LINE_AA)
            rvec_p = metrics.get("rvec")
            tvec_p = metrics.get("tvec")
            cam_p = metrics.get("cam_mat")
            if rvec_p is not None and tvec_p is not None and cam_p is not None:
                self._draw_pose_axes(img, rvec_p, tvec_p, cam_p, landmarks_2d[1])

        if draw_box:
            pad_x = int(bw * 0.10)
            pad_y_top = int(bh * 0.22)
            pad_y_bot = int(bh * 0.08)
            bx1 = max(0, min_x - pad_x)
            by1 = max(0, min_y - pad_y_top)
            bx2 = min(w, max_x + pad_x)
            by2 = min(h, max_y + pad_y_bot)

            if badge_mode == "driver":
                prefix = role_title if role_title else "TAI XE #01"
                if metrics.get("is_drowsy"):
                    box_color = (0, 0, 255)
                    badge_txt = f"{prefix}: NGU GAT!"
                elif metrics.get("is_yawning"):
                    box_color = (0, 140, 255)
                    badge_txt = f"{prefix}: BIEU HIEN NGU GAT (NGAP)"
                elif metrics.get("is_distracted"):
                    box_color = (0, 69, 255)
                    badge_txt = f"{prefix}: {metrics.get('distraction_type', 'MAT TAP TRUNG')}"
                else:
                    box_color = (0, 255, 180)
                    badge_txt = f"{prefix}: TINH TAO"
            elif badge_mode in ["codriver", "passenger"]:
                prefix = role_title if role_title else "PHU XE #02"
                if metrics.get("is_drowsy"):
                    box_color = (180, 0, 255)
                    badge_txt = f"{prefix}: NGU GAT"
                elif metrics.get("is_yawning"):
                    box_color = (0, 165, 255)
                    badge_txt = f"{prefix}: BIEU HIEN NGU GAT (NGAP)"
                elif metrics.get("is_distracted"):
                    box_color = (0, 200, 255)
                    badge_txt = f"{prefix}: QUAY MAT"
                else:
                    box_color = (255, 200, 0)
                    badge_txt = f"{prefix}: DONG HANH"
            elif badge_mode == "biometric":
                if metrics.get("liveness_success"):
                    box_color = (0, 255, 128)
                    badge_txt = "SINH TRAC HOC: NGUOI THAT"
                else:
                    box_color = (0, 242, 254)
                    badge_txt = "SINH TRAC HOC: DANG XAC THUC"
            else:
                box_color = (0, 242, 254)
                badge_txt = None

            c_len = min(28, int((bx2 - bx1) * 0.18))
            thick = 2
            cv2.line(img, (bx1, by1), (bx1 + c_len, by1), box_color, thick, cv2.LINE_AA)
            cv2.line(img, (bx1, by1), (bx1, by1 + c_len), box_color, thick, cv2.LINE_AA)
            cv2.line(img, (bx2, by1), (bx2 - c_len, by1), box_color, thick, cv2.LINE_AA)
            cv2.line(img, (bx2, by1), (bx2, by1 + c_len), box_color, thick, cv2.LINE_AA)
            cv2.line(img, (bx1, by2), (bx1 + c_len, by2), box_color, thick, cv2.LINE_AA)
            cv2.line(img, (bx1, by2), (bx1, by2 - c_len), box_color, thick, cv2.LINE_AA)
            cv2.line(img, (bx2, by2), (bx2 - c_len, by2), box_color, thick, cv2.LINE_AA)
            cv2.line(img, (bx2, by2), (bx2, by2 - c_len), box_color, thick, cv2.LINE_AA)

            cv2.rectangle(img, (bx1, by1), (bx2, by2), (50, 60, 70), 1)

            if badge_txt:
                (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (bx1, max(0, by1 - 22)), (bx1 + tw + 14, by1), box_color, -1)
                cv2.putText(img, badge_txt, (bx1 + 7, max(14, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return img

    def align_face(self, frame_bgr, landmarks_2d, desired_width=256, desired_height=256):
        pt_l_outer, pt_l_inner = landmarks_2d[33], landmarks_2d[133]
        left_eye_center = (
            (pt_l_outer[0] + pt_l_inner[0]) / 2.0,
            (pt_l_outer[1] + pt_l_inner[1]) / 2.0
        )

        pt_r_inner, pt_r_outer = landmarks_2d[362], landmarks_2d[263]
        right_eye_center = (
            (pt_r_inner[0] + pt_r_outer[0]) / 2.0,
            (pt_r_inner[1] + pt_r_outer[1]) / 2.0
        )

        d_x = right_eye_center[0] - left_eye_center[0]
        d_y = right_eye_center[1] - left_eye_center[1]
        angle = math.degrees(math.atan2(d_y, d_x))

        desired_left_eye_x = 0.35
        desired_right_eye_x = 1.0 - desired_left_eye_x
        dist_current = math.sqrt((d_x ** 2) + (d_y ** 2))
        dist_desired = (desired_right_eye_x - desired_left_eye_x) * desired_width
        scale = dist_desired / max(1.0, dist_current)

        eye_center = (
            (left_eye_center[0] + right_eye_center[0]) / 2.0,
            (left_eye_center[1] + right_eye_center[1]) / 2.0
        )

        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        t_x = desired_width * 0.5
        t_y = desired_height * 0.38
        M[0, 2] += (t_x - eye_center[0])
        M[1, 2] += (t_y - eye_center[1])

        aligned_face = cv2.warpAffine(
            frame_bgr,
            M,
            (desired_width, desired_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned_face, angle

    def overlay_glasses_homography(self, frame_bgr, landmarks_2d, glasses_bgra):
        gh, gw = glasses_bgra.shape[:2]
        fh, fw = frame_bgr.shape[:2]

        p_left_temple = np.array(landmarks_2d[127], dtype=np.float32)
        p_right_temple = np.array(landmarks_2d[356], dtype=np.float32)
        p_nose_bridge = np.array(landmarks_2d[168], dtype=np.float32)

        vec_lr = p_right_temple - p_left_temple
        temple_dist = float(np.linalg.norm(vec_lr))
        if temple_dist < 10.0:
            return frame_bgr

        u_vec = vec_lr / temple_dist
        up_vec = np.array([u_vec[1], -u_vec[0]], dtype=np.float32)

        p_left_anchor = p_left_temple - u_vec * (temple_dist * 0.04)
        p_right_anchor = p_right_temple + u_vec * (temple_dist * 0.04)

        d_left = float(np.linalg.norm(p_nose_bridge - p_left_temple))
        d_right = float(np.linalg.norm(p_right_temple - p_nose_bridge))
        d_avg = max(1.0, (d_left + d_right) * 0.5)

        scale_l = np.clip(0.65 + 0.35 * (d_left / d_avg), 0.60, 1.35)
        scale_r = np.clip(0.65 + 0.35 * (d_right / d_avg), 0.60, 1.35)

        base_height = temple_dist * 0.43
        h_left = base_height * scale_l
        h_right = base_height * scale_r

        mid_temples = 0.5 * (p_left_temple + p_right_temple)
        bridge_vert_offset = float(np.dot(p_nose_bridge - mid_temples, up_vec))
        vert_shift = up_vec * (bridge_vert_offset * 0.65)

        dst_top_left = p_left_anchor + vert_shift + up_vec * (h_left * 0.46)
        dst_bot_left = p_left_anchor + vert_shift - up_vec * (h_left * 0.54)
        dst_top_right = p_right_anchor + vert_shift + up_vec * (h_right * 0.46)
        dst_bot_right = p_right_anchor + vert_shift - up_vec * (h_right * 0.54)

        h_center = 0.5 * (h_left + h_right)
        dst_bridge_top = p_nose_bridge + up_vec * (h_center * 0.30)
        dst_bridge_bot = p_nose_bridge - up_vec * (h_center * 0.38)

        src_pts = np.array([
            [0.0, 0.0],
            [gw - 1.0, 0.0],
            [gw - 1.0, gh - 1.0],
            [0.0, gh - 1.0],
            [gw * 0.5, gh * 0.22],
            [gw * 0.5, gh * 0.68]
        ], dtype=np.float32)

        dst_pts = np.array([
            dst_top_left,
            dst_top_right,
            dst_bot_right,
            dst_bot_left,
            dst_bridge_top,
            dst_bridge_bot
        ], dtype=np.float32)

        x_min = max(0, int(np.floor(np.min(dst_pts[:, 0]))) - 4)
        y_min = max(0, int(np.floor(np.min(dst_pts[:, 1]))) - 4)
        x_max = min(fw, int(np.ceil(np.max(dst_pts[:, 0]))) + 4)
        y_max = min(fh, int(np.ceil(np.max(dst_pts[:, 1]))) + 4)

        if x_max <= x_min or y_max <= y_min:
            return frame_bgr

        dst_pts_roi = dst_pts - np.array([x_min, y_min], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts_roi, method=0)
        if H is None:
            return frame_bgr

        roi_w = x_max - x_min
        roi_h = y_max - y_min
        warped_roi = cv2.warpPerspective(
            glasses_bgra,
            H,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        output = frame_bgr.copy()
        roi_bg = output[y_min:y_max, x_min:x_max].astype(np.float32)
        alpha = (warped_roi[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
        blended_roi = (1.0 - alpha) * roi_bg + alpha * warped_roi[:, :, :3].astype(np.float32)
        output[y_min:y_max, x_min:x_max] = np.clip(blended_roi, 0, 255).astype(np.uint8)
        return output

    def apply_makeup(
        self,
        frame_bgr,
        landmarks_2d,
        lip_bgr,
        lip_alpha=0.65,
        blush_bgr=None,
        blush_alpha=0.45,
        color_space="HYBRID"
    ):
        h, w = frame_bgr.shape[:2]
        result = frame_bgr.copy()

        outer_lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        inner_lip_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        outer_pts = np.array([landmarks_2d[i] for i in outer_lip_idx], dtype=np.int32)
        inner_pts = np.array([landmarks_2d[i] for i in inner_lip_idx], dtype=np.int32)

        pad = 18
        lx1 = max(0, int(np.min(outer_pts[:, 0])) - pad)
        ly1 = max(0, int(np.min(outer_pts[:, 1])) - pad)
        lx2 = min(w, int(np.max(outer_pts[:, 0])) + pad)
        ly2 = min(h, int(np.max(outer_pts[:, 1])) + pad)

        if lx2 > lx1 and ly2 > ly1:
            roi_h, roi_w = ly2 - ly1, lx2 - lx1
            lip_mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            offset = np.array([lx1, ly1], dtype=np.int32)
            cv2.fillPoly(lip_mask_roi, [outer_pts - offset], 255)
            cv2.fillPoly(lip_mask_roi, [inner_pts - offset], 0)

            lip_mask_blur = cv2.GaussianBlur(lip_mask_roi, (7, 7), 2.0)
            norm_lip_mask = (lip_mask_blur.astype(np.float32) / 255.0) * float(lip_alpha)

            roi_bgr = result[ly1:ly2, lx1:lx2]
            target_lip_pixel = np.uint8([[lip_bgr]])
            mode = str(color_space).upper()

            bgr_from_lab = None
            bgr_from_hls = None

            if mode in ("LAB", "HYBRID"):
                roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                target_lab = cv2.cvtColor(target_lip_pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                colored_lab = roi_lab.copy()
                colored_lab[:, :, 0] = np.clip(roi_lab[:, :, 0] * 0.40 + target_lab[0] * 0.60, 0, 255)
                colored_lab[:, :, 1] = np.clip(roi_lab[:, :, 1] * 0.10 + target_lab[1] * 0.90, 0, 255)
                colored_lab[:, :, 2] = np.clip(roi_lab[:, :, 2] * 0.10 + target_lab[2] * 0.90, 0, 255)
                bgr_from_lab = cv2.cvtColor(np.clip(colored_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

            if mode in ("HLS", "HYBRID"):
                roi_hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
                target_hls = cv2.cvtColor(target_lip_pixel, cv2.COLOR_BGR2HLS)[0, 0].astype(np.float32)
                colored_hls = roi_hls.copy()
                colored_hls[:, :, 0] = target_hls[0]
                colored_hls[:, :, 1] = np.clip(roi_hls[:, :, 1] * 0.35 + target_hls[1] * 0.65, 0, 255)
                colored_hls[:, :, 2] = np.clip(target_hls[2] * 0.95 + roi_hls[:, :, 2] * 0.10, 0, 255)
                bgr_from_hls = cv2.cvtColor(np.clip(colored_hls, 0, 255).astype(np.uint8), cv2.COLOR_HLS2BGR)

            if mode == "LAB":
                colored_bgr = bgr_from_lab
            elif mode == "HLS":
                colored_bgr = bgr_from_hls
            else:
                colored_bgr = cv2.addWeighted(bgr_from_lab, 0.50, bgr_from_hls, 0.50, 0)

            mask_3d = norm_lip_mask[:, :, np.newaxis]
            blended_lip = (1.0 - mask_3d) * roi_bgr.astype(np.float32) + mask_3d * colored_bgr.astype(np.float32)
            result[ly1:ly2, lx1:lx2] = np.clip(blended_lip, 0, 255).astype(np.uint8)

        if blush_bgr is not None and blush_alpha > 0.05:
            face_width = self._euclidean_dist(landmarks_2d[234], landmarks_2d[454])
            eye_dx = float(landmarks_2d[263][0] - landmarks_2d[33][0])
            eye_dy = float(landmarks_2d[263][1] - landmarks_2d[33][1])
            roll_deg = math.degrees(math.atan2(eye_dy, max(1.0, abs(eye_dx))))

            nose_pt = landmarks_2d[168]
            c_r = (int(landmarks_2d[50][0]), int(landmarks_2d[50][1]))
            c_l = (int(landmarks_2d[280][0]), int(landmarks_2d[280][1]))

            # Tính tỷ lệ co giãn phối cảnh riêng cho từng bên gò má khi quay mặt nghiêng 3D
            dist_r = self._euclidean_dist(nose_pt, c_r)
            dist_l = self._euclidean_dist(nose_pt, c_l)
            dist_avg = max(1.0, 0.5 * (dist_r + dist_l))
            scale_r = float(np.clip(dist_r / dist_avg, 0.45, 1.25))
            scale_l = float(np.clip(dist_l / dist_avg, 0.45, 1.25))

            base_rx = max(12, int(face_width * 0.17))
            base_ry = max(9, int(face_width * 0.12))

            for (cx, cy), s_factor in ((c_r, scale_r), (c_l, scale_l)):
                rx = max(8, int(base_rx * s_factor))
                ry = max(6, int(base_ry * (0.75 + 0.25 * s_factor)))
                ksize = max(21, (int(max(rx, ry) * 1.5) // 2) * 2 + 1)
                pad_b = ksize

                bx1 = max(0, cx - rx - pad_b)
                by1 = max(0, cy - ry - pad_b)
                bx2 = min(w, cx + rx + pad_b)
                by2 = min(h, cy + ry + pad_b)

                if bx2 <= bx1 or by2 <= by1:
                    continue

                b_mask = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
                cv2.ellipse(b_mask, (cx - bx1, cy - by1), (rx, ry), roll_deg, 0, 360, 255, -1)
                b_blur = cv2.GaussianBlur(b_mask, (ksize, ksize), ksize * 0.40)
                norm_b = ((b_blur.astype(np.float32) / 255.0) * float(blush_alpha))[:, :, np.newaxis]

                cheek_roi = result[by1:by2, bx1:bx2].astype(np.float32)
                blush_color = np.array(blush_bgr, dtype=np.float32).reshape((1, 1, 3))
                result[by1:by2, bx1:bx2] = np.clip((1.0 - norm_b) * cheek_roi + norm_b * blush_color, 0, 255).astype(np.uint8)

        return result

    def analyze_gaze(self, landmarks_2d):
        iris_l = np.array(landmarks_2d[468], dtype=np.float64)
        eye_l_out = np.array(landmarks_2d[33], dtype=np.float64)
        eye_l_in = np.array(landmarks_2d[133], dtype=np.float64)
        lid_l_top = np.array(landmarks_2d[159], dtype=np.float64)
        lid_l_bot = np.array(landmarks_2d[145], dtype=np.float64)

        iris_r = np.array(landmarks_2d[473], dtype=np.float64)
        eye_r_in = np.array(landmarks_2d[362], dtype=np.float64)
        eye_r_out = np.array(landmarks_2d[263], dtype=np.float64)
        lid_r_top = np.array(landmarks_2d[386], dtype=np.float64)
        lid_r_bot = np.array(landmarks_2d[374], dtype=np.float64)

        vec_l = eye_l_in - eye_l_out
        width_l = max(1e-6, float(np.linalg.norm(vec_l)))
        unit_l = vec_l / width_l
        proj_h_l = float(np.dot(iris_l - eye_l_out, unit_l)) / width_l

        vec_r = eye_r_out - eye_r_in
        width_r = max(1e-6, float(np.linalg.norm(vec_r)))
        unit_r = vec_r / width_r
        proj_h_r = float(np.dot(iris_r - eye_r_in, unit_r)) / width_r

        gaze_h_ratio = float(np.clip((proj_h_l + proj_h_r) * 0.5, 0.0, 1.0))

        norm_l = np.array([-unit_l[1], unit_l[0]], dtype=np.float64)
        mid_l = 0.5 * (eye_l_out + eye_l_in)
        offset_v_l = float(np.dot(iris_l - mid_l, norm_l)) / width_l

        norm_r = np.array([-unit_r[1], unit_r[0]], dtype=np.float64)
        mid_r = 0.5 * (eye_r_in + eye_r_out)
        offset_v_r = float(np.dot(iris_r - mid_r, norm_r)) / width_r

        dist_l_top = float(np.linalg.norm(iris_l - lid_l_top))
        dist_l_bot = float(np.linalg.norm(iris_l - lid_l_bot))
        lid_ratio_l = dist_l_top / max(1e-6, dist_l_top + dist_l_bot)

        dist_r_top = float(np.linalg.norm(iris_r - lid_r_top))
        dist_r_bot = float(np.linalg.norm(iris_r - lid_r_bot))
        lid_ratio_r = dist_r_top / max(1e-6, dist_r_top + dist_r_bot)

        lid_v_ratio = (lid_ratio_l + lid_ratio_r) * 0.5
        corner_v_ratio = 0.50 + ((offset_v_l + offset_v_r) * 0.5 + 0.025) * 2.8
        gaze_v_ratio = float(np.clip(0.55 * corner_v_ratio + 0.45 * lid_v_ratio, 0.0, 1.0))

        gaze_x = float(np.clip((gaze_h_ratio - 0.50) * 4.5, -1.0, 1.0))
        gaze_y = float(np.clip((gaze_v_ratio - 0.50) * 4.5, -1.0, 1.0))

        if gaze_h_ratio < 0.42:
            h_direction = "LIẾC TRÁI"
        elif gaze_h_ratio > 0.58:
            h_direction = "LIẾC PHẢI"
        else:
            h_direction = "NHÌN THẲNG"

        if gaze_v_ratio < 0.42:
            v_direction = "NHÌN LÊN"
        elif gaze_v_ratio > 0.58:
            v_direction = "NHÌN XUỐNG"
        else:
            v_direction = "CÂN BẰNG"

        return {
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "h_ratio": gaze_h_ratio,
            "v_ratio": gaze_v_ratio,
            "ratio_l": float(proj_h_l),
            "ratio_r": float(proj_h_r),
            "h_direction": h_direction,
            "v_direction": v_direction,
            "iris_left": (int(round(iris_l[0])), int(round(iris_l[1]))),
            "iris_right": (int(round(iris_r[0])), int(round(iris_r[1])))
        }

    def draw_iris_landmarks(self, frame_bgr, landmarks_2d, gaze_data=None):
        """
        Vẽ tâm mống mắt (468, 473), vòng mống mắt và vector chiếu ánh mắt 2D trực tiếp lên ảnh.
        Tương thích đồng nhất cho cả Live Camera, Video, Ảnh tĩnh và luồng WebRTC từ xa.
        """
        if frame_bgr is None or not landmarks_2d or len(landmarks_2d) < 474:
            return frame_bgr

        output = frame_bgr.copy()
        if gaze_data is None:
            gaze_data = self.analyze_gaze(landmarks_2d)

        il = gaze_data["iris_left"]
        ir = gaze_data["iris_right"]
        gx = gaze_data["gaze_x"]
        gy = gaze_data["gaze_y"]
        h_dir = gaze_data["h_direction"]

        # Vẽ vòng mống mắt (469-472 và 474-477) nếu có đủ 478 mốc
        if len(landmarks_2d) >= 478:
            pts_il = np.array([landmarks_2d[i] for i in [469, 470, 471, 472]], dtype=np.int32)
            pts_ir = np.array([landmarks_2d[i] for i in [474, 475, 476, 477]], dtype=np.int32)
            cv2.polylines(output, [pts_il], True, (0, 242, 254), 1, cv2.LINE_AA)
            cv2.polylines(output, [pts_ir], True, (0, 242, 254), 1, cv2.LINE_AA)

        cv2.circle(output, il, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, ir, 4, (0, 255, 255), -1, cv2.LINE_AA)
        arrow_len = 48.0
        end_l = (int(il[0] + gx * arrow_len), int(il[1] + gy * arrow_len))
        end_r = (int(ir[0] + gx * arrow_len), int(ir[1] + gy * arrow_len))
        cv2.arrowedLine(output, il, end_l, (0, 242, 254), 2, cv2.LINE_AA, tipLength=0.28)
        cv2.arrowedLine(output, ir, end_r, (0, 242, 254), 2, cv2.LINE_AA, tipLength=0.28)
        cv2.putText(
            output,
            f"GAZE: {h_dir} (L:{gaze_data['ratio_l']:.2f} R:{gaze_data['ratio_r']:.2f})",
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        return output

    # -----------------------------------------------------------------
    # PHÂN HỆ 6: FACE MORPHING & FACE SWAPPING (DELAUNAY TRIANGULATION,
    # PIECE-WISE AFFINE WARPING, POISSON SEAMLESS CLONING & LAB TRANSFER)
    # -----------------------------------------------------------------
    MORPH_KEY_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109,
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        168, 6, 197, 195, 5, 4, 1, 98, 327,
        61, 40, 0, 270, 291, 321, 17, 84, 78, 13, 308, 14
    ]

    FACE_OVAL_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109
    ]

    def extract_single_face_landmarks(self, img_bgr):
        """Trích xuất nhanh 478 mốc 2D trên ảnh tĩnh mẫu phục vụ Face Swap / Morphing."""
        if img_bgr is None or img_bgr.size == 0:
            return None
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with self._infer_lock:
            res = self.face_mesh.process(rgb)
        if not res or not res.multi_face_landmarks:
            return None
        face_lm = res.multi_face_landmarks[0]
        return [(int(round(lm.x * w)), int(round(lm.y * h))) for lm in face_lm.landmark]

    @classmethod
    def compute_delaunay_triangles(cls, pts_2d, img_w, img_h):
        """
        Chia lưới tam giác Delaunay (cv2.Subdiv2D) trên tập điểm mốc đặc trưng.
        Trả về danh sách bộ 3 chỉ số đỉnh (i1, i2, i3) thỏa mãn tính chất đường tròn ngoại tiếp rỗng.
        """
        pts_arr = np.array(pts_2d, dtype=np.float32)
        rect = (0, 0, int(img_w), int(img_h))
        subdiv = cv2.Subdiv2D(rect)
        for pt in pts_arr:
            px = float(np.clip(pt[0], 1.0, img_w - 2.0))
            py = float(np.clip(pt[1], 1.0, img_h - 2.0))
            subdiv.insert((px, py))

        tri_list = subdiv.getTriangleList()
        tri_indices = []
        seen = set()
        for t in tri_list:
            v1 = np.array([t[0], t[1]], dtype=np.float32)
            v2 = np.array([t[2], t[3]], dtype=np.float32)
            v3 = np.array([t[4], t[5]], dtype=np.float32)
            if (
                min(v1[0], v2[0], v3[0]) < 0 or max(v1[0], v2[0], v3[0]) >= img_w or
                min(v1[1], v2[1], v3[1]) < 0 or max(v1[1], v2[1], v3[1]) >= img_h
            ):
                continue
            i1 = int(np.argmin(np.sum((pts_arr - v1) ** 2, axis=1)))
            i2 = int(np.argmin(np.sum((pts_arr - v2) ** 2, axis=1)))
            i3 = int(np.argmin(np.sum((pts_arr - v3) ** 2, axis=1)))
            if len({i1, i2, i3}) == 3:
                key = tuple(sorted((i1, i2, i3)))
                if key not in seen:
                    seen.add(key)
                    tri_indices.append((i1, i2, i3))
        return tri_indices

    @staticmethod
    def _warp_affine_triangle(img_src, img_dst, t_src, t_dst):
        """Biến đổi Affine từng mảnh tam giác (Piece-wise Affine Triangle Warping)."""
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return
        sh, sw = img_src.shape[:2]
        dh, dw = img_dst.shape[:2]
        if x1 < 0 or y1 < 0 or x1 + w1 > sw or y1 + h1 > sh:
            return
        if x2 < 0 or y2 < 0 or x2 + w2 > dw or y2 + h2 > dh:
            return

        t1_rect = np.array([[t_src[i][0] - x1, t_src[i][1] - y1] for i in range(3)], dtype=np.float32)
        t2_rect = np.array([[t_dst[i][0] - x2, t_dst[i][1] - y2] for i in range(3)], dtype=np.float32)

        mask = np.zeros((h2, w2, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

        src_patch = img_src[y1:y1+h1, x1:x1+w1]
        M = cv2.getAffineTransform(t1_rect, t2_rect)
        warped_patch = cv2.warpAffine(
            src_patch, M, (w2, h2), None,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        dst_roi = img_dst[y2:y2+h2, x2:x2+w2].astype(np.float32)
        blended = dst_roi * (1.0 - mask) + warped_patch.astype(np.float32) * mask
        img_dst[y2:y2+h2, x2:x2+w2] = np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _morph_affine_triangle(img1, img2, img_morph, t1, t2, t_mid, alpha):
        """Nội suy đồng thời hình học và cường độ màu của tam giác với trọng số alpha ∈ [0, 1]."""
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        rm = cv2.boundingRect(np.float32([t_mid]))
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        xm, ym, wm, hm = rm
        if min(w1, h1, w2, h2, wm, hm) <= 0:
            return
        h, w = img_morph.shape[:2]
        if (
            x1 < 0 or y1 < 0 or x1 + w1 > img1.shape[1] or y1 + h1 > img1.shape[0] or
            x2 < 0 or y2 < 0 or x2 + w2 > img2.shape[1] or y2 + h2 > img2.shape[0] or
            xm < 0 or ym < 0 or xm + wm > w or ym + hm > h
        ):
            return

        t1_rect = np.array([[t1[i][0] - x1, t1[i][1] - y1] for i in range(3)], dtype=np.float32)
        t2_rect = np.array([[t2[i][0] - x2, t2[i][1] - y2] for i in range(3)], dtype=np.float32)
        tm_rect = np.array([[t_mid[i][0] - xm, t_mid[i][1] - ym] for i in range(3)], dtype=np.float32)

        mask = np.zeros((hm, wm, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tm_rect), (1.0, 1.0, 1.0), 16, 0)

        M1 = cv2.getAffineTransform(t1_rect, tm_rect)
        M2 = cv2.getAffineTransform(t2_rect, tm_rect)
        warp1 = cv2.warpAffine(img1[y1:y1+h1, x1:x1+w1], M1, (wm, hm), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warp2 = cv2.warpAffine(img2[y2:y2+h2, x2:x2+w2], M2, (wm, hm), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        morphed_patch = (1.0 - float(alpha)) * warp1.astype(np.float32) + float(alpha) * warp2.astype(np.float32)
        dst_roi = img_morph[ym:ym+hm, xm:xm+wm].astype(np.float32)
        img_morph[ym:ym+hm, xm:xm+wm] = np.clip(dst_roi * (1.0 - mask) + morphed_patch * mask, 0, 255).astype(np.uint8)

    @staticmethod
    def _reinhard_color_transfer_roi(src_roi, dst_roi, mask_roi):
        """Cân bằng thống kê màu da (Mean & Std) trên không gian màu CIELAB (Reinhard et al.)."""
        if src_roi.size == 0 or dst_roi.size == 0:
            return src_roi
        src_lab = cv2.cvtColor(src_roi, cv2.COLOR_BGR2LAB)
        dst_lab = cv2.cvtColor(dst_roi, cv2.COLOR_BGR2LAB)
        s_mean, s_std = cv2.meanStdDev(src_lab, mask=mask_roi)
        d_mean, d_std = cv2.meanStdDev(dst_lab, mask=mask_roi)
        scale = np.clip(d_std.flatten() / np.maximum(1.0, s_std.flatten()), 0.55, 1.80)
        offset = d_mean.flatten() - s_mean.flatten() * scale
        out_lab = np.clip(src_lab.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    def swap_and_morph_faces(
        self,
        img_target,
        landmarks_target,
        img_source,
        landmarks_source,
        alpha=1.0,
        blend_mode="LAB_FEATHER",
        show_delaunay=False
    ):
        """
        Thực hiện Hoán đổi khuôn mặt (Face Swapping khi alpha = 1.0) hoặc Biến hình khuôn mặt
        liên tục (Face Morphing với 0.0 <= alpha <= 1.0) giữa khuôn mặt nguồn và khuôn mặt đích:
        1. Trích xuất 79 mốc cấu trúc giải phẫu chính từ lưới 478 mốc MediaPipe.
        2. Chia lưới tam giác Delaunay (cv2.Subdiv2D) trên mặt đích hoặc mặt trung gian P_m = (1-alpha)*P_dst + alpha*P_src.
        3. Ánh xạ từng tam giác bằng biến đổi Affine 2x3 (cv2.getAffineTransform & cv2.warpAffine).
        4. Hòa trộn biên mượt mà qua mặt nạ bao lồi (Convex Hull) bằng Chuyển giao màu CIELAB + Gaussian Feathering
           (tốc độ cao 30 FPS) hoặc Trộn miền đạo hàm Poisson (cv2.seamlessClone).
        """
        if (
            img_target is None or img_source is None or
            not landmarks_target or not landmarks_source or
            len(landmarks_target) < 468 or len(landmarks_source) < 468
        ):
            return img_target, 0

        h_t, w_t = img_target.shape[:2]
        h_s, w_s = img_source.shape[:2]
        alpha_f = float(np.clip(alpha, 0.0, 1.0))

        pts_dst = np.array([landmarks_target[i] for i in self.MORPH_KEY_INDICES], dtype=np.float32)
        pts_src = np.array([landmarks_source[i] for i in self.MORPH_KEY_INDICES], dtype=np.float32)

        mode_up = str(blend_mode).upper()
        if "MORPH" in mode_up and (h_s, w_s) == (h_t, w_t):
            # Chế độ Morphing toàn ảnh giữa 2 chân dung có cùng kích thước
            pts_mid = (1.0 - alpha_f) * pts_dst + alpha_f * pts_src
            tri_indices = self.compute_delaunay_triangles(pts_mid, w_t, h_t)
            morphed_canvas = cv2.addWeighted(img_target, 1.0 - alpha_f, img_source, alpha_f, 0)
            for i1, i2, i3 in tri_indices:
                t_d = [pts_dst[i1], pts_dst[i2], pts_dst[i3]]
                t_s = [pts_src[i1], pts_src[i2], pts_src[i3]]
                t_m = [pts_mid[i1], pts_mid[i2], pts_mid[i3]]
                self._morph_affine_triangle(img_target, img_source, morphed_canvas, t_d, t_s, t_m, alpha_f)

            if show_delaunay:
                for i1, i2, i3 in tri_indices:
                    pt1 = (int(pts_mid[i1][0]), int(pts_mid[i1][1]))
                    pt2 = (int(pts_mid[i2][0]), int(pts_mid[i2][1]))
                    pt3 = (int(pts_mid[i3][0]), int(pts_mid[i3][1]))
                    cv2.line(morphed_canvas, pt1, pt2, (0, 242, 254), 1, cv2.LINE_AA)
                    cv2.line(morphed_canvas, pt2, pt3, (0, 242, 254), 1, cv2.LINE_AA)
                    cv2.line(morphed_canvas, pt3, pt1, (0, 242, 254), 1, cv2.LINE_AA)
            return morphed_canvas, len(tri_indices)

        # Chế độ Face Swap / In-Place Morphing trên khung hình đích (camera trực tiếp hoặc ảnh đích)
        tri_indices = self.compute_delaunay_triangles(pts_dst, w_t, h_t)
        warped_src_on_dst = img_target.copy()
        for i1, i2, i3 in tri_indices:
            t_s = [pts_src[i1], pts_src[i2], pts_src[i3]]
            t_d = [pts_dst[i1], pts_dst[i2], pts_dst[i3]]
            self._warp_affine_triangle(img_source, warped_src_on_dst, t_s, t_d)

        oval_pts = np.array([landmarks_target[i] for i in self.FACE_OVAL_INDICES], dtype=np.int32)
        hull = cv2.convexHull(oval_pts)
        rx, ry, rw, rh = cv2.boundingRect(hull)
        pad = 12
        x1 = max(0, rx - pad)
        y1 = max(0, ry - pad)
        x2 = min(w_t, rx + rw + pad)
        y2 = min(h_t, ry + rh + pad)

        if x2 <= x1 or y2 <= y1:
            return img_target, len(tri_indices)

        output = img_target.copy()
        if "POISSON" in mode_up:
            full_mask = np.zeros((h_t, w_t), dtype=np.uint8)
            cv2.fillConvexPoly(full_mask, hull, 255)
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            full_mask = cv2.erode(full_mask, kernel_erode, iterations=1)
            center = (rx + rw // 2, ry + rh // 2)
            center = (max(1, min(w_t - 2, center[0])), max(1, min(h_t - 2, center[1])))
            try:
                swapped_full = cv2.seamlessClone(warped_src_on_dst, img_target, full_mask, center, cv2.NORMAL_CLONE)
                if alpha_f < 0.99:
                    output = cv2.addWeighted(img_target, 1.0 - alpha_f, swapped_full, alpha_f, 0)
                else:
                    output = swapped_full
            except Exception:
                mode_up = "LAB_FEATHER"

        if "POISSON" not in mode_up:
            roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            cv2.fillConvexPoly(roi_mask, hull - np.array([x1, y1], dtype=np.int32), 255)
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            roi_mask_eroded = cv2.erode(roi_mask, kernel_erode, iterations=1)

            roi_warped = warped_src_on_dst[y1:y2, x1:x2]
            roi_target = img_target[y1:y2, x1:x2]
            roi_color_matched = self._reinhard_color_transfer_roi(roi_warped, roi_target, roi_mask_eroded)

            k_blur = max(15, (int(min(rw, rh) * 0.14) // 2) * 2 + 1)
            feather = cv2.GaussianBlur(roi_mask_eroded, (k_blur, k_blur), k_blur * 0.35)
            alpha_mask = (feather.astype(np.float32) / 255.0) * alpha_f
            alpha_3d = alpha_mask[:, :, np.newaxis]

            blended_roi = (1.0 - alpha_3d) * roi_target.astype(np.float32) + alpha_3d * roi_color_matched.astype(np.float32)
            output[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)

        if show_delaunay:
            for i1, i2, i3 in tri_indices:
                pt1 = (int(pts_dst[i1][0]), int(pts_dst[i1][1]))
                pt2 = (int(pts_dst[i2][0]), int(pts_dst[i2][1]))
                pt3 = (int(pts_dst[i3][0]), int(pts_dst[i3][1]))
                cv2.line(output, pt1, pt2, (0, 242, 254), 1, cv2.LINE_AA)
                cv2.line(output, pt2, pt3, (0, 242, 254), 1, cv2.LINE_AA)
                cv2.line(output, pt3, pt1, (0, 242, 254), 1, cv2.LINE_AA)

        return output, len(tri_indices)

