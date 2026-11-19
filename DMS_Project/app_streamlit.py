"""
HỆ THỐNG GIÁM SÁT TÀI XẾ THÔNG MINH & ĐA ỨNG DỤNG MỐC HÌNH HỌC KHUÔN MẶT
(DRIVER MONITORING SYSTEM & FACE LANDMARK MULTI-APPLICATIONS)
Học phần: Xử lý ảnh số (Digital Image Processing) - ĐH Tôn Đức Thắng
Giảng viên hướng dẫn: TS. Phạm Văn Huy

Phiên bản tối ưu hóa toàn diện V16 (Cyberpunk HUD Redesign, Zero Icons, Zero Stray Divs & Multi-Threaded Camera):
- Tuân thủ tuyệt đối tiêu chuẩn kỹ thuật: Không sử dụng bất kỳ ký tự icon hay emoji nào trong toàn bộ mã nguồn.
- Chuẩn hóa 100% cấu trúc thẻ HTML qua bộ nén dòng clean_html(), triệt tiêu hoàn toàn lỗi rò rỉ thẻ HTML
  do trình phân tích CommonMark của Streamlit gây ra khi thụt lề nhiều dòng.
- Giao diện Cyberpunk ADAS HUD thế hệ mới: Tích hợp hiệu ứng 7 màu Gravity, tia quét Laser Scanline,
  đèn xung nhịp CSS thuần, thanh đo lường trực quan (Gauge Bar), bộ bước thử thách Liveness 3 giai đoạn
  và màn hình chờ Radar quang học động.
- Luồng đọc Camera bất đồng bộ thực thụ (True Background Grabber Thread) kết hợp truyền ảnh Base64 trực tiếp
  qua WebSocket giúp đạt tốc độ thời gian thực mượt mà và không bao giờ vỡ ảnh (Zero 404).
"""

import os
import sys
import time
import math
import base64
import tempfile
import threading
import subprocess
import warnings
warnings.filterwarnings("ignore")

# 1. Triệt tiêu ngoại lệ mạng WinError 10054 khi đóng / chuyển tab trên Windows
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _patched_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                pass
            except Exception:
                pass

        _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
    except Exception:
        pass

# 2. Triệt tiêu các thông báo C++ MediaPipe, TensorFlow Lite và cảnh báo thư viện
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

# Tự động gọi Streamlit run nếu người dùng chạy bằng "python app_streamlit.py"
if __name__ == "__main__" and not any("streamlit" in arg for arg in sys.argv):
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    sys.exit(0)

import cv2
import numpy as np
import streamlit as st
from dms_engine import DMSEngine, apply_camera_scale, suppress_c_stderr
from audio_manager import AudioManager
from glasses_generator import get_glasses_dict

# Hỗ trợ WebRTC Video Live Streaming từ xa cho Client Browser
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
except ImportError:
    HAS_WEBRTC = False
    webrtc_streamer = None
    WebRtcMode = None
    RTCConfiguration = None
    RTC_CONFIGURATION = None

# -------------------------------------------------------------
# CẤU HÌNH TRANG & THEME TỐI CHUYÊN NGHIỆP
# -------------------------------------------------------------
st.set_page_config(
    page_title="DMS & Face Landmark Applications - TDTU DIP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# BỘ CHUẨN HÓA HTML & KẾT XUẤT KHUNG HÌNH KHÔNG LỖI (ZERO STRAY DIV / ZERO 404)
# -------------------------------------------------------------
def clean_html(raw_html):
    """
    Nén chuỗi HTML nhiều dòng thành một dòng chuẩn xác, loại bỏ hoàn toàn khoảng trắng đầu dòng
    và dòng trống để bộ phân tích Markdown (CommonMark) của Streamlit không bao giờ hiểu nhầm
    thẻ đóng HTML là khối code thụt lề (indented code block).
    """
    if not raw_html:
        return ""
    return " ".join(line.strip() for line in str(raw_html).splitlines() if line.strip())


def encode_bgr_to_b64(img_bgr, quality=85):
    """Mã hóa mảng ảnh BGR sang chuỗi Base64 JPEG siêu tốc."""
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return ""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def safe_display_image(placeholder, img_or_bytes, caption=None, max_width=None):
    """
    Hiển thị khung hình video / ảnh trực tiếp bằng Data URI Base64 qua WebSocket.
    Đóng gói trong khung HUD hiện đại với thẻ HTML cân bằng tuyệt đối.
    """
    if img_or_bytes is None:
        return

    b64_str = ""
    if isinstance(img_or_bytes, (bytes, bytearray)):
        b64_str = base64.b64encode(img_or_bytes).decode("ascii")
    elif isinstance(img_or_bytes, str):
        b64_str = img_or_bytes
    elif isinstance(img_or_bytes, np.ndarray) and img_or_bytes.size > 0:
        ok, buf = cv2.imencode(".jpg", img_or_bytes, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            b64_str = base64.b64encode(buf.tobytes()).decode("ascii")

    if not b64_str:
        return

    width_style = f"max-width: {int(max_width)}px; width: 100%;" if max_width else "width: 100%;"
    caption_block = (
        f'<div class="viewport-caption">{caption}</div>'
        if caption else ""
    )

    html_out = clean_html(f"""
        <div class="viewport-hud-frame" style="{width_style}">
            <div class="viewport-topbar">
                <span class="live-dot"></span>
                <span class="viewport-topbar-title">OPTICAL SENSOR STREAM // REAL-TIME DIP</span>
                <span class="viewport-topbar-tag">ACTIVE</span>
            </div>
            <div class="viewport-img-wrap">
                <img src="data:image/jpeg;base64,{b64_str}" class="viewport-img" alt="Viewport Stream" />
            </div>
            {caption_block}
        </div>
    """)
    placeholder.markdown(html_out, unsafe_allow_html=True)


def render_aligned_face_card(placeholder, b64_img, caption="Ảnh chân dung đã xoay căn chỉnh Affine (256x256)", width=220):
    """
    Hiển thị ảnh chân dung 256x256 đã chuẩn hóa Affine bằng Base64 trực tiếp,
    kèm khung quét sinh trắc học Cyberpunk động.
    """
    if not b64_img:
        return
    html_out = clean_html(f"""
        <div class="bio-crop-card">
            <div class="bio-crop-header">
                <span class="bio-dot"></span>
                <span>CHUẨN HÓA AFFINE 256x256</span>
            </div>
            <div class="bio-crop-img-box" style="width: {int(width)}px; height: {int(width)}px;">
                <img src="data:image/jpeg;base64,{b64_img}" style="width: 100%; height: 100%; object-fit: cover; display: block; border-radius: 8px;" alt="Aligned Face 256x256" />
                <div class="bio-scan-bar"></div>
            </div>
            <div class="bio-crop-caption">{caption}</div>
        </div>
    """)
    placeholder.markdown(html_out, unsafe_allow_html=True)


def build_metric_card(title, value, sub, pct=None, bar_color="#00f2fe", badge_text=None):
    """Tạo thẻ chỉ số Telemetry có thanh đo trực quan (Gauge Bar) và không bao giờ rò rỉ thẻ div."""
    badge_html = f'<span class="metric-badge">{badge_text}</span>' if badge_text else ""
    bar_html = ""
    if pct is not None:
        clamped_pct = max(0.0, min(100.0, float(pct)))
        bar_html = (
            f'<div class="metric-bar-bg">'
            f'<div class="metric-bar-fill" style="width: {clamped_pct:.1f}%; background: {bar_color}; box-shadow: 0 0 8px {bar_color};"></div>'
            f'</div>'
        )
    return clean_html(f"""
        <div class="metric-card">
            <div class="metric-header-row">
                <span class="metric-title">{title}</span>
                {badge_html}
            </div>
            <div class="metric-value">{value}</div>
            {bar_html}
            <div class="metric-sub">{sub}</div>
        </div>
    """)


def build_status_banner(level, text):
    """Tạo hộp cảnh báo trạng thái có đèn xung nhịp CSS thuần."""
    cls_map = {
        "safe": "alert-box-safe",
        "warn": "alert-box-warn",
        "danger": "alert-box-danger"
    }
    dot_map = {
        "safe": "status-dot-safe",
        "warn": "status-dot-warn",
        "danger": "status-dot-danger"
    }
    box_cls = cls_map.get(level, "alert-box-safe")
    dot_cls = dot_map.get(level, "status-dot-safe")
    return clean_html(f"""
        <div class="{box_cls}">
            <span class="{dot_cls}"></span>
            <span>{text}</span>
        </div>
    """)


def build_liveness_stepper(stage, active_toggle, detail_title, detail_sub):
    """Tạo giao diện 3 bước thử thách Liveness trực quan kèm hiệu ứng sáng đèn từng bước."""
    steps = [
        (0, "BƯỚC 1", "Chớp mắt (2 lần)"),
        (1, "BƯỚC 2", "Nghiêng mặt (>15°)"),
        (2, "BƯỚC 3", "Giữ thẳng mặt")
    ]
    step_items = []
    for idx, code, name in steps:
        if stage > idx or stage == 3:
            st_cls = "lv-step lv-step-done"
            st_tag = "HOÀN TẤT"
        elif stage == idx and active_toggle:
            st_cls = "lv-step lv-step-active"
            st_tag = "ĐANG THỰC HIỆN"
        else:
            st_cls = "lv-step"
            st_tag = "CHỜ"
        step_items.append(
            f'<div class="{st_cls}">'
            f'<div class="lv-step-top"><span class="lv-step-code">{code}</span><span class="lv-step-status">{st_tag}</span></div>'
            f'<div class="lv-step-name">{name}</div>'
            f'</div>'
        )
    steps_html = "".join(step_items)
    banner_cls = "lv-challenge-box lv-challenge-done" if stage == 3 else (
        "lv-challenge-box lv-challenge-active" if active_toggle else "lv-challenge-box"
    )
    return clean_html(f"""
        <div class="lv-panel-wrap">
            <div class="lv-stepper-grid">{steps_html}</div>
            <div class="{banner_cls}">
                <div class="lv-challenge-title">{detail_title}</div>
                <div class="lv-challenge-sub">{detail_sub}</div>
            </div>
        </div>
    """)


def safe_button(label, key=None, **kwargs):
    """Tạo nút bấm an toàn với mọi phiên bản Streamlit."""
    try:
        return st.button(label, key=key, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, **kwargs)

# -------------------------------------------------------------
# CSS NÂNG CAO: CYBERPUNK HUD, GLASSMORPHISM & ANIMATION ĐA TẦNG
# -------------------------------------------------------------
st.markdown(clean_html("""
<style>
    /* 1. Header chuẩn */
    header[data-testid="stHeader"], .stAppHeader {
        background: rgba(8, 12, 22, 0.92) !important;
        backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(0, 242, 254, 0.22) !important;
    }
    header[data-testid="stHeader"] * {
        color: #e2e8f0 !important;
    }

    /* 2. Nền tảng tổng thể có lưới vi mạch mờ */
    .stApp {
        background-color: #070b14;
        background-image:
            radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.06) 0%, transparent 35%),
            radial-gradient(circle at 85% 80%, rgba(155, 81, 224, 0.06) 0%, transparent 35%),
            linear-gradient(rgba(0, 242, 254, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 242, 254, 0.025) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* 3. ANIMATION CHUYỂN ĐỔI 7 MÀU GRAVITY & TIA QUÉT LASER CHO KHUNG TIÊU ĐỀ */
    @keyframes gravity_rainbow_shift {
        0% {
            background-position: 0% 50%;
            filter: drop-shadow(0 0 12px rgba(255, 8, 68, 0.55));
        }
        25% {
            background-position: 50% 100%;
            filter: drop-shadow(0 0 16px rgba(0, 242, 254, 0.65));
        }
        50% {
            background-position: 100% 50%;
            filter: drop-shadow(0 0 18px rgba(0, 230, 118, 0.65));
        }
        75% {
            background-position: 50% 0%;
            filter: drop-shadow(0 0 16px rgba(255, 222, 0, 0.65));
        }
        100% {
            background-position: 0% 50%;
            filter: drop-shadow(0 0 12px rgba(255, 8, 68, 0.55));
        }
    }

    @keyframes gravity_rainbow_border {
        0% {
            border-color: #ff0844;
            box-shadow: 0 0 22px rgba(255, 8, 68, 0.55), inset 0 0 15px rgba(255, 8, 68, 0.15);
        }
        20% {
            border-color: #ff7b00;
            box-shadow: 0 0 22px rgba(255, 123, 0, 0.55), inset 0 0 15px rgba(255, 123, 0, 0.15);
        }
        40% {
            border-color: #ffde00;
            box-shadow: 0 0 22px rgba(255, 222, 0, 0.55), inset 0 0 15px rgba(255, 222, 0, 0.15);
        }
        60% {
            border-color: #00e676;
            box-shadow: 0 0 22px rgba(0, 230, 118, 0.55), inset 0 0 15px rgba(0, 230, 118, 0.15);
        }
        80% {
            border-color: #00f2fe;
            box-shadow: 0 0 22px rgba(0, 242, 254, 0.65), inset 0 0 15px rgba(0, 242, 254, 0.18);
        }
        100% {
            border-color: #ff0844;
            box-shadow: 0 0 22px rgba(255, 8, 68, 0.55), inset 0 0 15px rgba(255, 8, 68, 0.15);
        }
    }

    @keyframes header_laser_scan {
        0% { left: -35%; opacity: 0; }
        20% { opacity: 0.85; }
        80% { opacity: 0.85; }
        100% { left: 115%; opacity: 0; }
    }

    .header-container {
        background: linear-gradient(135deg, rgba(13, 20, 38, 0.96), rgba(8, 12, 24, 0.98)) !important;
        border: 2.5px solid #00f2fe !important;
        border-radius: 16px !important;
        padding: 20px 26px 18px 26px !important;
        text-align: center !important;
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(12px) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        animation: gravity_rainbow_border 7s ease-in-out infinite !important;
        margin-bottom: 18px !important;
    }

    .header-container::after {
        content: "";
        position: absolute;
        top: 0;
        left: -35%;
        width: 30%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.14), transparent);
        transform: skewX(-25deg);
        animation: header_laser_scan 5.5s ease-in-out infinite;
        pointer-events: none;
    }

    .header-top-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.35);
        border-radius: 999px;
        padding: 3px 14px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #00f2fe;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .header-gravity-title {
        font-size: 2.0rem !important;
        font-weight: 900 !important;
        text-align: center !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        margin: 4px 0 8px 0 !important;
        line-height: 1.28 !important;
        background: linear-gradient(
            135deg,
            #ff0844 0%,
            #ff7b00 16.6%,
            #ffde00 33.3%,
            #00e676 50%,
            #00f2fe 66.6%,
            #3a7bd5 83.3%,
            #9b51e0 100%
        ) !important;
        background-size: 300% 300% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        animation: gravity_rainbow_shift 6s ease-in-out infinite !important;
    }

    .header-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 2px;
        letter-spacing: 0.3px;
    }
    .header-subtitle b {
        color: #38bdf8;
    }

    /* 4. ĐỊNH DẠNG THANH TAB ĐIỀU HƯỚNG CHÍNH (CYBERPUNK TAB BAR) */
    @keyframes tab_active_pulse {
        0%, 100% { box-shadow: 0 0 12px rgba(0, 242, 254, 0.35), inset 0 0 8px rgba(0, 242, 254, 0.15); }
        50% { box-shadow: 0 0 20px rgba(0, 242, 254, 0.65), inset 0 0 12px rgba(0, 242, 254, 0.28); }
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        background: rgba(12, 19, 36, 0.92) !important;
        padding: 10px 14px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0, 242, 254, 0.28) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stRadio"] label {
        background: rgba(24, 34, 56, 0.65) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: rgba(0, 242, 254, 0.65) !important;
        background: rgba(0, 242, 254, 0.12) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label p {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        margin: 0 !important;
        letter-spacing: 0.2px !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.22) 0%, rgba(58, 123, 213, 0.32) 100%) !important;
        border: 1.5px solid #00f2fe !important;
        animation: tab_active_pulse 2.8s infinite ease-in-out !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #00f2fe !important;
        font-weight: 800 !important;
    }

    /* 5. HỘP MÔ TẢ THUẬT TOÁN DIP (DIP THEORY BANNER) */
    .dip-theory-card {
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.92) 0%, rgba(15, 28, 52, 0.78) 100%);
        border: 1px solid rgba(0, 242, 254, 0.24);
        border-left: 4px solid #00f2fe;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .dip-theory-header {
        font-size: 12px;
        font-weight: 800;
        color: #00f2fe;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .dip-theory-list {
        margin: 0;
        padding-left: 18px;
        color: #cbd5e1;
        font-size: 13.5px;
        line-height: 1.6;
    }
    .dip-theory-list b {
        color: #38bdf8;
    }

    /* 6. KHUNG VIEWPORT CAMERA & STANDBY RADAR HUD */
    @keyframes dot_pulse {
        0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 8px #00e676; }
        50% { transform: scale(1.3); opacity: 0.55; box-shadow: 0 0 14px #00e676; }
    }

    .viewport-hud-frame {
        margin: 0 auto;
        background: #060a12;
        border: 1.5px solid rgba(0, 242, 254, 0.35);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.55), 0 0 18px rgba(0, 242, 254, 0.1);
        transition: border-color 0.3s ease;
    }
    .viewport-topbar {
        display: flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.95), rgba(10, 16, 30, 0.95));
        border-bottom: 1px solid rgba(0, 242, 254, 0.2);
        padding: 8px 14px;
    }
    .live-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #00e676;
        display: inline-block;
        animation: dot_pulse 1.6s infinite ease-in-out;
    }
    .viewport-topbar-title {
        font-size: 11.5px;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 1px;
        flex: 1;
        font-family: "Courier New", Courier, monospace;
    }
    .viewport-topbar-tag {
        font-size: 10.5px;
        font-weight: 800;
        color: #00f2fe;
        background: rgba(0, 242, 254, 0.12);
        border: 1px solid rgba(0, 242, 254, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.8px;
    }
    .viewport-img-wrap {
        position: relative;
        width: 100%;
        background: #050811;
    }
    .viewport-img {
        width: 100%;
        height: auto;
        display: block;
        object-fit: contain;
        transform: translateZ(0);
        backface-visibility: hidden;
    }
    .viewport-caption {
        color: #94a3b8;
        font-size: 12.5px;
        padding: 8px 12px;
        text-align: center;
        font-weight: 600;
        background: rgba(12, 18, 32, 0.95);
        border-top: 1px solid rgba(0, 242, 254, 0.15);
    }

    /* Màn hình chờ Radar Quang học */
    @keyframes radar_spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes standby_glow {
        0%, 100% { border-color: rgba(0, 242, 254, 0.3); box-shadow: 0 0 15px rgba(0, 242, 254, 0.08); }
        50% { border-color: rgba(0, 242, 254, 0.65); box-shadow: 0 0 28px rgba(0, 242, 254, 0.22); }
    }
    .standby-hud-box {
        background: radial-gradient(circle at center, rgba(16, 26, 48, 0.85) 0%, rgba(9, 14, 26, 0.95) 100%);
        border: 1.5px dashed rgba(0, 242, 254, 0.4);
        border-radius: 14px;
        padding: 46px 24px;
        text-align: center;
        color: #94a3b8;
        position: relative;
        overflow: hidden;
        animation: standby_glow 3.5s ease-in-out infinite;
    }
    .radar-ring {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        border: 2px solid rgba(0, 242, 254, 0.25);
        border-top: 2.5px solid #00f2fe;
        border-right: 2.5px solid rgba(0, 230, 118, 0.8);
        margin: 0 auto 16px auto;
        animation: radar_spin 2.4s linear infinite;
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.25);
    }
    .standby-title {
        font-size: 16px;
        font-weight: 800;
        color: #00f2fe;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .standby-desc {
        font-size: 13.5px;
        line-height: 1.65;
        color: #cbd5e1;
        max-width: 520px;
        margin: 0 auto;
    }

    /* 7. THẺ SỐ LIỆU TELEMETRY KÈM GAUGE BAR ĐỘNG */
    .metric-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(11, 18, 34, 0.92));
        border: 1px solid rgba(0, 242, 254, 0.22);
        border-left: 3.5px solid #00f2fe;
        border-radius: 12px;
        padding: 13px 16px;
        margin-bottom: 11px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.38);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 242, 254, 0.6);
        transform: translateX(3px);
        box-shadow: 0 8px 22px rgba(0, 242, 254, 0.14);
    }
    .metric-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .metric-title {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        font-weight: 700;
    }
    .metric-badge {
        font-size: 10px;
        font-weight: 700;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 1px 7px;
        border-radius: 999px;
    }
    .metric-value {
        font-size: 21px;
        font-weight: 800;
        color: #00f2fe;
        font-family: "Courier New", Courier, monospace;
        letter-spacing: 0.5px;
    }
    .metric-bar-bg {
        width: 100%;
        height: 5px;
        background: rgba(30, 41, 59, 0.85);
        border-radius: 999px;
        overflow: hidden;
        margin: 6px 0 5px 0;
    }
    .metric-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.25s ease, background-color 0.25s ease;
    }
    .metric-sub {
        font-size: 11.5px;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* 8. HỘP CẢNH BÁO TRẠNG THÁI (SAFE / WARN / DANGER) */
    @keyframes pulse_danger_box {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.65); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .alert-box-safe {
        display: flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.18), rgba(16, 185, 129, 0.06));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 4px solid #10b981;
        padding: 13px 16px;
        border-radius: 10px;
        font-weight: 700;
        color: #34d399;
        font-size: 13.5px;
        margin-bottom: 14px;
    }
    .alert-box-warn {
        display: flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.22), rgba(245, 158, 11, 0.08));
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-left: 4px solid #f59e0b;
        padding: 13px 16px;
        border-radius: 10px;
        font-weight: 700;
        color: #fbbf24;
        font-size: 13.5px;
        margin-bottom: 14px;
    }
    .alert-box-danger {
        display: flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.28), rgba(239, 68, 68, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.6);
        border-left: 4px solid #ef4444;
        padding: 13px 16px;
        border-radius: 10px;
        font-weight: 800;
        color: #f87171;
        font-size: 13.5px;
        margin-bottom: 14px;
        animation: pulse_danger_box 1.2s infinite;
    }
    .status-dot-safe {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 10px #10b981;
        flex-shrink: 0;
    }
    .status-dot-warn {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #f59e0b;
        box-shadow: 0 0 10px #f59e0b;
        flex-shrink: 0;
    }
    .status-dot-danger {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #ef4444;
        box-shadow: 0 0 12px #ef4444;
        flex-shrink: 0;
    }

    /* 9. GIAO DIỆN STEPPER 3 BƯỚC & KHUNG QUÉT SINH TRẮC HỌC TAB 2 */
    @keyframes bio_scan_move {
        0% { top: 0%; opacity: 0.9; }
        50% { top: 92%; opacity: 0.9; }
        100% { top: 0%; opacity: 0.9; }
    }
    .lv-panel-wrap {
        margin: 10px 0 12px 0;
    }
    .lv-stepper-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 12px;
    }
    .lv-step {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        padding: 9px 10px;
        transition: all 0.25s ease;
    }
    .lv-step-active {
        background: rgba(0, 242, 254, 0.14);
        border: 1.5px solid #00f2fe;
        box-shadow: 0 0 14px rgba(0, 242, 254, 0.28);
    }
    .lv-step-done {
        background: rgba(16, 185, 129, 0.16);
        border: 1.5px solid #10b981;
    }
    .lv-step-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .lv-step-code {
        font-size: 10px;
        font-weight: 800;
        color: #00f2fe;
        letter-spacing: 0.6px;
    }
    .lv-step-done .lv-step-code {
        color: #34d399;
    }
    .lv-step-status {
        font-size: 9.5px;
        font-weight: 700;
        color: #94a3b8;
    }
    .lv-step-active .lv-step-status {
        color: #00f2fe;
    }
    .lv-step-done .lv-step-status {
        color: #34d399;
    }
    .lv-step-name {
        font-size: 12px;
        font-weight: 700;
        color: #e2e8f0;
    }
    .lv-challenge-box {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .lv-challenge-active {
        border-color: #00f2fe;
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.18);
    }
    .lv-challenge-done {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.12);
    }
    .lv-challenge-title {
        font-size: 15px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .lv-challenge-sub {
        font-size: 12.5px;
        color: #38bdf8;
        font-family: "Courier New", Courier, monospace;
        font-weight: 600;
    }

    .bio-crop-card {
        margin-top: 12px;
        display: inline-block;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(10, 16, 30, 0.98));
        border: 1.5px solid #00e676;
        border-radius: 14px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 0 22px rgba(0, 230, 118, 0.25);
    }
    .bio-crop-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 800;
        color: #00e676;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .bio-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00e676;
        box-shadow: 0 0 8px #00e676;
    }
    .bio-crop-img-box {
        position: relative;
        margin: 0 auto;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(0, 242, 254, 0.4);
    }
    .bio-scan-bar {
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent, #00f2fe, #00e676, transparent);
        box-shadow: 0 0 10px #00f2fe;
        animation: bio_scan_move 2.8s ease-in-out infinite;
        pointer-events: none;
    }
    .bio-crop-caption {
        color: #cbd5e1;
        font-size: 12px;
        margin-top: 8px;
        font-weight: 600;
    }

    /* 10. THANH SÓNG ÂM THANH TRÌNH DUYỆT (PURE CSS EQUALIZER BARS) */
    @keyframes eq_bar_anim {
        0%, 100% { height: 6px; }
        50% { height: 18px; }
    }
    .audio-alert-banner {
        background: linear-gradient(90deg, rgba(255, 8, 68, 0.24), rgba(15, 23, 42, 0.92));
        border: 1.5px solid #ff0844;
        border-radius: 10px;
        padding: 8px 16px;
        margin: 10px 0 14px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 0 18px rgba(255, 8, 68, 0.38);
    }
    .audio-alert-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .eq-bars {
        display: inline-flex;
        align-items: flex-end;
        gap: 3px;
        height: 18px;
    }
    .eq-bar {
        width: 3.5px;
        background: #ffde00;
        border-radius: 2px;
        animation: eq_bar_anim 0.6s ease-in-out infinite;
    }
    .eq-bar:nth-child(2) { animation-delay: 0.18s; background: #00f2fe; }
    .eq-bar:nth-child(3) { animation-delay: 0.35s; background: #ff0844; }
    .audio-alert-text {
        color: #ffde00;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.3px;
    }

    /* 11. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #0b1222 !important;
        border-right: 1px solid rgba(0, 242, 254, 0.18) !important;
    }
    .section-panel-title {
        font-size: 13px;
        font-weight: 800;
        color: #00f2fe;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(0, 242, 254, 0.22);
        margin-bottom: 12px;
    }
</style>
"""), unsafe_allow_html=True)

# -------------------------------------------------------------
# KHỞI TẠO SINGLETON ENGINE & QUẢN LÝ BỘ NHỚ ĐỆM
# -------------------------------------------------------------
@st.cache_resource
def get_dms_engine():
    with suppress_c_stderr():
        return DMSEngine()

@st.cache_resource
def get_audio_manager():
    return AudioManager()

@st.cache_resource
def get_glasses_database():
    return get_glasses_dict()

engine = get_dms_engine()
audio = get_audio_manager()

# Khởi tạo session state
if "glasses_db" not in st.session_state:
    st.session_state.glasses_db = get_glasses_database()
if "selected_glasses_choice" not in st.session_state:
    st.session_state.selected_glasses_choice = list(st.session_state.glasses_db.keys())[0]

# Session state cho Tab 4 (Makeup)
if "makeup_lip_color" not in st.session_state:
    st.session_state.makeup_lip_color = "Hồng Cherry Tươi Trẻ"
if "makeup_lip_alpha" not in st.session_state:
    st.session_state.makeup_lip_alpha = 0.55
if "makeup_enable_blush" not in st.session_state:
    st.session_state.makeup_enable_blush = True
if "makeup_blush_alpha" not in st.session_state:
    st.session_state.makeup_blush_alpha = 0.35
if "makeup_color_space" not in st.session_state:
    st.session_state.makeup_color_space = "HYBRID (Kết hợp CIELAB + HLS)"

palette_lip = {
    "Đỏ Ruby Cổ Điển": (20, 15, 235),
    "Hồng Cherry Tươi Trẻ": (150, 30, 255),
    "Cam Đất Thời Thượng": (30, 110, 250),
    "Đỏ Mận Quyến Rũ": (55, 10, 160),
    "Hồng San Hô Ngọt Ngào": (80, 80, 255)
}

palette_blush = {
    "Hồng Baby Xinh Xắn": (160, 110, 255),
    "Cam Đào Rạng Rỡ": (80, 140, 255),
    "Đỏ Hồng Quyến Rũ": (80, 60, 245)
}
if "makeup_blush_color" not in st.session_state:
    st.session_state.makeup_blush_color = "Hồng Baby Xinh Xắn"

# Session state cho Tab 2 (Liveness)
if "liveness_stage" not in st.session_state:
    st.session_state.liveness_stage = 0
if "blink_completed_count" not in st.session_state:
    st.session_state.blink_completed_count = 0
if "blink_closed_frames" not in st.session_state:
    st.session_state.blink_closed_frames = 0
if "turn_counter_liveness" not in st.session_state:
    st.session_state.turn_counter_liveness = 0
if "straight_counter_liveness" not in st.session_state:
    st.session_state.straight_counter_liveness = 0
if "liveness_success" not in st.session_state:
    st.session_state.liveness_success = False
if "liveness_aligned_face" not in st.session_state:
    st.session_state.liveness_aligned_face = None
if "liveness_aligned_b64" not in st.session_state:
    st.session_state.liveness_aligned_b64 = ""
if "liveness_crop_rendered" not in st.session_state:
    st.session_state.liveness_crop_rendered = False
if "liveness_active_toggle" not in st.session_state:
    st.session_state.liveness_active_toggle = False
if "liveness_last_center" not in st.session_state:
    st.session_state.liveness_last_center = None
if "liveness_lost_frames" not in st.session_state:
    st.session_state.liveness_lost_frames = 0
if "liveness_laplacian_score" not in st.session_state:
    st.session_state.liveness_laplacian_score = 0.0
if "liveness_spoof_warn" not in st.session_state:
    st.session_state.liveness_spoof_warn = ""

def reset_liveness_fsm():
    st.session_state.liveness_stage = 0
    st.session_state.blink_completed_count = 0
    st.session_state.blink_closed_frames = 0
    st.session_state.turn_counter_liveness = 0
    st.session_state.straight_counter_liveness = 0
    st.session_state.liveness_success = False
    st.session_state.liveness_aligned_face = None
    st.session_state.liveness_aligned_b64 = ""
    st.session_state.liveness_crop_rendered = False
    st.session_state.liveness_last_center = None
    st.session_state.liveness_lost_frames = 0
    st.session_state.liveness_spoof_warn = "" 

# Session state cho Tab 5 (Gaze)
if "gaze_selected_card" not in st.session_state:
    st.session_state.gaze_selected_card = "THẺ TRUNG TÂM"
if "gaze_last_clicked_card" not in st.session_state:
    st.session_state.gaze_last_clicked_card = "Chưa có thao tác nhấp"
if "gaze_click_hold" not in st.session_state:
    st.session_state.gaze_click_hold = 0
if "gaze_blink_frames" not in st.session_state:
    st.session_state.gaze_blink_frames = 0

# Session state cho Tab 6 (Face Morphing & Face Swapping)
if "morph_alpha" not in st.session_state:
    st.session_state.morph_alpha = 1.0
if "morph_blend_mode" not in st.session_state:
    st.session_state.morph_blend_mode = "LAB_FEATHER (Reinhard CIELAB + Gaussian Feather - 30 FPS)"
if "morph_show_delaunay" not in st.session_state:
    st.session_state.morph_show_delaunay = False
if "morph_source_choice" not in st.session_state:
    st.session_state.morph_source_choice = "Khuôn mặt mẫu B (Nữ - Chân dung chuẩn)"
if "morph_custom_src_bgr" not in st.session_state:
    st.session_state.morph_custom_src_bgr = None
if "morph_cached_templates" not in st.session_state:
    st.session_state.morph_cached_templates = {}

MORPH_SOURCE_PRESETS = {
    "Khuôn mặt mẫu B (Nữ - Chân dung chuẩn)": "test_images/Tab6_FaceMorph_FaceSwap/02_KhuonMat_Mau_B_Nu.jpg",
    "Khuôn mặt mẫu A (Nam - Chân dung chuẩn)": "test_images/Tab6_FaceMorph_FaceSwap/01_KhuonMat_Mau_A_Nam.jpg",
    "Khuôn mặt mẫu C (Tài xế khoang lái)": "test_images/Tab6_FaceMorph_FaceSwap/03_KhuonMat_Mau_C_TaiXe.jpg"
}

def get_morph_source_data():
    choice = st.session_state.get("morph_source_choice", "Khuôn mặt mẫu B (Nữ - Chân dung chuẩn)")
    if choice == "Tải ảnh khuôn mặt nguồn riêng" and st.session_state.get("morph_custom_src_bgr") is not None:
        src_img = st.session_state.morph_custom_src_bgr
        cache_key = f"custom_{src_img.shape}_{int(src_img[::32, ::32].sum())}"
    else:
        rel_p = MORPH_SOURCE_PRESETS.get(choice, "test_images/Tab6_FaceMorph_FaceSwap/02_KhuonMat_Mau_B_Nu.jpg")
        full_p = os.path.join(os.path.dirname(__file__), rel_p)
        if not os.path.exists(full_p):
            full_p = rel_p
        cache_key = f"preset_{choice}"
        if cache_key in st.session_state.morph_cached_templates:
            return st.session_state.morph_cached_templates[cache_key]
        src_img = cv2.imread(full_p) if os.path.exists(full_p) else None

    if src_img is None or src_img.size == 0:
        return None, None
    if cache_key in st.session_state.morph_cached_templates:
        return st.session_state.morph_cached_templates[cache_key]

    sh, sw = src_img.shape[:2]
    if sw > 640:
        src_img = cv2.resize(src_img, (640, int(sh * (640.0 / sw))), interpolation=cv2.INTER_LINEAR)
    src_lm = engine.extract_single_face_landmarks(src_img)
    st.session_state.morph_cached_templates[cache_key] = (src_img, src_lm)
    return src_img, src_lm

# Session state cho Camera Persistent
if "cap" not in st.session_state:
    st.session_state.cap = None
if "active_source_type" not in st.session_state:
    st.session_state.active_source_type = None
if "active_cam_idx" not in st.session_state:
    st.session_state.active_cam_idx = None
if "active_cam_drv" not in st.session_state:
    st.session_state.active_cam_drv = None
if "last_pitch" not in st.session_state:
    st.session_state.last_pitch = 0.0
if "last_yaw" not in st.session_state:
    st.session_state.last_yaw = 0.0
if "last_roll" not in st.session_state:
    st.session_state.last_roll = 0.0
if "last_ear" not in st.session_state:
    st.session_state.last_ear = 0.28

# -------------------------------------------------------------
# TIÊU ĐỀ ỨNG DỤNG VỚI ANIMATION 7 MÀU GRAVITY & TIA LASER QUÉT NGANG
# (Đã loại bỏ thẻ div dư thừa, đóng/mở thẻ đối xứng 100%)
# -------------------------------------------------------------
st.markdown(clean_html("""
<div class="header-container">
    <div class="header-top-badge">
        <span class="live-dot"></span>
        <span>SYSTEM ONLINE // REAL-TIME DIP ENGINE</span>
    </div>
    <h1 class="header-gravity-title">HỆ THỐNG GIÁM SÁT TÀI XẾ &amp; ỨNG DỤNG MỐC HÌNH HỌC KHUÔN MẶT</h1>
    <div class="header-subtitle">Đề tài môn học: <b>Xử lý ảnh số (DIP)</b> &nbsp;|&nbsp; GVHD: <b>TS. Phạm Văn Huy</b> &nbsp;|&nbsp; Khoa CNTT - ĐH Tôn Đức Thắng</div>
</div>
"""), unsafe_allow_html=True)

# -------------------------------------------------------------
# THANH ĐIỀU HƯỚNG PHÂN HỆ THỐNG NHẤT (SINGLE SOURCE OF TRUTH)
# -------------------------------------------------------------
ACTIVE_MODULE_OPTIONS = [
    "[TAB 1] Giám sát buồn ngủ (DMS)",
    "[TAB 2] Điểm danh & Chống giả mạo",
    "[TAB 3] Thử kính thực tế ảo",
    "[TAB 4] Trang điểm son môi & má hồng",
    "[TAB 5] Tương tác qua ánh mắt",
    "[TAB 6] Biến hình & Hoán đổi mặt",
    "[TOÀN CẢNH] Cả 6 Phân hệ"
]

selected_module = st.radio(
    "CHỌN PHÂN HỆ ĐANG HOẠT ĐỘNG:",
    ACTIVE_MODULE_OPTIONS,
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_active_module"
)

is_tab1 = "[TAB 1]" in selected_module
is_tab2 = "[TAB 2]" in selected_module
is_tab3 = "[TAB 3]" in selected_module
is_tab4 = "[TAB 4]" in selected_module
is_tab5 = "[TAB 5]" in selected_module
is_tab6 = "[TAB 6]" in selected_module
is_all  = "[TOÀN CẢNH]" in selected_module

# -------------------------------------------------------------
# TRÌNH PHÁT ÂM THANH TRÌNH DUYỆT (BROWSER AUDIO CHO NGƯỜI DÙNG TỪ XA)
# -------------------------------------------------------------
browser_audio_holder = st.empty()

ALERT_DISPLAY_NAMES = {
    "drowsy": "Cảnh báo! Tài xế đang ngủ gật! Hãy tập trung lái xe!",
    "yawn": "Cảnh báo! Tài xế đang có biểu hiện ngủ gật! Hãy tập trung lái xe!",
    "distracted": "Cảnh báo! Tài xế đang mất tập trung! Hãy tập trung lái xe!",
    "dual_drowsy": "Cảnh báo! Phát hiện tài xế đang ngủ gật! Bạn đồng hành hãy chú ý hỗ trợ xử lý!",
    "success": "Xác thực danh tính KYC thành công!"
}

def trigger_system_alert(alert_type, force=False):
    """
    Kích hoạt âm thanh thông minh: Hỗ trợ 3 kênh phát:
    - 'os': Phát trực tiếp qua Loa máy tính / OS (winsound) - Cực nhanh, 0ms lag, không bao giờ bị trình duyệt chặn, TRIỆT TIÊU HOÀN TOÀN DOUBLE ÂM THANH!
    - 'browser': Phát qua Loa trình duyệt Web (HTML5 Audio) - Phù hợp khi người dùng truy cập từ xa qua điện thoại/mạng LAN.
    - 'both': Đồng bộ cả 2 kênh (chỉ bật khi máy chủ và người nghe ở 2 phòng khác nhau).
    """
    channel = st.session_state.get("audio_channel", "os")
    play_os = (channel in ["os", "both"])
    play_browser = (channel in ["browser", "both"])

    res = audio.trigger_alert(alert_type, force=force, play_os=play_os)

    if res:
        a_type, a_path = res
        if a_path and os.path.isfile(a_path):
            alert_title = ALERT_DISPLAY_NAMES.get(a_type, "Cảnh báo an toàn DMS")
            play_nonce = int(time.time() * 1000)
            st.session_state["browser_alert_active"] = True
            st.session_state["browser_alert_time"] = time.time()
            st.session_state["last_played_alert"] = alert_title

            if play_browser:
                b64_sound = audio.get_sound_base64(a_path)
                if b64_sound:
                    browser_audio_holder.markdown(clean_html(f"""
                        <div data-play-nonce="{play_nonce}" class="audio-alert-banner">
                            <div class="audio-alert-left">
                                <div class="eq-bars">
                                    <span class="eq-bar"></span>
                                    <span class="eq-bar"></span>
                                    <span class="eq-bar"></span>
                                </div>
                                <span class="audio-alert-text">[ÂM THANH CẢNH BÁO TRÌNH DUYỆT]: {alert_title}</span>
                            </div>
                            <audio id="dms_audio_{play_nonce}" autoplay controls style="height: 32px; max-width: 220px;">
                                <source src="data:audio/wav;base64,{b64_sound}" type="audio/wav">
                            </audio>
                            <script>
                                (function() {{
                                    var audios = document.querySelectorAll('audio');
                                    audios.forEach(function(el) {{
                                        if (el.id !== 'dms_audio_{play_nonce}') {{
                                            el.pause();
                                            el.currentTime = 0;
                                        }}
                                    }});
                                }})();
                            </script>
                        </div>
                    """), unsafe_allow_html=True)
            else:
                # Kênh Loa máy tính trực tiếp (OS Sound):
                # Hiển thị Banner trực quan có EQ động và dòng trạng thái rõ ràng,
                # TUYỆT ĐỐI KHÔNG chèn thẻ <audio autoplay> để tránh hiện tượng double âm thanh!
                browser_audio_holder.markdown(clean_html(f"""
                    <div data-play-nonce="{play_nonce}" class="audio-alert-banner">
                        <div class="audio-alert-left">
                            <div class="eq-bars">
                                <span class="eq-bar"></span>
                                <span class="eq-bar"></span>
                                <span class="eq-bar"></span>
                            </div>
                            <span class="audio-alert-text">[ÂM THANH CẢNH BÁO]: {alert_title}</span>
                        </div>
                        <span style="font-size: 11.5px; color: #10b981; font-weight: 700; padding: 4px 10px; background: rgba(16,185,129,0.18); border-radius: 6px; border: 1px solid #10b981;">[LOA MÁY TÍNH - KHÔNG ECHO]</span>
                    </div>
                """), unsafe_allow_html=True)
    return res

# -------------------------------------------------------------
# THANH ĐIỀU KHIỂN & MỞ KHÓA ÂM THANH TRÌNH DUYỆT (BROWSER AUDIO UNLOCK BAR)
# -------------------------------------------------------------
with st.expander(" TRÌNH ĐIỀU KHIỂN ÂM THANH TRÌNH DUYỆT (BẤM ĐỂ MỞ QUYỀN LOA / TEST CÒI TỪ XA)", expanded=False):
    st.markdown(clean_html("""
        <div style="font-size: 12.5px; color: #cbd5e1; margin-bottom: 8px;">
            <b>Lưu ý chính sách bảo mật trình duyệt (Chrome, Safari, Edge, Cốc Cốc, Zalo):</b><br>
            Trình duyệt mặc định chặn phát âm thanh tự động (Autoplay Policy) cho đến khi người dùng tương tác bấm vào trang ít nhất 1 lần.<br>
            Hãy bấm các nút bên dưới để mở khóa loa thiết bị và nghe thử âm thanh cảnh báo:
        </div>
    """), unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns(5)
    with c_btn1:
        if st.button("Test Loa Chung", key="test_audio_chime"):
            trigger_system_alert("success", force=True)
    with c_btn2:
        if st.button("Còi Ngủ Gật", key="test_audio_drowsy"):
            trigger_system_alert("drowsy", force=True)
    with c_btn3:
        if st.button("Còi Ngáp Ngủ", key="test_audio_yawn"):
            trigger_system_alert("yawn", force=True)
    with c_btn4:
        if st.button("Còi Mất Tập Trung", key="test_audio_distracted"):
            trigger_system_alert("distracted", force=True)
    with c_btn5:
        if st.button("Còi Cabin 2 Người", key="test_audio_dual"):
            trigger_system_alert("dual_drowsy", force=True)


# CƠ CHẾ CHUYỂN TAB MƯỢT MÀ TỨC THÌ (SEAMLESS TAB SWITCHING):
if "current_active_tab" not in st.session_state:
    st.session_state.current_active_tab = selected_module

if st.session_state.current_active_tab != selected_module:
    st.session_state.current_active_tab = selected_module
    active_src = str(st.session_state.get("active_source_type") or "")
    is_currently_live_cam = active_src.startswith("Camera trực tiếp") or active_src.startswith("Camera máy chủ")
    if not is_currently_live_cam and st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None
        st.session_state.active_video_path = None

# DANH MỤC VIDEO MẪU KIỂM THỬ TÍCH HỢP SẴN THEO PHÂN HỆ
SAMPLE_TEST_VIDEOS = {
    "[TAB 1]": [
        ("01_TaiXe_Ngap_Yawn_MAR.mp4 (Case 2: Ngáp ngủ / Biểu hiện ngủ)", "test_videos/Tab1_DMS_GiamSatTaiXe/01_TaiXe_Ngap_Yawn_MAR.mp4"),
        ("02_TaiXe_NguGat_Drowsy_EAR.mp4 (Case 1: Ngủ gật / Nhắm mắt)", "test_videos/Tab1_DMS_GiamSatTaiXe/02_TaiXe_NguGat_Drowsy_EAR.mp4"),
        ("03_TaiXe_QuayDau_MatTapTrung_Yaw.mp4 (Case 3: Mất tập trung / Quay đầu)", "test_videos/Tab1_DMS_GiamSatTaiXe/03_TaiXe_QuayDau_MatTapTrung_Yaw.mp4"),
        ("04_Cabin_2Nguoi_TaiXeNguGat_PhuXeXuLy.mp4 (Case 4: Cabin 2 người - Tài xế ngủ gật, phụ xe xử lý)", "test_videos/Tab1_DMS_GiamSatTaiXe/04_Cabin_2Nguoi_TaiXeNguGat_PhuXeXuLy.mp4")
    ],
    "[TAB 2]": [
        ("01_NguoiThat_Liveness_Pass.mp4 (Người thật - Điểm danh Pass)", "test_videos/Tab2_Liveness_ChongGiaMao/01_NguoiThat_Liveness_Pass.mp4"),
        ("02_GiaMao_InAnh_Spoof_Attack.mp4 (Tấn công ảnh in - Anti-Spoofing)", "test_videos/Tab2_Liveness_ChongGiaMao/02_GiaMao_InAnh_Spoof_Attack.mp4")
    ],
    "[TAB 3]": [
        ("01_AR_Kinh_XoayDau_3D_Homography.mp4 (Thử kính - Xoay đầu 3D)", "test_videos/Tab3_AR_Kinh_Homography3D/01_AR_Kinh_XoayDau_3D_Homography.mp4"),
        ("02_AR_Kinh_NhinThang_JitterFree.mp4 (Thử kính - Nhìn thẳng chống rung)", "test_videos/Tab3_AR_Kinh_Homography3D/02_AR_Kinh_NhinThang_JitterFree.mp4")
    ],
    "[TAB 4]": [
        ("01_TrangDiem_CuDongMoi_SonKhongLem.mp4 (Trang điểm - Môi cử động son không lem)", "test_videos/Tab4_TrangDiem_Makeup_HLS_LAB/01_TrangDiem_CuDongMoi_SonKhongLem.mp4"),
        ("02_TrangDiem_ChanDung_SonVaMaHong.mp4 (Trang điểm - Chân dung son & má hồng)", "test_videos/Tab4_TrangDiem_Makeup_HLS_LAB/02_TrangDiem_ChanDung_SonVaMaHong.mp4")
    ],
    "[TAB 5]": [
        ("01_AnhMat_LiecTrai_LiecPhai_Gaze.mp4 (Ánh mắt - Liếc trái phải điều hướng)", "test_videos/Tab5_AnhMat_Gaze_IrisTracking/01_AnhMat_LiecTrai_LiecPhai_Gaze.mp4"),
        ("02_AnhMat_NhayMat_VirtualClick.mp4 (Ánh mắt - Nháy mắt kích hoạt click ảo)", "test_videos/Tab5_AnhMat_Gaze_IrisTracking/02_AnhMat_NhayMat_VirtualClick.mp4")
    ],
    "[TAB 6]": [
        ("01_FaceSwap_NguoiThat_ChinhDien.mp4 (Hoán đổi khuôn mặt Delaunay thời gian thực)", "test_videos/Tab6_FaceMorph_FaceSwap/01_FaceSwap_NguoiThat_ChinhDien.mp4"),
        ("02_FaceSwap_XoayDau_3D.mp4 (Biến hình & Hoán đổi khuôn mặt khi xoay đầu 3D)", "test_videos/Tab6_FaceMorph_FaceSwap/02_FaceSwap_XoayDau_3D.mp4")
    ]
}

SAMPLE_TEST_IMAGES = {
    "[TAB 1]": [
        ("01_TaiXe_Ngap_Yawn_MAR.jpg (Case 2: Há miệng ngáp / Biểu hiện ngủ)", "test_images/Tab1_DMS_GiamSatTaiXe/01_TaiXe_Ngap_Yawn_MAR.jpg"),
        ("02_TaiXe_NguGat_Drowsy_EAR.jpg (Case 1: Nhắm mắt ngủ gật)", "test_images/Tab1_DMS_GiamSatTaiXe/02_TaiXe_NguGat_Drowsy_EAR.jpg"),
        ("03_TaiXe_QuayDau_MatTapTrung_Yaw.jpg (Case 3: Mất tập trung / Quay đầu)", "test_images/Tab1_DMS_GiamSatTaiXe/03_TaiXe_QuayDau_MatTapTrung_Yaw.jpg"),
        ("04_Cabin_2Nguoi_TaiXeNguGat.jpg (Case 4: Cabin 2 người - Tài xế ngủ gật, phụ xe tỉnh táo)", "test_images/Tab1_DMS_GiamSatTaiXe/04_Cabin_2Nguoi_TaiXeNguGat.jpg")
    ],
    "[TAB 2]": [
        ("01_NguoiThat_Liveness_Pass.jpg (Người thật - Chân dung)", "test_images/Tab2_Liveness_ChongGiaMao/01_NguoiThat_Liveness_Pass.jpg"),
        ("02_GiaMao_InAnh_Spoof_Attack.jpg (Tấn công ảnh in)", "test_images/Tab2_Liveness_ChongGiaMao/02_GiaMao_InAnh_Spoof_Attack.jpg")
    ],
    "[TAB 3]": [
        ("01_AR_Kinh_XoayDau.jpg (Thử kính AR)", "test_images/Tab3_AR_Kinh_Homography3D/01_AR_Kinh_XoayDau.jpg")
    ],
    "[TAB 4]": [
        ("01_TrangDiem_Moi.jpg (Trang điểm son môi & phấn má)", "test_images/Tab4_TrangDiem_Makeup_HLS_LAB/01_TrangDiem_Moi.jpg")
    ],
    "[TAB 5]": [
        ("01_AnhMat_Liec.jpg (Theo dõi ánh mắt & đồng tử)", "test_images/Tab5_AnhMat_Gaze_IrisTracking/01_AnhMat_Liec.jpg")
    ],
    "[TAB 6]": [
        ("01_KhuonMat_Mau_A_Nam.jpg (Khuôn mặt Đích A - Nam chính diện)", "test_images/Tab6_FaceMorph_FaceSwap/01_KhuonMat_Mau_A_Nam.jpg"),
        ("02_KhuonMat_Mau_B_Nu.jpg (Khuôn mặt Đích B - Nữ chính diện)", "test_images/Tab6_FaceMorph_FaceSwap/02_KhuonMat_Mau_B_Nu.jpg"),
        ("03_KhuonMat_Mau_C_TaiXe.jpg (Khuôn mặt Đích C - Tài xế trong cabin)", "test_images/Tab6_FaceMorph_FaceSwap/03_KhuonMat_Mau_C_TaiXe.jpg")
    ]
}

# -------------------------------------------------------------
# THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# -------------------------------------------------------------
st.sidebar.markdown("### [THIẾT LẬP HỆ THỐNG]")
input_source = st.sidebar.radio(
    "[NGUỒN DỮ LIỆU ĐẦU VÀO]",
    [
        "Camera trực tiếp thời gian thực (Webcam máy tính / USB / DirectShow - Video Live 30-60 FPS)",
        "Camera thiết bị từ xa qua Trình duyệt (Webcam từ xa: Video Live WebRTC / Chụp ảnh & Báo loa)",
        "Tải lên tệp video kiểm thử (.mp4)",
        "Tải lên tệp hình ảnh kiểm thử (.jpg / .png)"
    ],
    index=0
)

uploaded_image_path = None
uploaded_img_bgr = None
uploaded_video_path = None
loop_video = False

if input_source.startswith("Camera trực tiếp") or input_source.startswith("Camera máy chủ"):
    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        cam_index = st.selectbox(
            "Cổng Camera:",
            [0, 1, 2, 3],
            index=0,
            help="0: Camera mặc định. Nếu dùng DroidCam/Iriun thường là 1 hoặc 2."
        )
    with col_c2:
        cam_driver = st.selectbox("Driver:", ["DirectShow (Windows)", "Mặc định (MSMF / Auto)"])

    run_camera = st.sidebar.checkbox("Khởi động Camera trực tiếp", value=False)
    fps_mode = st.sidebar.selectbox(
        "Tốc độ khung hình (Hardware Target):",
        ["Tối ưu thời gian thực (Zero Latency - 30 FPS)", "Mượt mà cao (60 FPS)", "Tiết kiệm tài nguyên (24 FPS)"],
        index=0
    )
elif input_source.startswith("Camera thiết bị"):
    cam_index = 0
    cam_driver = "DirectShow (Windows)"
    run_camera = False
    fps_mode = "30 FPS"
    st.sidebar.success(
        "**Chế độ Camera Trình duyệt (Client Webcam):**\n"
        "- Dành cho người test từ xa qua Ngrok hoặc điện thoại/laptop.\n"
        "- Hỗ trợ cả 3 tùy chọn linh hoạt:\n"
        "  1. **[VIDEO LIVE] WebRTC 30 FPS:** Camera chuyển động thời gian thực.\n"
        "  2. **[SNAPSHOT] Chụp ảnh tức thời:** Bấm chụp ảnh 1-chạm (chuẩn ban đầu).\n"
        "  3. **[CAMERA DI ĐỘNG] Chụp ảnh / Quay video:** Vượt 100% rào cản Zalo & HTTP.\n"
        "- Còi và giọng nói cảnh báo tự động phát trên loa thiết bị của bạn!"
    )
elif input_source.startswith("Tải lên tệp video"):
    cam_index = 0
    cam_driver = "DirectShow (Windows)"
    video_source_mode = st.sidebar.radio(
        "[CHỌN NGUỒN TỆP VIDEO]",
        ["Video mẫu kiểm thử chuẩn theo Tab", "Tải lên tệp video riêng (.mp4 / .avi)"]
    )

    if video_source_mode.startswith("Video mẫu"):
        tab_key = None
        for k in SAMPLE_TEST_VIDEOS:
            if k in selected_module:
                tab_key = k
                break

        if tab_key is not None:
            opts = SAMPLE_TEST_VIDEOS[tab_key]
        else:
            opts = []
            for k, v in SAMPLE_TEST_VIDEOS.items():
                opts.extend(v)

        selected_sample = st.sidebar.selectbox(
            "Chọn video kiểm thử:",
            opts,
            format_func=lambda x: x[0],
            key=f"sb_sample_vid_{selected_module}"
        )
        if selected_sample:
            candidate_p = os.path.join(os.path.dirname(__file__), selected_sample[1])
            if os.path.exists(candidate_p):
                uploaded_video_path = candidate_p
            elif os.path.exists(selected_sample[1]):
                uploaded_video_path = selected_sample[1]
    else:
        uploaded_video_file = st.sidebar.file_uploader(
            "Tải lên tệp video kiểm thử (.mp4 / .avi):",
            type=["mp4", "avi", "mov"]
        )
        if uploaded_video_file is not None:
            file_key = f"uploaded_{uploaded_video_file.name}_{uploaded_video_file.size}"
            if st.session_state.get("cached_uploaded_key") != file_key:
                old_tmp = st.session_state.get("cached_uploaded_path")
                if old_tmp and os.path.isfile(old_tmp):
                    try:
                        os.remove(old_tmp)
                    except Exception:
                        pass
                uploaded_video_file.seek(0)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video_file.read())
                tfile.flush()
                tfile.close()
                st.session_state["cached_uploaded_path"] = tfile.name
                st.session_state["cached_uploaded_key"] = file_key
            uploaded_video_path = st.session_state.get("cached_uploaded_path")

    loop_video = st.sidebar.checkbox("Lặp lại video vô tận", value=True)
    run_camera = st.sidebar.checkbox("Bắt đầu phát Video", value=False)
    fps_mode = "30 FPS"
else:
    cam_index = 0
    cam_driver = "DirectShow (Windows)"
    image_source_mode = st.sidebar.radio(
        "[CHỌN NGUỒN TỆP ẢNH]",
        ["Ảnh mẫu kiểm thử chuẩn theo Tab", "Tải lên tệp ảnh riêng (.jpg / .png / .jpeg)"]
    )

    if image_source_mode.startswith("Ảnh mẫu"):
        tab_key = None
        for k in SAMPLE_TEST_IMAGES:
            if k in selected_module:
                tab_key = k
                break

        if tab_key is not None:
            opts_img = SAMPLE_TEST_IMAGES[tab_key]
        else:
            opts_img = []
            for k, v in SAMPLE_TEST_IMAGES.items():
                opts_img.extend(v)

        selected_sample_img = st.sidebar.selectbox(
            "Chọn ảnh kiểm thử:",
            opts_img,
            format_func=lambda x: x[0],
            key=f"sb_sample_img_{selected_module}"
        )
        if selected_sample_img:
            candidate_img_p = os.path.join(os.path.dirname(__file__), selected_sample_img[1])
            if os.path.exists(candidate_img_p):
                uploaded_image_path = candidate_img_p
                uploaded_img_bgr = cv2.imread(candidate_img_p)
            elif os.path.exists(selected_sample_img[1]):
                uploaded_image_path = selected_sample_img[1]
                uploaded_img_bgr = cv2.imread(selected_sample_img[1])
    else:
        uploaded_img_file = st.sidebar.file_uploader(
            "Tải lên ảnh chân dung (.jpg / .png / .jpeg):",
            type=["jpg", "png", "jpeg"]
        )
        if uploaded_img_file is not None:
            img_bytes = uploaded_img_file.read()
            uploaded_img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    run_camera = st.sidebar.checkbox("Bắt đầu Phân tích Ảnh", value=True)
    fps_mode = "30 FPS"

if "60" in fps_mode:
    target_hardware_fps = 60
elif "24" in fps_mode:
    target_hardware_fps = 24
else:
    target_hardware_fps = 30

st.sidebar.markdown("---")
st.sidebar.markdown("### [SCALE KHUNG HÌNH & CABIN 2 NGƯỜI]")
col_z1, col_z2 = st.sidebar.columns(2)
with col_z1:
    cam_zoom_scale = st.slider(
        "Phóng to / Thu nhỏ (Scale):",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="< 1.0: Thu nhỏ toàn cảnh (Wide View) bao quát 2 ghế lái & phụ. > 1.0: Phóng to cận cảnh khuôn mặt."
    )
with col_z2:
    cam_pan_x = st.slider(
        "Góc nhìn ngang (Pan):",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="-1.0: Lệch trái (Tài xế). +1.0: Lệch phải (Phụ xe)."
    )

cabin_mode_choice = st.sidebar.selectbox(
    "Chế độ Giám sát Cabin:",
    ["Tự động phát hiện (1 hoặc 2 Người)", "Giám sát 1 Người (Tài xế đơn)", "Giám sát Cabin 2 Người (Tài xế & Phụ xe)"],
    index=0
)
if "1 Người" in cabin_mode_choice:
    cabin_mode_code = "single"
elif "2 Người" in cabin_mode_choice:
    cabin_mode_code = "dual"
else:
    cabin_mode_code = "auto"

st.sidebar.markdown("---")
st.sidebar.markdown("### [HIỂN THỊ TRỰC QUAN AI]")
show_face_box = st.sidebar.checkbox("Khung nhận diện (Face Bounding Box & HUD)", value=True)
show_mesh_contours = st.sidebar.checkbox("Lưới đa giác nét mặt (Face Mesh Contours)", value=True)
show_geo_vectors = st.sidebar.checkbox("Vector hình học giải tích (EAR & MAR Lines)", value=True)
clahe_night_toggle = st.sidebar.checkbox("Cân bằng sáng ban đêm (CLAHE Night-Vision)", value=False, help="Tự động cân bằng sáng cục bộ trên kênh L* (CIELAB) khi lái xe ban đêm hoặc ngược sáng.")
engine.enable_clahe_night = clahe_night_toggle

# Khối tinh chỉnh ngưỡng DMS - Chỉ hiển thị khi ở Tab 1 hoặc Toàn cảnh
def render_dms_tuning_sidebar():
    st.sidebar.markdown("---")
    sound_toggle = st.sidebar.checkbox("Bật âm thanh cảnh báo (Chỉ Tab 1)", value=True)
    audio.set_enabled(sound_toggle)

    sound_mode_map = {
        "Âm báo Beep + Giọng nói tiếng Việt": "combined",
        "Chỉ Giọng nói tiếng Việt": "voice_only",
        "Chỉ Âm báo Beep": "beep_only"
    }
    sound_mode_choice = st.sidebar.selectbox(
        "Chế độ cảnh báo âm thanh",
        list(sound_mode_map.keys()),
        index=0,
        disabled=not sound_toggle,
        key="sb_sound_mode"
    )
    audio.set_mode(sound_mode_map[sound_mode_choice])

    audio_channel_map = {
        "Loa máy tính trực tiếp (OS Sound - Triệt tiêu Echo/Double)": "os",
        "Loa trình duyệt Web (Browser HTML5 - Cho máy khách từ xa)": "browser",
        "Đồng bộ cả hai kênh (Server OS + Client Browser)": "both"
    }
    audio_channel_choice = st.sidebar.selectbox(
        "Kênh phát âm thanh (Chống lặp âm)",
        list(audio_channel_map.keys()),
        index=0,
        disabled=not sound_toggle,
        key="sb_audio_channel"
    )
    st.session_state["audio_channel"] = audio_channel_map[audio_channel_choice]

    st.sidebar.markdown("### [TINH CHỈNH NGƯỠNG DMS]")
    ear_val = st.sidebar.slider("Ngưỡng nhắm mắt EAR", 0.15, 0.30, engine.ear_thresh, 0.01, key="sb_ear")
    ear_f_val = st.sidebar.slider("Số frames nhắm mắt liên tiếp", 3, 30, engine.ear_consec_frames, 1, key="sb_ear_f")
    mar_val = st.sidebar.slider("Ngưỡng ngáp ngủ MAR", 0.40, 0.85, engine.mar_thresh, 0.01, key="sb_mar")
    pitch_val = st.sidebar.slider("Ngưỡng cúi đầu Pitch (độ)", 8.0, 30.0, engine.pitch_down_thresh, 1.0, key="sb_pitch")
    yaw_val = st.sidebar.slider("Ngưỡng quay mặt Yaw (độ)", 12.0, 45.0, engine.yaw_turn_thresh, 1.0, key="sb_yaw")
    perclos_val = st.sidebar.slider("Ngưỡng PERCLOS (%)", 15.0, 45.0, engine.perclos_thresh, 1.0, key="sb_perclos")

    engine.ear_thresh = ear_val
    engine.ear_consec_frames = ear_f_val
    engine.mar_thresh = mar_val
    engine.pitch_down_thresh = pitch_val
    engine.yaw_turn_thresh = yaw_val
    engine.perclos_thresh = perclos_val

    col_b1, col_b2 = st.sidebar.columns(2)
    with col_b1:
        if safe_button("Hiệu chuẩn 0", key="btn_calib"):
            if st.session_state.last_pitch != 0.0 or st.session_state.last_yaw != 0.0:
                engine.calibrate_neutral_pose(
                    st.session_state.last_pitch,
                    st.session_state.last_yaw,
                    st.session_state.last_roll,
                    st.session_state.get("last_ear", 0.28)
                )
                st.sidebar.success(f"Đã hiệu chuẩn Góc 0 & Ngưỡng EAR cá nhân ({engine.ear_thresh:.2f})!")
    with col_b2:
        if safe_button("Reset bộ đếm", key="btn_reset_cnt"):
            engine.reset_states()
            engine.reset_calibration()
            reset_liveness_fsm()
            st.sidebar.info("Đã reset trạng thái.")

if is_tab1 or is_all:
    render_dms_tuning_sidebar()

# -------------------------------------------------------------
# KHUNG NGẮM CAMERA THIẾT BỊ TRÌNH DUYỆT & BYPASS ZALO / HTTP (DUAL-ENGINE CLIENT CAM)
# -------------------------------------------------------------
if input_source.startswith("Camera thiết bị"):
    import streamlit.components.v1 as components

    # Tự động phát hiện Zalo In-App Browser hoặc giao thức HTTP không bảo mật để hỗ trợ mở trình duyệt ngoài / nâng cấp HTTPS
    components.html(
        """
        <div id="env_guard_box" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f1f5f9;"></div>
        <script>
        (function() {
            var ua = navigator.userAgent || '';
            var isZalo = /Zalo/i.test(ua);
            var isFB = /FBAN|FBAV|Messenger/i.test(ua);
            var loc = window.top.location;
            var isHttpRemote = (loc.protocol === 'http:' && loc.hostname !== 'localhost' && loc.hostname !== '127.0.0.1');
            var box = document.getElementById('env_guard_box');

            // Tự động chuyển sang HTTPS nếu đang mở link Ngrok bằng HTTP
            if (isHttpRemote && (loc.hostname.indexOf('ngrok') !== -1)) {
                loc.replace('https://' + loc.host + loc.pathname + loc.search);
                return;
            }

            if (isZalo || isFB || isHttpRemote) {
                var targetUrl = 'https://' + loc.host + loc.pathname + loc.search;
                var androidIntent = 'intent://' + loc.host + loc.pathname + loc.search + '#Intent;scheme=https;package=com.android.chrome;end';
                var reasonTxt = isZalo ? 'Trình duyệt nhúng Zalo (Zalo WebView)' : (isFB ? 'Trình duyệt nhúng Messenger/Facebook' : 'Giao thức HTTP chưa mã hóa');
                box.innerHTML = '<div style="background: rgba(245, 158, 11, 0.18); border: 1.5px solid #f59e0b; border-left: 4px solid #f59e0b; border-radius: 10px; padding: 10px 14px; font-size: 12.5px; line-height: 1.5;">' +
                    '<div style="color: #fbbf24; font-weight: 800; text-transform: uppercase; margin-bottom: 4px;">[PHÁT HIỆN MÔI TRƯỜNG: ' + reasonTxt + ']</div>' +
                    '<div style="color: #e2e8f0; margin-bottom: 8px;">Để dùng Camera không bị chặn quyền WebRTC, bạn hãy chọn <b>[Chế độ 2: Camera Hệ thống Điện thoại (Zalo / HTTP Bypass)]</b> ngay bên dưới, hoặc bấm nút mở sang Chrome/Safari:</div>' +
                    '<div style="display: flex; gap: 8px; flex-wrap: wrap;">' +
                        '<a href="' + androidIntent + '" target="_top" style="background: #00f2fe; color: #070b14; font-weight: 800; padding: 5px 12px; border-radius: 6px; text-decoration: none; font-size: 12px;">Mở ngay bằng Chrome (Android)</a>' +
                        '<a href="' + targetUrl + '" target="_blank" rel="noopener" style="background: rgba(0, 242, 254, 0.15); color: #00f2fe; border: 1px solid #00f2fe; font-weight: 700; padding: 5px 12px; border-radius: 6px; text-decoration: none; font-size: 12px;">Chuyển sang HTTPS / Mở Tab Mới</a>' +
                    '</div>' +
                '</div>';
            } else {
                box.innerHTML = '<div style="background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(16, 185, 129, 0.4); border-left: 4px solid #10b981; border-radius: 10px; padding: 8px 14px; font-size: 12.5px; color: #34d399; font-weight: 700;">[MÔI TRƯỜNG BẢO MẬT CHUẨN HTTPS / LOCALHOST] Trình duyệt hỗ trợ đầy đủ cả WebRTC Live Camera và Camera Hệ thống.</div>';
            }
        })();
        </script>
        """,
        height=95
    )

    remote_cam_mode = st.radio(
        "CHỌN PHƯƠNG THỨC KIỂM THỬ TỪ THIẾT BỊ CỦA BẠN:",
        [
            "Chế độ 1: [VIDEO LIVE] Trực tiếp từ xa (WebRTC 30 FPS Stream - Camera chuyển động thời gian thực)",
            "Chế độ 2: [SNAPSHOT] Chụp ảnh tức thời từ Webcam (Snapshot 1-Chạm - Giữ nguyên chuẩn ban đầu)",
            "Chế độ 3: [CAMERA DI ĐỘNG] Mở Camera Hệ thống Điện thoại - Chụp ảnh / Quay Video (Vượt rào cản Zalo & HTTP)"
        ],
        horizontal=False,
        key="rad_remote_cam_mode"
    )

    if remote_cam_mode.startswith("Chế độ 1"):
        if HAS_WEBRTC:
            st.markdown(clean_html("""
                <div style="background: rgba(0, 242, 254, 0.08); border: 1.5px solid #00f2fe; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="color: #00f2fe; font-weight: 800; font-size: 14px; margin-bottom: 4px;">
                        [WEBRTC LIVE STREAM] LUỒNG VIDEO LIVE THỜI GIAN THỰC TỪ CAMERA THIẾT BỊ TỪ XA
                    </div>
                    <div style="color: #e2e8f0; font-size: 12.5px; line-height: 1.6;">
                        - Nhấn nút <b>START</b> bên dưới để bật Camera thiết bị của bạn.<br>
                        - Trình duyệt sẽ hỏi: <i>"Cho phép sử dụng Máy ảnh"</i> -&gt; Chọn <b>Cho phép (Allow)</b>.<br>
                        - Hình ảnh khuôn mặt của bạn sẽ chuyển động trực tiếp theo thời gian thực (20 - 30 FPS), kèm HUD đồ họa Cyberpunk và còi báo động!
                    </div>
                </div>
            """), unsafe_allow_html=True)

            # Đóng gói toàn bộ biến cấu hình từ luồng chính để truyền an toàn sang luồng Worker WebRTC (Không gọi st.session_state trong recv)
            _init_sel_g = st.session_state.get("selected_glasses_choice", "")
            _init_glasses_db = st.session_state.get("glasses_db", {})
            _init_glasses_bgra = _init_glasses_db.get(_init_sel_g) if _init_glasses_db else None
            _init_lip_name = st.session_state.get("makeup_lip_color", "Hồng Cherry Tươi Trẻ")
            _init_lip_bgr = palette_lip.get(_init_lip_name, (20, 15, 235))
            _init_lip_alpha = float(st.session_state.get("makeup_lip_alpha", 0.55))
            _init_blush_name = st.session_state.get("makeup_blush_color", "Hồng Baby Xinh Xắn")
            _init_enable_blush = bool(st.session_state.get("makeup_enable_blush", True))
            _init_blush_bgr = palette_blush.get(_init_blush_name, (160, 110, 255)) if _init_enable_blush else None
            _init_blush_alpha = float(st.session_state.get("makeup_blush_alpha", 0.35))
            _init_cs_choice = str(st.session_state.get("makeup_color_space", "HYBRID"))
            _init_cs_m = "LAB" if _init_cs_choice.startswith("LAB") else ("HLS" if _init_cs_choice.startswith("HLS") else "HYBRID")
            _init_morph_src_img, _init_morph_src_lm = get_morph_source_data()
            _init_morph_alpha = float(st.session_state.get("morph_alpha", 1.0))
            _init_morph_mode = str(st.session_state.get("morph_blend_mode", "LAB_FEATHER"))
            _init_morph_delaunay = bool(st.session_state.get("morph_show_delaunay", False))

            class WebRTCVideoProcessor:
                def __init__(self):
                    self.last_alert_time = 0.0
                    self.zoom_scale = float(cam_zoom_scale)
                    self.pan_x = float(cam_pan_x)
                    self.cabin_mode = str(cabin_mode_code)
                    self.draw_box = bool(show_face_box)
                    self.draw_contours = bool(show_mesh_contours)
                    self.draw_vectors = bool(show_geo_vectors)
                    self.is_t1 = bool(is_tab1)
                    self.is_t2 = bool(is_tab2)
                    self.is_t3 = bool(is_tab3)
                    self.is_t4 = bool(is_tab4)
                    self.is_t5 = bool(is_tab5)
                    self.is_t6 = bool(is_tab6)
                    self.is_all_tabs = bool(is_all)
                    self.morph_src_img = _init_morph_src_img
                    self.morph_src_lm = _init_morph_src_lm
                    self.morph_alpha = _init_morph_alpha
                    self.morph_mode = _init_morph_mode
                    self.morph_delaunay = _init_morph_delaunay
                    self.glasses_bgra = _init_glasses_bgra
                    self.lip_bgr = _init_lip_bgr
                    self.lip_alpha = _init_lip_alpha
                    self.blush_bgr = _init_blush_bgr
                    self.blush_alpha = _init_blush_alpha
                    self.cs_m = _init_cs_m
                    # Bộ đếm FSM Liveness nội bộ cho luồng WebRTC Tab 2
                    self.w_stage = 0
                    self.w_blinks = 0
                    self.w_blink_f = 0
                    self.w_turn_f = 0
                    self.w_straight_f = 0
                    self.w_aligned_crop = None

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.flip(img, 1)
                    h_f, w_f = img.shape[:2]
                    if w_f > 640:
                        new_h = int(h_f * (640.0 / w_f))
                        img = cv2.resize(img, (640, new_h), interpolation=cv2.INTER_LINEAR)

                    if self.zoom_scale != 1.0 or self.pan_x != 0.0:
                        img = apply_camera_scale(img, scale=self.zoom_scale, pan_x=self.pan_x)

                    need_dms_hud = self.is_t1 or self.is_all_tabs
                    metrics, annotated = engine.process_frame(
                        img,
                        is_static_image=False,
                        cabin_mode=self.cabin_mode,
                        draw_box=self.draw_box,
                        draw_contours=self.draw_contours,
                        draw_vectors=self.draw_vectors,
                        render_hud=need_dms_hud
                    )

                    landmarks_2d = metrics.get("landmarks_2d")

                    # Xử lý theo từng phân hệ hoàn toàn Thread-Safe (Không truy cập st.session_state)
                    if self.is_t2 and landmarks_2d:
                        annotated = engine.draw_face_annotations(
                            annotated,
                            landmarks_2d,
                            {"liveness_success": (self.w_stage == 3)},
                            draw_box=True,
                            draw_contours=self.draw_contours,
                            draw_vectors=False,
                            badge_mode="biometric"
                        )
                        if self.w_stage == 0:
                            if metrics["ear"] < engine.ear_thresh:
                                self.w_blink_f += 1
                            else:
                                if 1 <= self.w_blink_f <= 18:
                                    self.w_blinks += 1
                                self.w_blink_f = 0
                            if self.w_blinks >= 2:
                                self.w_stage = 1
                                audio.trigger_alert("success")
                            step_txt = f"STEP 1/3: BLINK EYES ({self.w_blinks}/2)"
                        elif self.w_stage == 1:
                            if abs(metrics["yaw"]) > 15.0:
                                self.w_turn_f += 1
                            else:
                                self.w_turn_f = max(0, self.w_turn_f - 1)
                            if self.w_turn_f >= 5:
                                self.w_stage = 2
                                audio.trigger_alert("success")
                            step_txt = f"STEP 2/3: TURN HEAD ({self.w_turn_f}/5)"
                        elif self.w_stage == 2:
                            if abs(metrics["yaw"]) < 12.0 and abs(metrics["pitch"]) < 14.0:
                                self.w_straight_f += 1
                            else:
                                self.w_straight_f = max(0, self.w_straight_f - 1)
                            if self.w_straight_f >= 4:
                                self.w_stage = 3
                                self.w_aligned_crop, _ = engine.align_face(img, landmarks_2d, 110, 110)
                                audio.trigger_alert("success")
                            step_txt = f"STEP 3/3: HOLD STRAIGHT ({self.w_straight_f}/4)"
                        else:
                            step_txt = "[VERIFIED] LIVENESS & AFFINE 256x256 OK"
                            if self.w_aligned_crop is not None:
                                ah, aw = self.w_aligned_crop.shape[:2]
                                annotated[10:10+ah, annotated.shape[1]-aw-10:annotated.shape[1]-10] = self.w_aligned_crop
                                cv2.rectangle(annotated, (annotated.shape[1]-aw-10, 10), (annotated.shape[1]-10, 10+ah), (0, 255, 128), 2)
                        cv2.putText(annotated, step_txt, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 180), 2, cv2.LINE_AA)
                    elif self.is_t3 and landmarks_2d:
                        if self.glasses_bgra is not None:
                            annotated = engine.overlay_glasses_homography(annotated, landmarks_2d, self.glasses_bgra)
                    elif self.is_t4 and landmarks_2d:
                        annotated = engine.apply_makeup(
                            annotated,
                            landmarks_2d,
                            self.lip_bgr,
                            lip_alpha=self.lip_alpha,
                            blush_bgr=self.blush_bgr,
                            blush_alpha=self.blush_alpha,
                            color_space=self.cs_m
                        )
                    elif self.is_t5 and landmarks_2d:
                        annotated = engine.draw_iris_landmarks(annotated, landmarks_2d)
                    elif self.is_t6 and landmarks_2d:
                        if self.morph_src_img is not None and self.morph_src_lm is not None:
                            annotated, _ = engine.swap_and_morph_faces(
                                annotated,
                                landmarks_2d,
                                self.morph_src_img,
                                self.morph_src_lm,
                                alpha=self.morph_alpha,
                                blend_mode=self.morph_mode,
                                show_delaunay=self.morph_delaunay
                            )

                    # Xử lý còi cảnh báo âm thanh cho Tab 1
                    if self.is_t1 or self.is_all_tabs:
                        now = time.time()
                        if metrics.get("is_dual_mode") and metrics.get("dual_alert_type") in ["driver_drowsy", "driver_yawn", "driver_distracted"]:
                            if now - self.last_alert_time > 3.5:
                                self.last_alert_time = now
                                a_key = "dual_drowsy" if metrics["dual_alert_type"] == "driver_drowsy" else metrics["dual_alert_type"].replace("driver_", "")
                                audio.trigger_alert(a_key)
                        else:
                            if metrics.get("is_yawning"):
                                if now - self.last_alert_time > 3.5:
                                    self.last_alert_time = now
                                    audio.trigger_alert("yawn")
                            elif metrics.get("is_drowsy"):
                                if now - self.last_alert_time > 3.5:
                                    self.last_alert_time = now
                                    audio.trigger_alert("drowsy")
                            elif metrics.get("is_distracted"):
                                if now - self.last_alert_time > 3.5:
                                    self.last_alert_time = now
                                    audio.trigger_alert("distracted")

                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            webrtc_ctx = webrtc_streamer(
                key=f"dms_webrtc_stream_{selected_module}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=WebRTCVideoProcessor,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}}, "audio": False},
                async_processing=True
            )
            st.session_state["active_webrtc_ctx"] = webrtc_ctx
        else:
            st.markdown(clean_html("""
                <div style="background: rgba(245, 158, 11, 0.15); border: 1.5px solid #f59e0b; border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;">
                    <div style="color: #fbbf24; font-weight: 800; font-size: 14px; margin-bottom: 6px;">
                        [TÍNH NĂNG VIDEO LIVE WEBRTC CHO MÁY KHÁCH TỪ XA]
                    </div>
                    <div style="color: #e2e8f0; font-size: 12.5px; line-height: 1.6;">
                        Để kích hoạt luồng <b>Video Live chuyển động liên tục 30 FPS</b> từ camera trình duyệt máy khách về server qua Internet, máy chủ chỉ cần cài thêm thư viện chuẩn:
                        <div style="background: #0f172a; border: 1px solid #334155; padding: 8px 12px; border-radius: 6px; font-family: monospace; color: #38bdf8; font-size: 13px; margin: 8px 0;">
                            pip install streamlit-webrtc av
                        </div>
                        <i>(Hoặc chỉ cần nhấp đúp tệp <b>cai_dat_webrtc.bat</b> có sẵn trong thư mục dự án để tự động cài đặt trong 30 giây!)</i><br><br>
                        Hiện tại, bạn có thể chọn ngay <b>[Chế độ 2: Chụp ảnh tức thời từ Webcam]</b> hoặc <b>[Chế độ 3: Mở Camera Điện thoại / Quay Video]</b> ở bên dưới để kiểm thử ngay lập tức!
                    </div>
                </div>
            """), unsafe_allow_html=True)

    elif remote_cam_mode.startswith("Chế độ 2"):
        client_cam_file = st.camera_input(
            "Khung ngắm Camera Trình duyệt (Bấm 'Allow / Cho phép' rồi nhấn 'Take Photo / Chụp ảnh' để phân tích):",
            key="client_cam_input"
        )
        if client_cam_file is not None:
            img_bytes = client_cam_file.getvalue()
            uploaded_img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            run_camera = True

    # Chế độ 3 hoặc ô tải lên camera hệ thống dự phòng
    native_label = (
        "Bấm 'Browse files' -> Chọn 'Chụp ảnh / Máy ảnh (Camera)' trên điện thoại để chụp ảnh hoặc quay video trực tiếp (Không bao giờ bị Zalo hay HTTP chặn):"
        if remote_cam_mode.startswith("Chế độ 3")
        else "Hoặc dùng Camera Gốc Hệ thống / Tải tệp dự phòng (Nếu trình duyệt Zalo / HTTP chặn khung ngắm):"
    )
    native_cam_file = st.file_uploader(
        native_label,
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov"],
        key="native_os_camera_bypass"
    )
    if native_cam_file is not None:
        fname_lower = native_cam_file.name.lower()
        if fname_lower.endswith((".mp4", ".mov")):
            f_key = f"remote_vid_{native_cam_file.name}_{native_cam_file.size}"
            if st.session_state.get("cached_remote_vid_key") != f_key:
                old_rv = st.session_state.get("cached_remote_vid_path")
                if old_rv and os.path.isfile(old_rv):
                    try:
                        os.remove(old_rv)
                    except Exception:
                        pass
                native_cam_file.seek(0)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(native_cam_file.read())
                tfile.flush()
                tfile.close()
                st.session_state["cached_remote_vid_path"] = tfile.name
                st.session_state["cached_remote_vid_key"] = f_key
            uploaded_video_path = st.session_state.get("cached_remote_vid_path")
            input_source = "Tải lên tệp video kiểm thử (.mp4)"
            loop_video = True
            run_camera = True
        else:
            img_bytes = native_cam_file.getvalue()
            uploaded_img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            run_camera = True

# -------------------------------------------------------------
# BỐ CỤC GIAO DIỆN THEO PHÂN HỆ ĐƯỢC CHỌN (CLEAN & MODULAR)
# -------------------------------------------------------------
if is_tab1:
    st.markdown(clean_html("""
    <div class="dip-theory-card">
        <div class="dip-theory-header">Bản chất Xử lý ảnh số (DIP) - Phân hệ 1: Giám sát Tài xế (DMS)</div>
        <ul class="dip-theory-list">
            <li><b>Hình học giải tích 2D/3D:</b> Tính tỷ lệ khía cạnh mắt <b>EAR (Soukupova &amp; Cech, 2016)</b> và tỷ lệ mở miệng <b>MAR</b>.</li>
            <li><b>Ước lượng tư thế đầu 3D:</b> Giải hệ phương trình hình chiếu <b>cv2.solvePnP</b> với mô hình chuẩn 3D Canonical Face Model, phân rã 3 góc Euler (<b>Pitch, Yaw, Roll</b>) qua biến đổi Rodrigues.</li>
            <li><b>Chuẩn đo lường vi giấc ngủ PERCLOS (NHTSA):</b> Tích lũy tỷ lệ khung hình nhắm mắt trên cửa sổ trượt 90 frames kèm cổng chặn Sanity Gate.</li>
        </ul>
    </div>
    """), unsafe_allow_html=True)
    col_view1, col_stat1 = st.columns([2.55, 1.45])
    with col_view1:
        viewport_t1 = st.empty()
    with col_stat1:
        st.markdown('<div class="section-panel-title">[TRẠNG THÁI CẢNH BÁO]</div>', unsafe_allow_html=True)
        status_box_t1 = st.empty()
        st.markdown('<div class="section-panel-title">[CHỈ SỐ TELEMETRY THỜI GIAN THỰC]</div>', unsafe_allow_html=True)
        ear_metric_t1 = st.empty()
        mar_metric_t1 = st.empty()
        perclos_metric_t1 = st.empty()
        pose_metric_t1 = st.empty()

elif is_tab2:
    st.markdown(clean_html("""
    <div class="dip-theory-card">
        <div class="dip-theory-header">Bản chất Xử lý ảnh số (DIP) - Phân hệ 2: Điểm danh &amp; Chống giả mạo (Active Liveness)</div>
        <ul class="dip-theory-list">
            <li><b>Máy trạng thái hữu hạn (FSM Active Liveness):</b> Thử thách chống giả mạo sinh trắc học 3 bước (Chớp mắt chủ động -&gt; Quay mặt nghiêng góc Yaw -&gt; Giữ thẳng mặt).</li>
            <li><b>Chuẩn hóa khuôn mặt qua phép biến đổi Affine (Face Alignment):</b> Định vị tâm hai mắt qua mốc <b>33</b> và <b>263</b>, tính góc nghiêng θ = arctan(Δy / Δx), xây dựng ma trận <b>cv2.getRotationMatrix2D</b> và nội suy <b>cv2.warpAffine</b> đưa ảnh về chuẩn kích thước 256 × 256 pixels.</li>
        </ul>
    </div>
    """), unsafe_allow_html=True)
    col_view2, col_stat2 = st.columns([2.2, 1.8])
    with col_view2:
        viewport_t2 = st.empty()
    with col_stat2:
        st.markdown('<div class="section-panel-title">[TRÌNH ĐIỀU KHIỂN ĐIỂM DANH SINH TRẮC HỌC]</div>', unsafe_allow_html=True)

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            lv_toggle = st.checkbox(
                "Bật kiểm tra Liveness",
                value=st.session_state.liveness_active_toggle,
                key="chk_liveness_active"
            )
            st.session_state.liveness_active_toggle = lv_toggle
        with col_l2:
            if safe_button("Làm lại Điểm danh", key="btn_redo_liveness"):
                reset_liveness_fsm()
                st.session_state.liveness_active_toggle = True
                st.rerun()

        liveness_instruction = st.empty()
        liveness_progress = st.empty()
        liveness_result_box = st.empty()
        liveness_face_crop = st.empty()

        # Khôi phục hiển thị bền vững nếu đã hoàn tất bước 3 (kể cả khi dừng camera hoặc rerun)
        if st.session_state.liveness_stage == 3 and st.session_state.liveness_aligned_b64:
            liveness_instruction.markdown(
                build_liveness_stepper(
                    3,
                    True,
                    "[XÁC THỰC THÀNH CÔNG] ĐÃ ĐIỂM DANH SINH TRẮC HỌC!",
                    "Đã hoàn tất 3/3 bước thử thách Liveness & chuẩn hóa Affine 256x256"
                ),
                unsafe_allow_html=True
            )
            liveness_progress.progress(100)
            liveness_result_box.markdown(
                build_status_banner("safe", "[HOÀN TẤT] XÁC THỰC NGƯỜI THẬT (REAL PERSON DETECTED)"),
                unsafe_allow_html=True
            )
            render_aligned_face_card(liveness_face_crop, st.session_state.liveness_aligned_b64)
        else:
            liveness_instruction.markdown(
                build_liveness_stepper(
                    st.session_state.liveness_stage,
                    st.session_state.liveness_active_toggle,
                    "[CHẾ ĐỘ CHỜ] Tích chọn 'Bật kiểm tra Liveness' để bắt đầu" if not st.session_state.liveness_active_toggle else "Đang sẵn sàng nhận diện khuôn mặt...",
                    "Khởi động Camera ở thanh bên trái để thực hiện thử thách 3 bước"
                ),
                unsafe_allow_html=True
            )
            liveness_progress.progress(0)

elif is_tab3:
    st.markdown(clean_html("""
    <div class="dip-theory-card">
        <div class="dip-theory-header">Bản chất Xử lý ảnh số (DIP) - Phân hệ 3: Thử kính thực tế ảo (3D Homography AR)</div>
        <ul class="dip-theory-list">
            <li><b>Biến đổi phối cảnh Homography 3D (3x3):</b> Trích xuất các mốc ở thái dương (<b>127</b>, <b>356</b>) và sống mũi (<b>168</b>), tính tỷ lệ co ngắn phối cảnh 3D (Foreshortening) khi quay mặt trái/phải và giải ma trận <b>cv2.findHomography</b> + <b>cv2.warpPerspective</b> trên vùng ROI mắt.</li>
            <li><b>Trộn kênh trong suốt (Alpha Blending):</b> Hòa trộn 4 kênh BGRA vào khung hình gốc theo hệ số độ mờ (Alpha Compositing).</li>
        </ul>
    </div>
    """), unsafe_allow_html=True)
    col_ctrl3, col_view3 = st.columns([1.2, 2.8])
    with col_ctrl3:
        st.markdown('<div class="section-panel-title">[CHỌN MẪU KÍNH THỜI TRANG]</div>', unsafe_allow_html=True)

        g_list = list(st.session_state.glasses_db.keys())
        g_idx = g_list.index(st.session_state.selected_glasses_choice) if st.session_state.selected_glasses_choice in g_list else 0
        selected_g = st.selectbox(
            "Bộ sưu tập kính thực tế ảo:",
            g_list,
            index=g_idx,
            key="sb_glasses_choice"
        )
        st.session_state.selected_glasses_choice = selected_g
        st.markdown("---")
        st.markdown('<div class="section-panel-title">[THÔNG SỐ KẾT XUẤT AR]</div>', unsafe_allow_html=True)
        st.markdown(
            build_metric_card(
                "Mẫu kính đang chọn",
                selected_g,
                "Nội suy phối cảnh 6 điểm mốc Homography 3D + Alpha Blending",
                pct=100,
                bar_color="#00f2fe",
                badge_text="BGRA 4-CH"
            ),
            unsafe_allow_html=True
        )

    with col_view3:
        viewport_t3 = st.empty()

elif is_tab4:
    st.markdown(clean_html("""
    <div class="dip-theory-card">
        <div class="dip-theory-header">Bản chất Xử lý ảnh số (DIP) - Phân hệ 4: Trang điểm thực tế ảo (CIELAB &amp; HLS)</div>
        <ul class="dip-theory-list">
            <li><b>Không gian màu CIELAB &amp; HLS:</b> Chuyển đổi từ BGR sang <b>CIELAB</b> (tách riêng độ sáng L* và hai trục màu đối thủ a*, b*) hoặc <b>HLS</b> để bảo toàn hoàn toàn chi tiết vân môi và độ bóng tự nhiên.</li>
            <li><b>Tạo mặt nạ nhị phân ROI &amp; Làm mờ biên Gaussian:</b> Dùng 40 mốc hình học môi (<b>cv2.fillPoly</b>), trừ vùng khoang miệng, lọc làm mờ biên Gaussian Blur (kernel 7 × 7) để màu son hòa trộn mềm mịn.</li>
        </ul>
    </div>
    """), unsafe_allow_html=True)
    col_ctrl4, col_view4 = st.columns([1.2, 2.8])
    with col_ctrl4:
        st.markdown('<div class="section-panel-title">[BẢNG MÀU SON MÔI]</div>', unsafe_allow_html=True)
        lip_options = list(palette_lip.keys())
        lip_sel = st.selectbox(
            "Màu son thời trang:",
            lip_options,
            index=lip_options.index(st.session_state.makeup_lip_color) if st.session_state.makeup_lip_color in lip_options else 0,
            key="sb_lip_color"
        )
        st.session_state.makeup_lip_color = lip_sel

        cs_options = [
            "HYBRID (Kết hợp CIELAB + HLS)",
            "LAB (Chuẩn thị giác CIELAB L*a*b*)",
            "HLS (Hue - Lightness - Saturation)"
        ]
        cs_sel = st.selectbox(
            "Thuật toán không gian màu:",
            cs_options,
            index=cs_options.index(st.session_state.makeup_color_space) if st.session_state.makeup_color_space in cs_options else 0,
            key="sb_color_space"
        )
        st.session_state.makeup_color_space = cs_sel

        alpha_sel = st.slider("Độ đậm son môi (Opacity)", 0.1, 1.0, float(st.session_state.makeup_lip_alpha), 0.05, key="sl_lip_alpha")
        st.session_state.makeup_lip_alpha = alpha_sel

        st.markdown('<div class="section-panel-title">[PHẤN MÁ HỒNG TỰ NHIÊN]</div>', unsafe_allow_html=True)
        blush_chk = st.checkbox("Bật má hồng ảo", value=bool(st.session_state.makeup_enable_blush), key="chk_blush")
        st.session_state.makeup_enable_blush = blush_chk

        blush_color_options = list(palette_blush.keys())
        blush_color_sel = st.selectbox(
            "Màu phấn má hồng:",
            blush_color_options,
            index=blush_color_options.index(st.session_state.makeup_blush_color) if st.session_state.makeup_blush_color in blush_color_options else 0,
            disabled=not blush_chk,
            key="sb_blush_color"
        )
        st.session_state.makeup_blush_color = blush_color_sel

        blush_alpha_sel = st.slider(
            "Độ đậm má hồng",
            0.05, 0.85, float(st.session_state.makeup_blush_alpha), 0.05,
            disabled=not blush_chk,
            key="sl_blush_alpha"
        )
        st.session_state.makeup_blush_alpha = blush_alpha_sel

    with col_view4:
        viewport_t4 = st.empty()

elif is_tab5:
    st.markdown(clean_html("""
    <div class="dip-theory-card">
        <div class="dip-theory-header">Bản chất Xử lý ảnh số (DIP) - Phân hệ 5: Điều khiển tương tác qua Ánh mắt (Iris &amp; Gaze Tracking)</div>
        <ul class="dip-theory-list">
            <li><b>Ước lượng tâm con ngươi (Iris Center Localization):</b> Trích xuất tọa độ 4 mốc mống mắt trái (<b>468 - 472</b>) và phải (<b>473 - 477</b>), đo tỷ lệ vị trí mống mắt so với khóe mắt trong/ngoài để xác định vector hướng nhìn 2D (Gaze Vector).</li>
            <li><b>Phát hiện kích hoạt cạnh nhấp mắt (Edge Triggered Blink):</b> Phát hiện xung nhắm mắt nhanh (1 - 12 frames) để mô phỏng sự kiện nhấp chuột ảo (Virtual Click) không chạm tay.</li>
        </ul>
    </div>
    """), unsafe_allow_html=True)
    col_view5, col_stat5 = st.columns([2.5, 1.5])
    with col_view5:
        viewport_t5 = st.empty()
    with col_stat5:
        st.markdown('<div class="section-panel-title">[HƯỚNG NHÌN ÁNH MẮT]</div>', unsafe_allow_html=True)
        gaze_direction_box = st.empty()
        st.markdown('<div class="section-panel-title">[TƯƠNG TÁC GIAO DIỆN (INTERACTION)]</div>', unsafe_allow_html=True)
        gaze_action_box = st.empty()
        st.markdown('<div class="section-panel-title">[TRẠNG THÁI NHẤP CHUỘT ẢO]</div>', unsafe_allow_html=True)
        gaze_click_result = st.empty()

elif is_tab6:
    st.markdown(clean_html("""
    <div class="dip-theory-card">
        <div class="dip-theory-header">Bản chất Xử lý ảnh số (DIP) - Phân hệ 6: Biến hình &amp; Hoán đổi khuôn mặt (Face Morphing &amp; Face Swapping)</div>
        <ul class="dip-theory-list">
            <li><b>Chia lưới tam giác Delaunay (Delaunay Triangulation - cv2.Subdiv2D):</b> Phân rã 79 điểm mốc giải phẫu khuôn mặt thành mạng lưới các tam giác không chồng lấp thỏa mãn tiêu chuẩn đường tròn ngoại tiếp rỗng (Empty Circumcircle Property).</li>
            <li><b>Biến đổi Affine từng mảnh (Piece-wise Affine Triangle Warping):</b> Nội suy tọa độ hình học trung gian P_m = (1 - α)·P_dst + α·P_src và giải ma trận Affine 2×3 (<b>cv2.getAffineTransform</b>, <b>cv2.warpAffine</b>) cho từng cặp tam giác tương ứng.</li>
            <li><b>Hòa trộn biên Poisson &amp; Chuyển giao màu CIELAB (Seamless Cloning &amp; Reinhard Color Transfer):</b> Cân bằng kỳ vọng và độ lệch chuẩn 3 kênh L*a*b* (<b>cv2.meanStdDev</b>) kết hợp mặt nạ bao lồi <b>cv2.convexHull</b> làm mềm biên Gaussian hoặc giải phương trình đạo hàm bậc hai Poisson <b>cv2.seamlessClone</b>.</li>
        </ul>
    </div>
    """), unsafe_allow_html=True)
    col_ctrl6, col_view6 = st.columns([1.3, 2.7])
    with col_ctrl6:
        st.markdown('<div class="section-panel-title">[CHỌN KHUÔN MẶT NGUỒN (SOURCE FACE)]</div>', unsafe_allow_html=True)
        src_opts = list(MORPH_SOURCE_PRESETS.keys()) + ["Tải ảnh khuôn mặt nguồn riêng"]
        src_idx = src_opts.index(st.session_state.morph_source_choice) if st.session_state.morph_source_choice in src_opts else 0
        src_sel = st.selectbox(
            "Khuôn mặt mẫu dùng để hoán đổi / biến hình:",
            src_opts,
            index=src_idx,
            key="sb_morph_source_choice"
        )
        st.session_state.morph_source_choice = src_sel

        if src_sel == "Tải ảnh khuôn mặt nguồn riêng":
            custom_src_file = st.file_uploader("Tải ảnh chân dung khuôn mặt nguồn (.jpg / .png):", type=["jpg", "jpeg", "png"], key="up_morph_src")
            if custom_src_file is not None:
                c_bytes = custom_src_file.read()
                st.session_state.morph_custom_src_bgr = cv2.imdecode(np.frombuffer(c_bytes, np.uint8), cv2.IMREAD_COLOR)

        src_preview_bgr, src_preview_lm = get_morph_source_data()
        morph_preview_holder = st.empty()
        if src_preview_bgr is not None:
            prev_vis = src_preview_bgr.copy()
            if src_preview_lm and st.session_state.morph_show_delaunay:
                pts_s = np.array([src_preview_lm[i] for i in engine.MORPH_KEY_INDICES], dtype=np.float32)
                tris_s = engine.compute_delaunay_triangles(pts_s, prev_vis.shape[1], prev_vis.shape[0])
                for i1, i2, i3 in tris_s:
                    cv2.line(prev_vis, tuple(pts_s[i1].astype(int)), tuple(pts_s[i2].astype(int)), (0, 242, 254), 1, cv2.LINE_AA)
                    cv2.line(prev_vis, tuple(pts_s[i2].astype(int)), tuple(pts_s[i3].astype(int)), (0, 242, 254), 1, cv2.LINE_AA)
                    cv2.line(prev_vis, tuple(pts_s[i3].astype(int)), tuple(pts_s[i1].astype(int)), (0, 242, 254), 1, cv2.LINE_AA)
            b64_src = encode_bgr_to_b64(prev_vis, quality=85)
            render_aligned_face_card(morph_preview_holder, b64_src, caption=f"Khuôn mặt Nguồn ({'Đã trích xuất 478 mốc' if src_preview_lm else 'Chưa thấy mặt'})", width=185)

        st.markdown("---")
        st.markdown('<div class="section-panel-title">[SIÊU THAM SỐ MORPHING &amp; SWAPPING]</div>', unsafe_allow_html=True)
        m_alpha = st.slider(
            "Trọng số biến hình / hoán đổi Alpha (0.0 = Mặt gốc, 0.5 = Lai 50/50, 1.0 = Hoán đổi 100%):",
            0.0, 1.0, float(st.session_state.morph_alpha), 0.05,
            key="sl_morph_alpha"
        )
        st.session_state.morph_alpha = m_alpha

        blend_opts = [
            "LAB_FEATHER (Reinhard CIELAB + Gaussian Feather - 30 FPS)",
            "POISSON_CLONE (Trộn miền đạo hàm Poisson cv2.seamlessClone)",
            "FULL_MORPH (Nội suy hình học cả 2 khuôn mặt theo Alpha)"
        ]
        b_idx = blend_opts.index(st.session_state.morph_blend_mode) if st.session_state.morph_blend_mode in blend_opts else 0
        b_sel = st.selectbox("Thuật toán hòa trộn biên:", blend_opts, index=b_idx, key="sb_morph_blend")
        st.session_state.morph_blend_mode = b_sel

        del_chk = st.checkbox("Hiển thị lưới tam giác Delaunay (Subdiv2D)", value=bool(st.session_state.morph_show_delaunay), key="chk_morph_delaunay")
        st.session_state.morph_show_delaunay = del_chk

        morph_metric_box = st.empty()
        morph_metric_box.markdown(
            build_metric_card(
                "Cấu trúc lưới Delaunay 2D",
                f"Alpha = {m_alpha:.2f}",
                "79 đỉnh giải phẫu | ~138 tam giác Affine | Hòa trộn màu CIELAB / Poisson",
                pct=m_alpha * 100.0,
                bar_color="#00f2fe",
                badge_text="DELAUNAY"
            ),
            unsafe_allow_html=True
        )
    with col_view6:
        viewport_t6 = st.empty()

elif is_all:
    st.markdown('<div class="section-panel-title">[TOÀN CẢNH] CHẠY ĐỒNG THỜI CẢ 6 PHÂN HỆ ĐỂ TRÌNH DIỄN ĐA NHIỆM</div>', unsafe_allow_html=True)
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        st.markdown("**1. Giám sát DMS**")
        viewport_t1 = st.empty()
    with row1_c2:
        st.markdown("**2. Điểm danh Liveness**")
        viewport_t2 = st.empty()
    with row1_c3:
        st.markdown("**3. Thử kính AR 3D**")
        viewport_t3 = st.empty()

    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1:
        st.markdown("**4. Trang điểm Makeup**")
        viewport_t4 = st.empty()
    with row2_c2:
        st.markdown("**5. Tương tác Ánh mắt**")
        viewport_t5 = st.empty()
    with row2_c3:
        st.markdown("**6. Face Morph &amp; Swap**")
        viewport_t6 = st.empty()

# -------------------------------------------------------------
# BỘ ĐỌC CAMERA ĐA LUỒNG THỰC THỤ TRIỆT TIÊU ĐỘ TRỄ (TRUE ASYNC THREADED CAMERA)
# -------------------------------------------------------------
class LiveCameraStream:
    """
    Bộ quản lý Camera trực tiếp đa luồng bất đồng bộ thực thụ (True Asynchronous Camera Grabber):
    1. Khắc phục triệt để lỗi đứng hình ở frame đầu tiên: Thiết lập cấu hình phần cứng (Buffer=1,
       Width=640, Height=480, FPS) TRƯỚC khi đọc frame; nếu codec MJPG không tương thích với webcam
       (làm cap.read() trả về False), tự động fallback về codec mặc định của thiết bị.
    2. Khắc phục lỗi khóa thiết bị (Device Lock Leak): Luôn gọi cap.release() giải phóng phần cứng
       cũ trước khi mở lại kết nối mới.
    3. Luồng đọc nền chuyên dụng (Background Daemon Thread): Liên tục hút frame mới nhất từ driver
       camera vào RAM với bộ đếm frame_seq tăng dần, giúp luồng chính Streamlit đọc frame trong 0ms
       mà không bao giờ bị nghẽn I/O hay xử lý lặp lại cùng một frame cũ.
    """
    def __init__(self, cam_idx=0, cam_drv="DirectShow (Windows)", target_fps=30):
        self.cam_idx = int(cam_idx)
        self.cam_drv = str(cam_drv)
        self.target_fps = int(target_fps)
        self.cap = None
        self.latest_frame = None
        self.frame_seq = 0
        self.last_frame_ts = 0.0
        self.last_checksum = None
        self.identical_streak = 0
        self.stopped = False
        self.fail_streak = 0
        self.lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._thread = None

        self._open_device()
        if self.cap is not None and self.cap.isOpened():
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def _safe_release_internal(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _try_configure_and_verify(self, backend, use_mjpg=True):
        cap = None
        try:
            cap = cv2.VideoCapture(self.cam_idx, backend) if backend is not None else cv2.VideoCapture(self.cam_idx)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                return None, None

            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            if use_mjpg:
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except Exception:
                pass

            try:
                cap.set(cv2.CAP_PROP_FPS, min(60, self.target_fps))
            except Exception:
                pass

            for _ in range(5):
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    return cap, test_frame
                time.sleep(0.015)

            cap.release()
        except Exception:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        return None, None

    def _open_device(self):
        with self._io_lock:
            self._safe_release_internal()

            if sys.platform == "win32":
                primary = cv2.CAP_DSHOW if "DirectShow" in self.cam_drv and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
                candidate_backends = [primary]
                if hasattr(cv2, "CAP_MSMF") and cv2.CAP_MSMF not in candidate_backends:
                    candidate_backends.append(cv2.CAP_MSMF)
                if hasattr(cv2, "CAP_DSHOW") and cv2.CAP_DSHOW not in candidate_backends:
                    candidate_backends.append(cv2.CAP_DSHOW)
                if cv2.CAP_ANY not in candidate_backends:
                    candidate_backends.append(cv2.CAP_ANY)
            else:
                candidate_backends = [cv2.CAP_ANY, None]

            for backend in candidate_backends:
                for use_mjpg in (True, False):
                    cap, first_frame = self._try_configure_and_verify(backend, use_mjpg=use_mjpg)
                    if cap is not None and first_frame is not None:
                        self.cap = cap
                        with self.lock:
                            self.latest_frame = first_frame
                            self.frame_seq += 1
                            self.last_frame_ts = time.time()
                            self.fail_streak = 0
                        return True
            return False

    def _reader_loop(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                ok = self._open_device()
                if not ok:
                    time.sleep(0.25)
                continue

            try:
                with self._io_lock:
                    if self.cap is not None and self.cap.isOpened():
                        ret, frame = self.cap.read()
                    else:
                        ret, frame = False, None
            except Exception:
                ret, frame = False, None

            if ret and frame is not None and frame.size > 0:
                # Kiểm tra chữ ký điểm ảnh O(1) phát hiện trường hợp driver USB trả về cùng 1 frame tĩnh bị treo
                sig = int(frame[::32, ::40, 0].sum()) ^ (int(frame[::32, ::40, 1].sum()) << 1)
                if self.last_checksum is not None and sig == self.last_checksum:
                    self.identical_streak += 1
                else:
                    self.identical_streak = 0
                    self.last_checksum = sig

                if self.identical_streak > 75:
                    # Driver camera bị kẹt bộ đệm tĩnh -> tự động khởi động lại thiết bị phần cứng
                    self.identical_streak = 0
                    self._open_device()
                    time.sleep(0.1)
                    continue

                with self.lock:
                    self.latest_frame = frame
                    self.frame_seq += 1
                    self.last_frame_ts = time.time()
                    self.fail_streak = 0
                time.sleep(0.002)
            else:
                self.fail_streak += 1
                if self.fail_streak > 45:
                    self.fail_streak = 0
                    self._open_device()
                    time.sleep(0.15)
                else:
                    time.sleep(0.01)

    def read(self):
        if self.stopped:
            return False, None
        with self.lock:
            if self.latest_frame is not None and (time.time() - self.last_frame_ts) < 2.0:
                return True, self.latest_frame.copy()
        return False, None

    def read_new(self, last_seq, timeout=0.05):
        """
        Chỉ trả về is_fresh=True khi thực sự có frame MỚI từ phần cứng (self.frame_seq > last_seq).
        Tuyệt đối KHÔNG trả lại frame cũ khi hết timeout để triệt tiêu hoàn toàn lỗi đứng hình sau vài phút.
        Nếu phần cứng ngưng cấp frame mới quá 1.8 giây, trả về ret=False để Watchdog tự kết nối lại.
        """
        if self.stopped:
            return False, None, last_seq, False
        deadline = time.time() + timeout
        while not self.stopped:
            with self.lock:
                if self.latest_frame is not None and self.frame_seq > last_seq:
                    return True, self.latest_frame.copy(), self.frame_seq, True
            if time.time() >= deadline:
                break
            time.sleep(0.004)

        with self.lock:
            age = time.time() - self.last_frame_ts
            if self.latest_frame is not None and age <= 1.8:
                # Camera vẫn sống nhưng chưa tới chu kỳ xuất frame mới -> báo is_fresh=False để bỏ qua vòng lặp thừa
                return True, None, self.frame_seq, False
        # Quá 1.8s không có frame mới -> báo lỗi để kích hoạt Watchdog mở lại cổng Camera
        return False, None, last_seq, False

    def isOpened(self):
        with self.lock:
            has_frame = self.latest_frame is not None
        return (not self.stopped) and ((self.cap is not None and self.cap.isOpened()) or has_frame)

    def release(self):
        self.stopped = True
        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.35)
            except Exception:
                pass
        with self._io_lock:
            self._safe_release_internal()


def get_camera_stream(source_type, cam_idx, cam_drv, video_path, target_fps, force_new=False):
    is_live = "Camera trực tiếp" in source_type or "Camera máy chủ" in source_type
    need_new = force_new
    if st.session_state.cap is None:
        need_new = True
    elif st.session_state.active_source_type != source_type:
        need_new = True
    elif is_live and (st.session_state.active_cam_idx != cam_idx or st.session_state.get("active_cam_drv") != cam_drv):
        need_new = True
    elif not is_live and st.session_state.get("active_video_path") != video_path:
        need_new = True
    elif not st.session_state.cap.isOpened():
        need_new = True

    if need_new:
        if st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except Exception:
                pass
            st.session_state.cap = None
            time.sleep(0.05)

        if not is_live:
            if not video_path:
                return None
            cap = cv2.VideoCapture(video_path)
        else:
            cap = LiveCameraStream(cam_idx, cam_drv, target_fps)

        st.session_state.cap = cap
        st.session_state.active_source_type = source_type
        st.session_state.active_cam_idx = cam_idx
        st.session_state.active_cam_drv = cam_drv
        st.session_state.active_video_path = video_path

    return st.session_state.cap

# -------------------------------------------------------------
# MÀN HÌNH CHỜ RADAR QUANG HỌC KHI CAMERA Ở TRẠNG THÁI STANDBY
# -------------------------------------------------------------
standby_card = clean_html("""
<div class="standby-hud-box">
    <div class="radar-ring"></div>
    <div class="standby-title">[ CAMERA STANDBY // HỆ THỐNG SẴN SÀNG ]</div>
    <div class="standby-desc">
        Vui lòng tích chọn ô <b>"Khởi động Camera"</b> hoặc <b>"Bắt đầu phát Video"</b> ở thanh điều khiển bên trái để bắt đầu.<br>
        Hệ thống tối ưu tốc độ thời gian thực với kiến trúc <b>Multi-Threaded Grabber</b> &amp; <b>Base64 Zero-Lag Stream</b>.
    </div>
</div>
""")


def render_tab1_telemetry(metrics):
    """Cập nhật toàn bộ cột chỉ số Telemetry của Tab 1 với thanh Gauge Bar động và 0 lỗi thẻ div."""
    if metrics.get("is_dual_mode"):
        p0 = metrics.get("driver")
        p1 = metrics.get("codriver")
        if metrics["dual_alert_type"] in ["driver_drowsy", "driver_yawn", "driver_distracted"]:
            status_box_t1.markdown(
                build_status_banner("danger", f"[CẢNH BÁO 2 NGƯỜI] {metrics['dual_alert_text']}"),
                unsafe_allow_html=True
            )
        else:
            status_box_t1.markdown(
                build_status_banner("safe", "[CABIN 2 NGƯỜI] TÀI XẾ & PHỤ XE AN TOÀN - TỈNH TÁO"),
                unsafe_allow_html=True
            )

        ear_pct = min(100.0, (p0["ear"] / 0.35) * 100.0)
        ear_col = "#ef4444" if p0["is_drowsy"] else "#00f2fe"
        ear_metric_t1.markdown(
            build_metric_card(
                "Tỷ lệ mắt EAR (Cabin 2 Người)",
                f"TX: {p0['ear']:.3f} | PX: {p1['ear']:.3f}",
                f"Tài xế ({'Ngủ' if p0['is_drowsy'] else 'Tỉnh'}) | Phụ xe ({'Ngủ' if p1['is_drowsy'] else 'Tỉnh'})",
                pct=ear_pct,
                bar_color=ear_col,
                badge_text="DUAL EAR"
            ),
            unsafe_allow_html=True
        )

        mar_pct = min(100.0, (p0["mar"] / 0.85) * 100.0)
        mar_col = "#f59e0b" if p0["is_yawning"] else "#00e676"
        mar_metric_t1.markdown(
            build_metric_card(
                "Tỷ lệ mở miệng MAR (Cabin 2 Người)",
                f"TX: {p0['mar']:.3f} | PX: {p1['mar']:.3f}",
                f"Tài xế ({'Ngáp' if p0['is_yawning'] else 'Bình thường'}) | Phụ xe ({'Ngáp' if p1['is_yawning'] else 'Bình thường'})",
                pct=mar_pct,
                bar_color=mar_col,
                badge_text="DUAL MAR"
            ),
            unsafe_allow_html=True
        )

        perclos_col = "#ef4444" if p0["perclos"] >= engine.perclos_thresh else "#38bdf8"
        perclos_metric_t1.markdown(
            build_metric_card(
                "Chỉ số vi giấc ngủ PERCLOS",
                f"TX: {p0['perclos']:.1f}% | PX: {p1['perclos']:.1f}%",
                f"Ngưỡng NHTSA: {engine.perclos_thresh:.0f}% (Cửa sổ 90 frames)",
                pct=min(100.0, p0["perclos"] * 2.5),
                bar_color=perclos_col,
                badge_text="NHTSA"
            ),
            unsafe_allow_html=True
        )

        pose_metric_t1.markdown(
            build_metric_card(
                "Góc đầu 3D (Cabin 2 Người)",
                f"TX Yaw: {p0['yaw']:+.1f}° | PX Yaw: {p1['yaw']:+.1f}°",
                f"Pitch: TX {p0['pitch']:+.1f}° | PX {p1['pitch']:+.1f}°",
                pct=min(100.0, (abs(p0["yaw"]) / 45.0) * 100.0),
                bar_color="#f59e0b" if p0["is_distracted"] else "#00f2fe",
                badge_text="3D POSE"
            ),
            unsafe_allow_html=True
        )
    else:
        if metrics["is_yawning"]:
            status_box_t1.markdown(
                build_status_banner("warn", "[CẢNH BÁO] TÀI XẾ CÓ BIỂU HIỆN BUỒN NGỦ (NGÁP NGỦ)!"),
                unsafe_allow_html=True
            )
        elif metrics["is_drowsy"]:
            status_box_t1.markdown(
                build_status_banner("danger", "[NGUY HIỂM] TÀI XẾ ĐANG NGỦ GẬT!"),
                unsafe_allow_html=True
            )
        elif metrics["is_distracted"]:
            status_box_t1.markdown(
                build_status_banner("warn", f"[MẤT TẬP TRUNG] {metrics['distraction_type']}"),
                unsafe_allow_html=True
            )
        else:
            status_box_t1.markdown(
                build_status_banner("safe", "[ỔN ĐỊNH] TÀI XẾ TỈNH TÁO BÌNH THƯỜNG"),
                unsafe_allow_html=True
            )

        ear_pct = min(100.0, (metrics["ear"] / 0.35) * 100.0)
        ear_col = "#ef4444" if metrics["ear"] < engine.ear_thresh else "#00f2fe"
        ear_metric_t1.markdown(
            build_metric_card(
                "Tỷ lệ mắt EAR (Soukupova)",
                f"{metrics['ear']:.3f}",
                f"Trái: {metrics['ear_left']:.3f} | Phải: {metrics['ear_right']:.3f}",
                pct=ear_pct,
                bar_color=ear_col,
                badge_text=f"Ngưỡng: {engine.ear_thresh:.2f}"
            ),
            unsafe_allow_html=True
        )

        mar_pct = min(100.0, (metrics["mar"] / 0.85) * 100.0)
        mar_col = "#f59e0b" if metrics["mar"] > engine.mar_thresh else "#00e676"
        mar_metric_t1.markdown(
            build_metric_card(
                "Tỷ lệ mở miệng MAR",
                f"{metrics['mar']:.3f}",
                "Đo độ giãn cách dọc / ngang vùng môi",
                pct=mar_pct,
                bar_color=mar_col,
                badge_text=f"Ngưỡng: {engine.mar_thresh:.2f}"
            ),
            unsafe_allow_html=True
        )

        perclos_col = "#ef4444" if metrics["perclos"] >= engine.perclos_thresh else "#38bdf8"
        perclos_metric_t1.markdown(
            build_metric_card(
                "Chỉ số vi giấc ngủ PERCLOS",
                f"{metrics['perclos']:.1f}%",
                "Tỷ lệ nhắm mắt tích lũy trên cửa sổ trượt 90f",
                pct=min(100.0, metrics["perclos"] * 2.5),
                bar_color=perclos_col,
                badge_text=f"NHTSA: {engine.perclos_thresh:.0f}%"
            ),
            unsafe_allow_html=True
        )

        pose_metric_t1.markdown(
            build_metric_card(
                "Góc đầu 3D (Euler Angles)",
                f"P:{metrics['pitch']:+.1f}° | Y:{metrics['yaw']:+.1f}°",
                f"Roll: {metrics['roll']:+.1f}° | Hiệu chuẩn 0: {'ĐÃ BẬT' if metrics['is_calibrated'] else 'CHƯA'}",
                pct=min(100.0, (abs(metrics["yaw"]) / 45.0) * 100.0),
                bar_color="#f59e0b" if metrics["is_distracted"] else "#9b51e0",
                badge_text="solvePnP"
            ),
            unsafe_allow_html=True
        )


# -------------------------------------------------------------
# Đồng bộ hóa tức thời các thông số kính AR / Makeup sang WebRTC Processor nếu đang chạy
_w_ctx = st.session_state.get("active_webrtc_ctx")
if _w_ctx is not None and getattr(_w_ctx, "video_processor", None) is not None:
    _vp = _w_ctx.video_processor
    _vp.zoom_scale = float(cam_zoom_scale)
    _vp.pan_x = float(cam_pan_x)
    _vp.cabin_mode = str(cabin_mode_code)
    _vp.draw_box = bool(show_face_box)
    _vp.draw_contours = bool(show_mesh_contours)
    _vp.draw_vectors = bool(show_geo_vectors)
    _vp.is_t1 = bool(is_tab1)
    _vp.is_t2 = bool(is_tab2)
    _vp.is_t3 = bool(is_tab3)
    _vp.is_t4 = bool(is_tab4)
    _vp.is_t5 = bool(is_tab5)
    _vp.is_t6 = bool(is_tab6)
    _vp.is_all_tabs = bool(is_all)
    _m_img_now, _m_lm_now = get_morph_source_data()
    _vp.morph_src_img = _m_img_now
    _vp.morph_src_lm = _m_lm_now
    _vp.morph_alpha = float(st.session_state.get("morph_alpha", 1.0))
    _vp.morph_mode = str(st.session_state.get("morph_blend_mode", "LAB_FEATHER"))
    _vp.morph_delaunay = bool(st.session_state.get("morph_show_delaunay", False))
    _sel_g_now = st.session_state.get("selected_glasses_choice", "")
    _vp.glasses_bgra = st.session_state.get("glasses_db", {}).get(_sel_g_now)
    _lip_n_now = st.session_state.get("makeup_lip_color", "Hồng Cherry Tươi Trẻ")
    _vp.lip_bgr = palette_lip.get(_lip_n_now, (20, 15, 235))
    _vp.lip_alpha = float(st.session_state.get("makeup_lip_alpha", 0.55))
    _blush_n_now = st.session_state.get("makeup_blush_color", "Hồng Baby Xinh Xắn")
    _vp.blush_bgr = palette_blush.get(_blush_n_now, (160, 110, 255)) if st.session_state.get("makeup_enable_blush", True) else None
    _vp.blush_alpha = float(st.session_state.get("makeup_blush_alpha", 0.35))
    _cs_c_now = str(st.session_state.get("makeup_color_space", "HYBRID"))
    _vp.cs_m = "LAB" if _cs_c_now.startswith("LAB") else ("HLS" if _cs_c_now.startswith("HLS") else "HYBRID")

# VÒNG LẶP XỬ LÝ CAMERA TỐC ĐỘ CAO (HIGH FPS REAL-TIME LOOP)
# -------------------------------------------------------------
if input_source.startswith("Tải lên tệp hình ảnh") or input_source.startswith("Camera thiết bị"):
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

    if run_camera:
        if uploaded_img_bgr is None:
            if input_source.startswith("Camera thiết bị"):
                st.info("Vui lòng nhìn vào khung ngắm camera bên trên và bấm nút 'Chụp ảnh' (Take Photo) để AI phân tích.")
            else:
                st.info("Vui lòng chọn hoặc tải lên tệp hình ảnh (.jpg / .png) từ Sidebar để tiến hành phân tích.")
        else:
            frame = uploaded_img_bgr.copy()
            h_f, w_f = frame.shape[:2]
            if w_f > 640:
                new_h = int(h_f * (640.0 / w_f))
                frame = cv2.resize(frame, (640, new_h), interpolation=cv2.INTER_LINEAR)

            if cam_zoom_scale != 1.0 or cam_pan_x != 0.0:
                frame = apply_camera_scale(frame, scale=cam_zoom_scale, pan_x=cam_pan_x)

            need_dms_hud = is_tab1 or is_all
            metrics, annotated_dms = engine.process_frame(
                frame,
                is_static_image=True,
                cabin_mode=cabin_mode_code,
                draw_box=show_face_box,
                draw_contours=show_mesh_contours,
                draw_vectors=show_geo_vectors,
                render_hud=need_dms_hud
            )
            landmarks_2d = metrics.get("landmarks_2d")

            if is_tab1:
                if metrics.get("is_dual_mode") and metrics.get("dual_alert_type") in ["driver_drowsy", "driver_yawn", "driver_distracted"]:
                    if metrics["dual_alert_type"] == "driver_drowsy":
                        trigger_system_alert("dual_drowsy")
                    elif metrics["dual_alert_type"] == "driver_yawn":
                        trigger_system_alert("yawn")
                    elif metrics["dual_alert_type"] == "driver_distracted":
                        trigger_system_alert("distracted")
                else:
                    if metrics["is_yawning"]:
                        trigger_system_alert("yawn")
                    elif metrics["is_drowsy"]:
                        trigger_system_alert("drowsy")
                    elif metrics["is_distracted"]:
                        trigger_system_alert("distracted")
                    else:
                        if input_source.startswith("Camera thiết bị") or input_source.startswith("Tải lên tệp hình ảnh"):
                            st.info(f"**[KẾT QUẢ PHÂN TÍCH ẢNH]: TÀI XẾ TỈNH TÁO BÌNH THƯỜNG**\n\n"
                                    f"- Mắt mở to (EAR: {metrics['ear']:.3f} >= {engine.ear_thresh:.2f})\n"
                                    f"- Miệng ngậm bình thường (MAR: {metrics['mar']:.3f} <= {engine.mar_thresh:.2f})\n"
                                    f"- Góc quay đầu nhìn thẳng (Yaw: {metrics['yaw']:+.1f}°, Roll: {metrics['roll']:+.1f}°)\n\n"
                                    f"- Trạng thái an toàn nên hệ thống không phát còi báo động. Bạn hãy thử chụp ảnh lúc **nhắm tịt mắt** (ngủ gật) hoặc **há to miệng** (ngáp) để kích hoạt còi, hoặc bấm các nút Test còi ở thanh âm thanh bên trên!")

                _, jpg_buf = cv2.imencode('.jpg', annotated_dms, [cv2.IMWRITE_JPEG_QUALITY, 90])
                safe_display_image(viewport_t1, jpg_buf.tobytes(), caption="Ảnh kiểm thử Phân hệ 1 (DMS)")
                render_tab1_telemetry(metrics)

            elif is_tab2:
                frame_t2 = frame.copy()
                if landmarks_2d:
                    frame_t2 = engine.draw_face_annotations(
                        frame_t2,
                        landmarks_2d,
                        {"liveness_success": True},
                        draw_box=True,
                        draw_contours=True,
                        draw_vectors=False,
                        badge_mode="biometric"
                    )
                    align_res = engine.align_face(frame, landmarks_2d)
                    aligned_crop = align_res[0] if isinstance(align_res, tuple) else align_res
                    liveness_instruction.markdown(
                        build_liveness_stepper(
                            3,
                            True,
                            "[ẢNH TĨNH] ĐÃ CĂN CHỈNH KHUÔN MẶT AFFINE THÀNH CÔNG!",
                            "Phép biến đổi cv2.getRotationMatrix2D & cv2.warpAffine (256x256)"
                        ),
                        unsafe_allow_html=True
                    )
                    liveness_progress.progress(100)
                    liveness_result_box.markdown(
                        build_status_banner("safe", "[HOÀN TẤT] XÁC THỰC SINH TRẮC HỌC KHUÔN MẶT"),
                        unsafe_allow_html=True
                    )
                    if aligned_crop is not None and isinstance(aligned_crop, np.ndarray) and aligned_crop.size > 0:
                        b64_crop = encode_bgr_to_b64(aligned_crop, quality=92)
                        st.session_state.liveness_aligned_b64 = b64_crop
                        render_aligned_face_card(liveness_face_crop, b64_crop)
                _, jpg_buf = cv2.imencode('.jpg', frame_t2, [cv2.IMWRITE_JPEG_QUALITY, 90])
                safe_display_image(viewport_t2, jpg_buf.tobytes(), caption="Ảnh kiểm thử Phân hệ 2 (Điểm danh)")

            elif is_tab3:
                frame_t3 = frame.copy()
                if landmarks_2d:
                    sel_g = st.session_state.selected_glasses_choice
                    current_glasses_bgra = st.session_state.glasses_db.get(sel_g)
                    if current_glasses_bgra is not None:
                        frame_t3 = engine.overlay_glasses_homography(frame_t3, landmarks_2d, current_glasses_bgra)
                    if show_mesh_contours:
                        frame_t3 = engine.draw_face_annotations(frame_t3, landmarks_2d, metrics, draw_box=False, draw_contours=True, draw_vectors=False)
                _, jpg_buf = cv2.imencode('.jpg', frame_t3, [cv2.IMWRITE_JPEG_QUALITY, 90])
                safe_display_image(viewport_t3, jpg_buf.tobytes(), caption="Ảnh kiểm thử Phân hệ 3 (Thử kính AR)")

            elif is_tab4:
                frame_t4 = frame.copy()
                if landmarks_2d:
                    lip_name = st.session_state.makeup_lip_color
                    sel_lip_bgr = palette_lip.get(lip_name, (20, 15, 235))
                    blush_name = st.session_state.makeup_blush_color
                    sel_blush_bgr = palette_blush.get(blush_name, (160, 110, 255)) if st.session_state.makeup_enable_blush else None
                    cs_choice = st.session_state.makeup_color_space
                    cs_m = "LAB" if cs_choice.startswith("LAB") else ("HLS" if cs_choice.startswith("HLS") else "HYBRID")
                    frame_t4 = engine.apply_makeup(
                        frame_t4,
                        landmarks_2d,
                        sel_lip_bgr,
                        lip_alpha=st.session_state.makeup_lip_alpha,
                        blush_bgr=sel_blush_bgr,
                        blush_alpha=st.session_state.makeup_blush_alpha,
                        color_space=cs_m
                    )
                    if show_mesh_contours:
                        frame_t4 = engine.draw_face_annotations(frame_t4, landmarks_2d, metrics, draw_box=False, draw_contours=True, draw_vectors=False)
                _, jpg_buf = cv2.imencode('.jpg', frame_t4, [cv2.IMWRITE_JPEG_QUALITY, 90])
                safe_display_image(viewport_t4, jpg_buf.tobytes(), caption="Ảnh kiểm thử Phân hệ 4 (Trang điểm)")

            elif is_tab5:
                frame_t5 = frame.copy()
                if landmarks_2d:
                    gaze_data = engine.analyze_gaze(landmarks_2d)
                    il = gaze_data["iris_left"]
                    ir = gaze_data["iris_right"]
                    gx = gaze_data["gaze_x"]
                    gy = gaze_data["gaze_y"]
                    h_dir = gaze_data["h_direction"]
                    r_l = gaze_data.get("ratio_l", gaze_data.get("h_ratio", 0.5))
                    r_r = gaze_data.get("ratio_r", gaze_data.get("h_ratio", 0.5))
                    cv2.circle(frame_t5, il, 4, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame_t5, ir, 4, (0, 255, 255), -1, cv2.LINE_AA)
                    arrow_len = 48.0
                    end_l = (int(il[0] + gx * arrow_len), int(il[1] + gy * arrow_len))
                    end_r = (int(ir[0] + gx * arrow_len), int(ir[1] + gy * arrow_len))
                    cv2.arrowedLine(frame_t5, il, end_l, (0, 242, 254), 2, cv2.LINE_AA, tipLength=0.28)
                    cv2.arrowedLine(frame_t5, ir, end_r, (0, 242, 254), 2, cv2.LINE_AA, tipLength=0.28)
                    cv2.putText(frame_t5, f"GAZE: {h_dir}", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                    gaze_direction_box.markdown(
                        build_metric_card(
                            "Vector ánh mắt & Mống mắt",
                            h_dir,
                            f"Tỷ lệ chiếu ngang Trái: {r_l:.2f} | Phải: {r_r:.2f}",
                            pct=float(gaze_data.get("h_ratio", 0.5)) * 100.0,
                            bar_color="#00f2fe",
                            badge_text="IRIS 468-477"
                        ),
                        unsafe_allow_html=True
                    )
                    gaze_action_box.markdown(
                        build_status_banner("safe", f"[ĐANG TRỎ] {st.session_state.gaze_selected_card}"),
                        unsafe_allow_html=True
                    )
                    gaze_click_result.markdown(
                        build_metric_card(
                            "Thao tác nhấp chuột ảo",
                            st.session_state.gaze_last_clicked_card,
                            "Kích hoạt qua xung chớp mắt Edge Triggered Blink",
                            pct=100,
                            bar_color="#10b981",
                            badge_text="VIRTUAL CLICK"
                        ),
                        unsafe_allow_html=True
                    )
                _, jpg_buf = cv2.imencode('.jpg', frame_t5, [cv2.IMWRITE_JPEG_QUALITY, 90])
                safe_display_image(viewport_t5, jpg_buf.tobytes(), caption="Ảnh kiểm thử Phân hệ 5 (Ánh mắt)")

            elif is_tab6:
                frame_t6 = frame.copy()
                src_img_m, src_lm_m = get_morph_source_data()
                num_tris = 0
                if landmarks_2d and src_img_m is not None and src_lm_m is not None:
                    frame_t6, num_tris = engine.swap_and_morph_faces(
                        frame_t6,
                        landmarks_2d,
                        src_img_m,
                        src_lm_m,
                        alpha=st.session_state.morph_alpha,
                        blend_mode=st.session_state.morph_blend_mode,
                        show_delaunay=st.session_state.morph_show_delaunay
                    )
                    morph_metric_box.markdown(
                        build_metric_card(
                            "Kết quả Chia lưới Delaunay & Biến đổi Affine",
                            f"{num_tris} Tam giác",
                            f"Alpha = {st.session_state.morph_alpha:.2f} | Chế độ: {st.session_state.morph_blend_mode[:12]}",
                            pct=st.session_state.morph_alpha * 100.0,
                            bar_color="#00e676",
                            badge_text="DELAUNAY OK"
                        ),
                        unsafe_allow_html=True
                    )
                _, jpg_buf = cv2.imencode('.jpg', frame_t6, [cv2.IMWRITE_JPEG_QUALITY, 90])
                safe_display_image(viewport_t6, jpg_buf.tobytes(), caption="Ảnh kiểm thử Phân hệ 6 (Biến hình & Hoán đổi khuôn mặt Delaunay)")

            elif is_all:
                _, b1 = cv2.imencode('.jpg', annotated_dms, [cv2.IMWRITE_JPEG_QUALITY, 85])
                safe_display_image(viewport_t1, b1.tobytes())
                safe_display_image(viewport_t2, b1.tobytes())
                safe_display_image(viewport_t3, b1.tobytes())
                safe_display_image(viewport_t4, b1.tobytes())
                safe_display_image(viewport_t5, b1.tobytes())
                f6_s = frame.copy()
                src_img_m, src_lm_m = get_morph_source_data()
                if landmarks_2d and src_img_m is not None and src_lm_m is not None:
                    f6_s, _ = engine.swap_and_morph_faces(f6_s, landmarks_2d, src_img_m, src_lm_m, alpha=1.0)
                _, b6 = cv2.imencode('.jpg', f6_s, [cv2.IMWRITE_JPEG_QUALITY, 85])
                safe_display_image(viewport_t6, b6.tobytes())

elif run_camera:
    is_live_cam = input_source.startswith("Camera trực tiếp") or input_source.startswith("Camera máy chủ")
    cap = get_camera_stream(input_source, cam_index, cam_driver, uploaded_video_path, target_hardware_fps)

    if cap is None or not cap.isOpened():
        st.error(f"Không thể kết nối với Camera ở cổng {cam_index}. Vui lòng kiểm tra lại thiết bị hoặc chuyển sang 'Driver: Mặc định (MSMF / Auto)'.")
    else:
        try:
            frame_idx = 0
            last_cam_seq = -1
            prev_frame_time = time.time()
            fps_display = 0.0
            consecutive_drop_count = 0

            last_t2_instr = None
            last_t2_prog = None
            last_t2_res = None

            while run_camera and cap is not None and cap.isOpened():
                loop_start_t = time.time()

                is_fresh_frame = True
                try:
                    if is_live_cam and hasattr(cap, "read_new"):
                        ret, frame, last_cam_seq, is_fresh_frame = cap.read_new(last_cam_seq, timeout=0.045)
                    else:
                        ret, frame = cap.read()
                except Exception:
                    ret, frame, is_fresh_frame = False, None, False

                if is_live_cam and ret and not is_fresh_frame:
                    # Chưa có frame phần cứng mới -> nhường CPU ngay, không vẽ lặp lại frame cũ
                    time.sleep(0.004)
                    continue

                if not ret or frame is None:
                    if not is_live_cam:
                        if loop_video and hasattr(cap, "set"):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break
                    else:
                        consecutive_drop_count += 1
                        if consecutive_drop_count < 20:
                            time.sleep(0.015)
                            continue
                        else:
                            cap = get_camera_stream(input_source, cam_index, cam_driver, uploaded_video_path, target_hardware_fps, force_new=True)
                            consecutive_drop_count = 0
                            time.sleep(0.1)
                            continue
                else:
                    consecutive_drop_count = 0

                frame_idx += 1
                curr_frame_time = time.time()
                dt = curr_frame_time - prev_frame_time
                prev_frame_time = curr_frame_time
                if frame_idx == 1 or frame_idx % 5 == 0:
                    inst_fps = 1.0 / max(1e-4, dt)
                    fps_display = inst_fps if fps_display == 0.0 else (0.7 * fps_display + 0.3 * inst_fps)

                if is_live_cam:
                    frame = cv2.flip(frame, 1)

                h_f, w_f = frame.shape[:2]
                if w_f > 640:
                    new_h = int(h_f * (640.0 / w_f))
                    frame = cv2.resize(frame, (640, new_h), interpolation=cv2.INTER_LINEAR)

                if cam_zoom_scale != 1.0 or cam_pan_x != 0.0:
                    frame = apply_camera_scale(frame, scale=cam_zoom_scale, pan_x=cam_pan_x)

                need_dms_hud = is_tab1 or is_all
                metrics, annotated_dms = engine.process_frame(
                    frame,
                    is_static_image=False,
                    cabin_mode=cabin_mode_code,
                    draw_box=show_face_box,
                    draw_contours=show_mesh_contours,
                    draw_vectors=show_geo_vectors,
                    render_hud=need_dms_hud
                )
                landmarks_2d = metrics.get("landmarks_2d")

                st.session_state.last_pitch = metrics.get("pitch_raw", 0.0)
                st.session_state.last_yaw = metrics.get("yaw_raw", 0.0)
                st.session_state.last_roll = metrics.get("roll_raw", 0.0)
                st.session_state.last_ear = metrics.get("ear", 0.28)

                # =========================================================
                # 1. XỬ LÝ DUY NHẤT TAB 1: GIÁM SÁT TÀI XẾ (DMS)
                # =========================================================
                if is_tab1:
                    if metrics.get("is_dual_mode") and metrics.get("dual_alert_type") in ["driver_drowsy", "driver_yawn", "driver_distracted"]:
                        if metrics["dual_alert_type"] == "driver_drowsy":
                            trigger_system_alert("dual_drowsy")
                        elif metrics["dual_alert_type"] == "driver_yawn":
                            trigger_system_alert("yawn")
                        elif metrics["dual_alert_type"] == "driver_distracted":
                            trigger_system_alert("distracted")
                    else:
                        if metrics["is_yawning"]:
                            trigger_system_alert("yawn")
                        elif metrics["is_drowsy"]:
                            trigger_system_alert("drowsy")
                        elif metrics["is_distracted"]:
                            trigger_system_alert("distracted")
                        else:
                            if st.session_state.get("browser_alert_active", False) and (time.time() - st.session_state.get("browser_alert_time", 0) > 5.0):
                                browser_audio_holder.empty()
                                st.session_state["browser_alert_active"] = False

                    cv2.putText(
                        annotated_dms,
                        f"FPS: {fps_display:.1f} (Target: {target_hardware_fps})",
                        (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    if fps_display < 15.0 and frame_idx > 8:
                        annotated_dms = engine.draw_lag_loading_hud(annotated_dms, fps=fps_display)

                    if frame_idx == 1 or frame_idx % 6 == 0:
                        render_tab1_telemetry(metrics)

                    _, jpg_buf = cv2.imencode('.jpg', annotated_dms, [cv2.IMWRITE_JPEG_QUALITY, 76])
                    safe_display_image(viewport_t1, jpg_buf.tobytes())

                # =========================================================
                # 2. XỬ LÝ DUY NHẤT TAB 2: ĐIỂM DANH & LIVENESS
                # =========================================================
                elif is_tab2:
                    frame_t2 = frame.copy()
                    if not landmarks_2d:
                        if st.session_state.liveness_stage in (1, 2):
                            st.session_state.liveness_lost_frames += 1
                            if st.session_state.liveness_lost_frames > 15:
                                reset_liveness_fsm()
                                st.session_state.liveness_spoof_warn = "[CẢNH BÁO CHỐNG GIAN LẬN] Phát hiện khuất mặt / gián đoạn giữa phiên! Đã đặt lại về Bước 1."
                    if landmarks_2d:
                        st.session_state.liveness_lost_frames = 0
                        curr_cx = (landmarks_2d[33][0] + landmarks_2d[263][0]) * 0.5 / max(1.0, float(frame_t2.shape[1]))
                        curr_cy = (landmarks_2d[33][1] + landmarks_2d[263][1]) * 0.5 / max(1.0, float(frame_t2.shape[0]))
                        if st.session_state.liveness_stage in (1, 2) and st.session_state.liveness_last_center is not None:
                            jump_dist = math.hypot(curr_cx - st.session_state.liveness_last_center[0], curr_cy - st.session_state.liveness_last_center[1])
                            if jump_dist > 0.35:
                                reset_liveness_fsm()
                                st.session_state.liveness_spoof_warn = "[CẢNH BÁO CHỐNG GIAN LẬN] Phát hiện dịch chuyển nhảy vọt (nghi vấn tráo ảnh)! Đã đặt lại về Bước 1."
                        st.session_state.liveness_last_center = (curr_cx, curr_cy)
                        pt_l = landmarks_2d[33]
                        pt_r = landmarks_2d[263]
                        cv2.line(frame_t2, pt_l, pt_r, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(frame_t2, pt_l, 4, (0, 255, 0), -1)
                        cv2.circle(frame_t2, pt_r, 4, (0, 255, 0), -1)

                        metrics["liveness_success"] = (st.session_state.liveness_stage == 3)
                        if show_face_box:
                            frame_t2 = engine.draw_face_annotations(
                                frame_t2,
                                landmarks_2d,
                                metrics,
                                draw_box=True,
                                draw_contours=show_mesh_contours,
                                draw_vectors=False,
                                badge_mode="biometric"
                            )

                        detail_title = ""
                        detail_sub = ""
                        new_prog = last_t2_prog
                        new_res = last_t2_res

                        if not st.session_state.liveness_active_toggle and st.session_state.liveness_stage < 3:
                            detail_title = "[CHẾ ĐỘ CHỜ] Tích chọn 'Bật kiểm tra Liveness' ở trên để bắt đầu"
                            detail_sub = f"EAR: {metrics['ear']:.2f} | Yaw: {metrics['yaw']:+.1f}° | Pitch: {metrics['pitch']:+.1f}°"
                            new_prog = 0
                            new_res = ""
                        elif st.session_state.liveness_stage == 0:
                            if metrics["ear"] < engine.ear_thresh:
                                st.session_state.blink_closed_frames += 1
                            else:
                                if 1 <= st.session_state.blink_closed_frames <= 18:
                                    st.session_state.blink_completed_count += 1
                                st.session_state.blink_closed_frames = 0

                            detail_title = f"Thử thách 1: Vui lòng CHỚP MẮT TỰ NHIÊN ({st.session_state.blink_completed_count}/2 lần)"
                            lap_val = engine.compute_face_laplacian_sharpness(frame, landmarks_2d)
                            detail_sub = f"EAR: {metrics['ear']:.2f} (< {engine.ear_thresh:.2f}) | Độ nét Laplacian: {lap_val:.1f}"
                            new_prog = min(50, st.session_state.blink_completed_count * 25)
                            new_res = ""
                            if st.session_state.blink_completed_count >= 2:
                                st.session_state.liveness_stage = 1
                                trigger_system_alert("success")

                        elif st.session_state.liveness_stage == 1:
                            if abs(metrics["yaw"]) > 15.0:
                                st.session_state.turn_counter_liveness += 1
                            else:
                                st.session_state.turn_counter_liveness = max(0, st.session_state.turn_counter_liveness - 1)

                            detail_title = f"Thử thách 2: Vui lòng QUAY MẶT NGHIÊNG TRÁI/PHẢI ({st.session_state.turn_counter_liveness}/5 frames)"
                            detail_sub = f"Góc nghiêng Yaw hiện tại: {abs(metrics['yaw']):.1f}° (Yêu cầu > 15.0°)"
                            new_prog = min(70, 50 + int(st.session_state.turn_counter_liveness * 4))
                            new_res = ""
                            if st.session_state.turn_counter_liveness >= 5:
                                st.session_state.liveness_stage = 2
                                st.session_state.turn_counter_liveness = 0
                                trigger_system_alert("success")

                        elif st.session_state.liveness_stage == 2:
                            if abs(metrics["yaw"]) < 12.0 and abs(metrics["pitch"]) < 14.0:
                                st.session_state.straight_counter_liveness += 1
                            else:
                                st.session_state.straight_counter_liveness = max(0, st.session_state.straight_counter_liveness - 1)

                            detail_title = f"Thử thách 3: Giữ thẳng mặt để căn chỉnh Affine ({st.session_state.straight_counter_liveness}/4 frames)"
                            detail_sub = f"Yaw: {metrics['yaw']:+.1f}° | Pitch: {metrics['pitch']:+.1f}° (Yêu cầu nhìn thẳng)"
                            new_prog = min(98, 70 + st.session_state.straight_counter_liveness * 7)
                            new_res = ""
                            if st.session_state.straight_counter_liveness >= 4:
                                lap_final = engine.compute_face_laplacian_sharpness(frame, landmarks_2d)
                                st.session_state.liveness_laplacian_score = lap_final
                                st.session_state.liveness_stage = 3
                                st.session_state.liveness_success = True
                                st.session_state.liveness_spoof_warn = ""
                                aligned_crop, _ = engine.align_face(frame, landmarks_2d, 256, 256)
                                st.session_state.liveness_aligned_face = aligned_crop
                                st.session_state.liveness_aligned_b64 = encode_bgr_to_b64(aligned_crop, quality=92)
                                st.session_state.liveness_crop_rendered = False
                                trigger_system_alert("success")

                        if st.session_state.liveness_stage == 3:
                            detail_title = "[XÁC THỰC THÀNH CÔNG] ĐÃ ĐIỂM DANH SINH TRẮC HỌC!"
                            lap_s = st.session_state.get("liveness_laplacian_score", 0.0)
                            detail_sub = f"Hoàn tất 3/3 bước Liveness | Chuẩn hóa Affine 256x256 | Độ nét Laplacian: {lap_s:.1f}"
                            new_prog = 100
                            new_res = build_status_banner("safe", "[HOÀN TẤT] XÁC THỰC NGƯỜI THẬT (REAL PERSON DETECTED)")
                            if not st.session_state.liveness_crop_rendered and st.session_state.liveness_aligned_b64:
                                render_aligned_face_card(liveness_face_crop, st.session_state.liveness_aligned_b64)
                                st.session_state.liveness_crop_rendered = True

                        new_instr = build_liveness_stepper(
                            st.session_state.liveness_stage,
                            st.session_state.liveness_active_toggle,
                            detail_title,
                            detail_sub
                        )

                        if new_instr != last_t2_instr and (frame_idx == 1 or frame_idx % 4 == 0 or st.session_state.liveness_stage == 3):
                            liveness_instruction.markdown(new_instr, unsafe_allow_html=True)
                            last_t2_instr = new_instr
                        if new_prog != last_t2_prog:
                            liveness_progress.progress(int(new_prog))
                            last_t2_prog = new_prog
                        if new_res != last_t2_res:
                            if new_res:
                                liveness_result_box.markdown(new_res, unsafe_allow_html=True)
                            else:
                                liveness_result_box.empty()
                            last_t2_res = new_res

                    cv2.putText(frame_t2, f"FPS: {fps_display:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    if fps_display < 15.0 and frame_idx > 8:
                        frame_t2 = engine.draw_lag_loading_hud(frame_t2, fps=fps_display)
                    _, jpg_buf = cv2.imencode('.jpg', frame_t2, [cv2.IMWRITE_JPEG_QUALITY, 76])
                    safe_display_image(viewport_t2, jpg_buf.tobytes())

                # =========================================================
                # 3. XỬ LÝ DUY NHẤT TAB 3: THỬ KÍNH THỰC TẾ ẢO (AR GLASSES)
                # =========================================================
                elif is_tab3:
                    frame_t3 = frame.copy()
                    if landmarks_2d:
                        sel_g = st.session_state.selected_glasses_choice
                        current_glasses_bgra = st.session_state.glasses_db.get(sel_g)
                        if current_glasses_bgra is not None:
                            frame_t3 = engine.overlay_glasses_homography(frame_t3, landmarks_2d, current_glasses_bgra)

                        if show_mesh_contours:
                            frame_t3 = engine.draw_face_annotations(frame_t3, landmarks_2d, metrics, draw_box=False, draw_contours=True, draw_vectors=False)

                    cv2.putText(frame_t3, f"FPS: {fps_display:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    if fps_display < 15.0 and frame_idx > 8:
                        frame_t3 = engine.draw_lag_loading_hud(frame_t3, fps=fps_display)
                    _, jpg_buf = cv2.imencode('.jpg', frame_t3, [cv2.IMWRITE_JPEG_QUALITY, 76])
                    safe_display_image(viewport_t3, jpg_buf.tobytes())

                # =========================================================
                # 4. XỬ LÝ DUY NHẤT TAB 4: TRANG ĐIỂM SON MÔI & MÁ HỒNG
                # =========================================================
                elif is_tab4:
                    frame_t4 = frame.copy()
                    if landmarks_2d:
                        lip_name = st.session_state.makeup_lip_color
                        sel_lip_bgr = palette_lip.get(lip_name, (20, 15, 235))
                        blush_name = st.session_state.makeup_blush_color
                        sel_blush_bgr = palette_blush.get(blush_name, (160, 110, 255)) if st.session_state.makeup_enable_blush else None

                        cs_choice = st.session_state.makeup_color_space
                        if cs_choice.startswith("LAB"):
                            cs_m = "LAB"
                        elif cs_choice.startswith("HLS"):
                            cs_m = "HLS"
                        else:
                            cs_m = "HYBRID"

                        frame_t4 = engine.apply_makeup(
                            frame_t4,
                            landmarks_2d,
                            sel_lip_bgr,
                            lip_alpha=st.session_state.makeup_lip_alpha,
                            blush_bgr=sel_blush_bgr,
                            blush_alpha=st.session_state.makeup_blush_alpha,
                            color_space=cs_m
                        )

                        if show_mesh_contours:
                            frame_t4 = engine.draw_face_annotations(frame_t4, landmarks_2d, metrics, draw_box=False, draw_contours=True, draw_vectors=False)

                    cv2.putText(frame_t4, f"FPS: {fps_display:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    if fps_display < 15.0 and frame_idx > 8:
                        frame_t4 = engine.draw_lag_loading_hud(frame_t4, fps=fps_display)
                    _, jpg_buf = cv2.imencode('.jpg', frame_t4, [cv2.IMWRITE_JPEG_QUALITY, 76])
                    safe_display_image(viewport_t4, jpg_buf.tobytes())

                # =========================================================
                # 5. XỬ LÝ DUY NHẤT TAB 5: TƯƠNG TÁC ÁNH MẮT (GAZE)
                # =========================================================
                elif is_tab5:
                    frame_t5 = frame.copy()
                    if landmarks_2d:
                        gaze_data = engine.analyze_gaze(landmarks_2d)
                        il = gaze_data["iris_left"]
                        ir = gaze_data["iris_right"]
                        gx = gaze_data["gaze_x"]
                        gy = gaze_data["gaze_y"]
                        h_dir = gaze_data["h_direction"]

                        r_l = gaze_data.get("ratio_l", gaze_data.get("h_ratio", 0.5))
                        r_r = gaze_data.get("ratio_r", gaze_data.get("h_ratio", 0.5))

                        cv2.circle(frame_t5, il, 4, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame_t5, ir, 4, (0, 255, 255), -1, cv2.LINE_AA)
                        arrow_len = 48.0
                        end_l = (int(il[0] + gx * arrow_len), int(il[1] + gy * arrow_len))
                        end_r = (int(ir[0] + gx * arrow_len), int(ir[1] + gy * arrow_len))
                        cv2.arrowedLine(frame_t5, il, end_l, (0, 242, 254), 2, cv2.LINE_AA, tipLength=0.28)
                        cv2.arrowedLine(frame_t5, ir, end_r, (0, 242, 254), 2, cv2.LINE_AA, tipLength=0.28)

                        if h_dir == "LIẾC TRÁI":
                            st.session_state.gaze_selected_card = "THẺ BÊN TRÁI"
                        elif h_dir == "LIẾC PHẢI":
                            st.session_state.gaze_selected_card = "THẺ BÊN PHẢI"
                        else:
                            st.session_state.gaze_selected_card = "THẺ TRUNG TÂM"

                        is_pose_stable_gaze = (abs(metrics["yaw"]) < 22.0) and (abs(metrics["pitch"]) < 18.0)
                        if metrics["ear"] < engine.ear_thresh and is_pose_stable_gaze:
                            st.session_state.gaze_blink_frames += 1
                        else:
                            if 1 <= st.session_state.gaze_blink_frames <= 12:
                                st.session_state.gaze_last_clicked_card = st.session_state.gaze_selected_card
                                st.session_state.gaze_click_hold = 10
                            st.session_state.gaze_blink_frames = 0

                        if frame_idx == 1 or frame_idx % 6 == 0:
                            gaze_direction_box.markdown(
                                build_metric_card(
                                    "Vector ánh mắt & Mống mắt",
                                    h_dir,
                                    f"Tỷ lệ L: {r_l:.2f} | R: {r_r:.2f}",
                                    pct=float(gaze_data.get("h_ratio", 0.5)) * 100.0,
                                    bar_color="#00f2fe",
                                    badge_text="GAZE 2D"
                                ),
                                unsafe_allow_html=True
                            )
                            gaze_action_box.markdown(
                                build_status_banner("safe", f"[ĐANG TRỎ] {st.session_state.gaze_selected_card}"),
                                unsafe_allow_html=True
                            )
                            click_col = "#10b981" if st.session_state.gaze_click_hold > 0 else "#38bdf8"
                            gaze_click_result.markdown(
                                build_metric_card(
                                    "Thao tác nhấp chuột ảo",
                                    st.session_state.gaze_last_clicked_card,
                                    "Kích hoạt qua xung chớp mắt Edge Triggered Blink",
                                    pct=100 if st.session_state.gaze_click_hold > 0 else 35,
                                    bar_color=click_col,
                                    badge_text="BLINK TRIGGER"
                                ),
                                unsafe_allow_html=True
                            )

                        if st.session_state.gaze_click_hold > 0:
                            st.session_state.gaze_click_hold -= 1

                    cv2.putText(frame_t5, f"FPS: {fps_display:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    if fps_display < 15.0 and frame_idx > 8:
                        frame_t5 = engine.draw_lag_loading_hud(frame_t5, fps=fps_display)
                    _, jpg_buf = cv2.imencode('.jpg', frame_t5, [cv2.IMWRITE_JPEG_QUALITY, 76])
                    safe_display_image(viewport_t5, jpg_buf.tobytes())

                # =========================================================
                # 6. XỬ LÝ DUY NHẤT TAB 6: BIẾN HÌNH & HOÁN ĐỔI MẶT (DELAUNAY)
                # =========================================================
                elif is_tab6:
                    frame_t6 = frame.copy()
                    src_img_m, src_lm_m = get_morph_source_data()
                    if landmarks_2d and src_img_m is not None and src_lm_m is not None:
                        frame_t6, num_tris = engine.swap_and_morph_faces(
                            frame_t6,
                            landmarks_2d,
                            src_img_m,
                            src_lm_m,
                            alpha=st.session_state.morph_alpha,
                            blend_mode=st.session_state.morph_blend_mode,
                            show_delaunay=st.session_state.morph_show_delaunay
                        )
                        if frame_idx == 1 or frame_idx % 8 == 0:
                            morph_metric_box.markdown(
                                build_metric_card(
                                    "Lưới tam giác Delaunay thời gian thực",
                                    f"{num_tris} Tam giác",
                                    f"Alpha = {st.session_state.morph_alpha:.2f} | FPS: {fps_display:.1f}",
                                    pct=st.session_state.morph_alpha * 100.0,
                                    bar_color="#00e676",
                                    badge_text="LIVE SWAP"
                                ),
                                unsafe_allow_html=True
                            )
                    cv2.putText(frame_t6, f"FPS: {fps_display:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    if fps_display < 15.0 and frame_idx > 8:
                        frame_t6 = engine.draw_lag_loading_hud(frame_t6, fps=fps_display)
                    _, jpg_buf = cv2.imencode('.jpg', frame_t6, [cv2.IMWRITE_JPEG_QUALITY, 76])
                    safe_display_image(viewport_t6, jpg_buf.tobytes())

                # =========================================================
                # 7. XỬ LÝ TOÀN CẢNH (CẢ 6 PHÂN HỆ ĐỒNG THỜI)
                # =========================================================
                elif is_all:
                    if metrics["is_yawning"]:
                        trigger_system_alert("yawn")
                    elif metrics["is_drowsy"]:
                        trigger_system_alert("drowsy")
                    elif metrics["is_distracted"]:
                        trigger_system_alert("distracted")

                    _, b1 = cv2.imencode('.jpg', annotated_dms, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    safe_display_image(viewport_t1, b1.tobytes())

                    f2 = frame.copy()
                    if landmarks_2d and show_face_box:
                        f2 = engine.draw_face_annotations(f2, landmarks_2d, metrics, draw_box=True, draw_contours=False, draw_vectors=False, badge_mode="biometric")
                    _, b2 = cv2.imencode('.jpg', f2, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    safe_display_image(viewport_t2, b2.tobytes())

                    f3 = frame.copy()
                    if landmarks_2d:
                        sel_g = st.session_state.selected_glasses_choice
                        cg = st.session_state.glasses_db.get(sel_g)
                        if cg is not None:
                            f3 = engine.overlay_glasses_homography(f3, landmarks_2d, cg)
                    _, b3 = cv2.imencode('.jpg', f3, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    safe_display_image(viewport_t3, b3.tobytes())

                    f4 = frame.copy()
                    if landmarks_2d:
                        lip_name = st.session_state.makeup_lip_color
                        sel_lip_bgr = palette_lip.get(lip_name, (20, 15, 235))
                        blush_name = st.session_state.makeup_blush_color
                        sel_blush_bgr = palette_blush.get(blush_name, (160, 110, 255)) if st.session_state.makeup_enable_blush else None
                        f4 = engine.apply_makeup(f4, landmarks_2d, sel_lip_bgr, lip_alpha=st.session_state.makeup_lip_alpha, blush_bgr=sel_blush_bgr, blush_alpha=st.session_state.makeup_blush_alpha)
                    _, b4 = cv2.imencode('.jpg', f4, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    safe_display_image(viewport_t4, b4.tobytes())

                    f5 = frame.copy()
                    if landmarks_2d:
                        gaze_data = engine.analyze_gaze(landmarks_2d)
                        il = gaze_data["iris_left"]
                        ir = gaze_data["iris_right"]
                        cv2.circle(f5, il, 4, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(f5, ir, 4, (0, 255, 255), -1, cv2.LINE_AA)
                    _, b5 = cv2.imencode('.jpg', f5, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    safe_display_image(viewport_t5, b5.tobytes())

                    f6 = frame.copy()
                    src_img_m, src_lm_m = get_morph_source_data()
                    if landmarks_2d and src_img_m is not None and src_lm_m is not None:
                        f6, _ = engine.swap_and_morph_faces(f6, landmarks_2d, src_img_m, src_lm_m, alpha=1.0)
                    _, b6 = cv2.imencode('.jpg', f6, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    safe_display_image(viewport_t6, b6.tobytes())

                elapsed_loop = time.time() - loop_start_t
                target_interval = 1.0 / max(15, min(40, target_hardware_fps))
                if not is_live_cam:
                    sleep_dur = max(0.012, (1.0 / 30.0) - elapsed_loop)
                    time.sleep(sleep_dur)
                else:
                    sleep_dur = max(0.004, target_interval - elapsed_loop)
                    time.sleep(sleep_dur)

                if frame_idx % 300 == 0:
                    import gc
                    gc.collect()

        finally:
            pass

else:
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

    if is_tab1: viewport_t1.markdown(standby_card, unsafe_allow_html=True)
    elif is_tab2: viewport_t2.markdown(standby_card, unsafe_allow_html=True)
    elif is_tab3: viewport_t3.markdown(standby_card, unsafe_allow_html=True)
    elif is_tab4: viewport_t4.markdown(standby_card, unsafe_allow_html=True)
    elif is_tab5: viewport_t5.markdown(standby_card, unsafe_allow_html=True)
    elif is_tab6: viewport_t6.markdown(standby_card, unsafe_allow_html=True)
    elif is_all:
        viewport_t1.markdown(standby_card, unsafe_allow_html=True)
        viewport_t2.markdown(standby_card, unsafe_allow_html=True)
        viewport_t3.markdown(standby_card, unsafe_allow_html=True)
        viewport_t4.markdown(standby_card, unsafe_allow_html=True)
        viewport_t5.markdown(standby_card, unsafe_allow_html=True)
        viewport_t6.markdown(standby_card, unsafe_allow_html=True)
