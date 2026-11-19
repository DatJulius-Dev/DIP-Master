"""
AUDIO MANAGER MODULE
Quan ly phat am thanh canh bao da luong bat dong bo (Asynchronous Threading).
Tich hop bo dinh thoi giam xoc Cooldown Timer thong minh cho giong noi tieng Viet (3 Case chuan + Case Cabin 2 nguoi):
- Case 1: Canh bao ngu gat (Drowsy): "Canh bao! Tai xe dang ngu gat! Hay tap trung lai xe!"
- Case 2: Canh bao ngap / bieu hien ngu gat (Yawn): "Canh bao! Tai xe dang co bieu hien ngu gat! Hay tap trung lai xe!"
- Case 3: Canh bao mat tap trung (Distracted): "Canh bao! Tai xe dang mat tap trung! Hay tap trung lai xe!"
- Case 4: Cabin 2 nguoi (Dual Drowsy): "Canh bao! Phat hien tai xe dang ngu gat! Ban dong hanh hay chu y ho tro xu ly!"
- Xac thuc thanh cong (Success): Am chime success
Tuan thu chat che tieu chuan ky thuat: khong su dung ky tu icon hay emoji.
"""

import os
import sys
import time
import base64
import threading
import subprocess


class AudioManager:
    def __init__(self, sounds_dir=None, cooldown_seconds=5.0, global_cooldown=4.5):
        if sounds_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            sounds_dir = os.path.join(base_dir, "sounds")
        self.sounds_dir = sounds_dir
        self.cooldown_seconds = float(cooldown_seconds)
        self.global_cooldown = float(global_cooldown)
        self.is_enabled = True
        self.mode = "combined"  # "combined" (Beep + Voice), "voice_only", "beep_only"
        self.last_play_time = {}
        self.last_global_play_time = 0.0
        self._lock = threading.Lock()
        self._play_lock = threading.Lock()
        self._b64_cache = {}

    def set_enabled(self, enabled):
        self.is_enabled = bool(enabled)

    def set_mode(self, mode):
        """
        Thiet lap che do phat am thanh:
        - "combined": Am beep + Giong noi tieng Viet (Mac dinh chuan)
        - "voice_only": Chi phat giong noi tieng Viet
        - "beep_only": Chi phat am thanh beep
        """
        valid_modes = ["combined", "voice_only", "beep_only"]
        if mode in valid_modes:
            self.mode = mode

    def trigger_alert(self, alert_type, force=False, play_os=True):
        """
        Phat am thanh theo loai canh bao va che do hien tai:
        - drowsy: Case 1 (Ngu gat)
        - yawn: Case 2 (Ngap ngu / Bieu hien ngu gat)
        - distracted: Case 3 (Mat tap trung)
        - dual_drowsy: Case Cabin 2 nguoi
        - success: Chime KYC
        force: Bo qua Cooldown khi nguoi dung bam nut Test coi
        play_os: Co phat qua loa phan cung OS (winsound) hay khong (chong Double am voi trinh duyet)
        """
        if not self.is_enabled:
            return None

        now = time.time()
        with self._lock:
            last_time = self.last_play_time.get(alert_type, 0.0)

            # Dinh thoi Cooldown rieng cho tung loai de dam bao giong noi doc tron cau
            if alert_type == "success":
                type_cd = 0.45
            elif alert_type == "drowsy":
                type_cd = 5.0 if self.mode != "beep_only" else 1.2
            elif alert_type == "dual_drowsy":
                type_cd = 5.5 if self.mode != "beep_only" else 1.2
            elif alert_type == "yawn":
                type_cd = 5.0 if self.mode != "beep_only" else 1.0
            elif alert_type == "distracted":
                type_cd = 4.8 if self.mode != "beep_only" else 1.0
            else:
                type_cd = self.cooldown_seconds

            if not force and (now - last_time < type_cd):
                return None

            # Chong chong am giua cac canh bao khac nhau (Anti-Voice Overlapping)
            # Dam bao cau thoai truoc doc xong het (~4.5s) moi cho phep cau khac chen vao
            min_global_cd = 1.0 if self.mode == "beep_only" else 4.5
            effective_global_cd = max(self.global_cooldown, min_global_cd)
            if not force and alert_type != "success" and (now - self.last_global_play_time < effective_global_cd):
                return None

            self.last_play_time[alert_type] = now
            self.last_global_play_time = now

        # Tra cuu file am thanh theo che do
        if self.mode == "voice_only":
            sound_files = {
                "drowsy": "voice_drowsy.wav",
                "dual_drowsy": "voice_dual_drowsy.wav",
                "yawn": "voice_yawn.wav",
                "distracted": "voice_distracted.wav",
                "success": "chime_success.wav"
            }
        elif self.mode == "beep_only":
            sound_files = {
                "drowsy": "beep_drowsy.wav",
                "dual_drowsy": "beep_drowsy.wav",
                "yawn": "beep_yawn.wav",
                "distracted": "beep_distracted.wav",
                "success": "chime_success.wav"
            }
        else:  # "combined" mac dinh
            sound_files = {
                "drowsy": "alert_drowsy.wav",
                "dual_drowsy": "alert_dual_drowsy.wav",
                "yawn": "alert_yawn.wav",
                "distracted": "alert_distracted.wav",
                "success": "chime_success.wav"
            }

        filename = sound_files.get(alert_type)
        if not filename:
            return None

        filepath = os.path.join(self.sounds_dir, filename)
        if not os.path.isfile(filepath):
            return None

        if play_os:
            worker = threading.Thread(target=self._play_file_async, args=(filepath,), daemon=True)
            worker.start()
        return (alert_type, filepath)

    def _play_file_async(self, filepath):
        if not self._play_lock.acquire(blocking=False):
            return
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            elif sys.platform.startswith("darwin"):
                subprocess.run(["afplay", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                if subprocess.call(["which", "paplay"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    subprocess.run(["paplay", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif subprocess.call(["which", "aplay"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    subprocess.run(["aplay", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        finally:
            time.sleep(0.08)
            self._play_lock.release()

    def get_sound_base64(self, filepath):
        """
        Doc file am thanh va chuyen doi sang chuoi Base64 co bo nho dem RAM (O(1) Cache)
        de phat tren trinh duyet ma khong gay tre khung hinh video.
        """
        if not filepath:
            return ""
        cached = self._b64_cache.get(filepath)
        if cached is not None:
            return cached
        try:
            with open(filepath, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            self._b64_cache[filepath] = encoded
            return encoded
        except Exception:
            return ""
