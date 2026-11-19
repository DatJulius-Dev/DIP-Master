# ==============================================================================
# HỆ THỐNG GIÁM SÁT TÀI XẾ & TƯƠNG TÁC THỊ GIÁC MÁY TÍNH (DMS CYBER DASHBOARD)
# MODULE: NGROK PUBLIC TUNNEL LAUNCHER FOR STREAMLIT (STRICT HTTPS & ZALO READY)
# ==============================================================================

import os
import sys
import time
import subprocess

def ensure_pyngrok():
    """Kiểm tra và tự động cài đặt thư viện pyngrok nếu chưa có."""
    try:
        from pyngrok import ngrok, conf
        return ngrok, conf
    except ImportError:
        print("[*] Thu vien 'pyngrok' chua duoc cai dat. Dang tu dong cai dat...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok>=7.0.0"])
            from pyngrok import ngrok, conf
            print("[+] Cai dat pyngrok thanh cong!\n")
            return ngrok, conf
        except Exception as e:
            print(f"[-] Loi tu dong cai dat pyngrok: {e}")
            print("[!] Vui long chay lenh: pip install pyngrok thu cong trong Terminal.")
            sys.exit(1)

def safe_apply_token(ngrok_module, conf_module, token):
    """Gán token an toàn, bỏ qua lỗi nếu token đã tồn tại trong ngrok.yml."""
    token = token.strip()
    if not token or len(token) < 15:
        return False
    try:
        conf_module.get_default().auth_token = token
    except Exception:
        pass
    try:
        ngrok_module.set_auth_token(token)
    except Exception as e:
        if "already set" in str(e).lower():
            pass
        else:
            print(f"[!] Thong bao token: {e}")
    return True

def get_or_prompt_token(conf_module, ngrok_module):
    """Lấy hoặc hướng dẫn người dùng nhập Ngrok Authtoken."""
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngrok_token.txt")

    # 1. Kiểm tra biến môi trường
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if token:
        safe_apply_token(ngrok_module, conf_module, token)
        return token

    # 2. Kiểm tra tệp ngrok_token.txt
    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content and not content.startswith("#") and len(content) > 15:
                    safe_apply_token(ngrok_module, conf_module, content)
                    return content
        except Exception:
            pass

    # 3. Kiểm tra config mặc định của pyngrok
    try:
        current_conf = conf_module.get_default()
        if current_conf.auth_token and len(current_conf.auth_token) > 15:
            return current_conf.auth_token
    except Exception:
        pass

    # 4. Kiểm tra file ngrok.yml trên Windows hoặc Linux/macOS
    candidate_paths = []
    appdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        candidate_paths.append(os.path.join(appdata, "ngrok", "ngrok.yml"))
    userprofile = os.environ.get("USERPROFILE", "")
    if userprofile:
        candidate_paths.append(os.path.join(userprofile, ".config", "ngrok", "ngrok.yml"))
        candidate_paths.append(os.path.join(userprofile, ".ngrok2", "ngrok.yml"))
    home = os.path.expanduser("~")
    candidate_paths.append(os.path.join(home, ".config", "ngrok", "ngrok.yml"))
    candidate_paths.append(os.path.join(home, ".ngrok2", "ngrok.yml"))

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if "authtoken:" in line:
                            tok = line.split("authtoken:")[1].strip().strip('"').strip("'")
                            if len(tok) > 15:
                                safe_apply_token(ngrok_module, conf_module, tok)
                                return tok
            except Exception:
                pass

    # 5. Nếu chưa có token, hiển thị hướng dẫn chi tiết
    print("=" * 72)
    print("   HUONG DAN CAU HINH NGROK AUTHTOKEN (CHI CAN NHAP 1 LAN DUY NHAT)  ")
    print("=" * 72)
    print(" Ngrok yeu cau Auth Token mien phi de bao mat va cap quyen mo cong:")
    print("   1. Dang ky tai khoan mien phi tai: https://dashboard.ngrok.com/signup")
    print("   2. Copy Auth Token cua ban tai:    https://dashboard.ngrok.com/get-started/your-authtoken")
    print("   3. Dan Token vao ben duoi va nhan Enter:")
    print("=" * 72)

    user_token = input(">> Nhap Ngrok Authtoken cua ban: ").strip()
    if not user_token:
        print("[-] Khong co token. Ngrok se khong the ket noi. Dang thoat...")
        sys.exit(1)

    safe_apply_token(ngrok_module, conf_module, user_token)
    try:
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(user_token)
        print(f"[+] Da luu token vao '{token_file}' de su dung cho cac lan sau!\n")
    except Exception:
        pass
    return user_token

def main():
    print("\n" + "=" * 72)
    print("   DMS CYBER DASHBOARD - KHOI DONG STREAMLIT KET HOP NGROK TUNNEL   ")
    print("=" * 72 + "\n")

    ngrok, conf = ensure_pyngrok()
    get_or_prompt_token(conf, ngrok)

    port = 8501
    print(f"[*] Dang khoi tao Ngrok HTTPS Secure Tunnel tren cong {port}...")
    public_url = None
    try:
        try:
            for t in ngrok.get_tunnels():
                ngrok.disconnect(t.public_url)
        except Exception:
            pass

        # Khởi tạo Ngrok Tunnel tương thích hoàn toàn ngrok v2 và ngrok v3
        # LƯU Ý: Tuyệt đối KHÔNG truyền bind_tls=True vì trên ngrok agent v3 sẽ gây lỗi:
        # 'yaml: unmarshal errors: field schemes not found in type config.Endpoint'
        tunnel = None
        try:
            tunnel = ngrok.connect(port)
        except Exception:
            try:
                tunnel = ngrok.connect(addr=port, proto="http")
            except Exception:
                tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://")

        print("\n" + "=" * 72)
        print("    CHUC MUNG! UNG DUNG DA DUOC PUBLIC THANH CONG RA INTERNET!   ")
        print("=" * 72)
        print("  * LINK HTTPS BAO MAT DE GUI QUA ZALO / MESSENGER CHO GIANG VIEN:")
        print(f"  -> {public_url}")
        print("=" * 72)
        print("  [Huong Dan Su Dung Camera Tu Xa Tren Zalo & Trinh Duyet]:")
        print("   1. Khi mo link lan dau, bam nut 'Visit Site' de vao Dashboard.")
        print("   2. Chon 'Camera thiet bi tu xa qua Trinh duyet' tren thanh ben trai.")
        print("   3. He thong ho tro 2 che do linh hoat:")
        print("      - Che do A (WebRTC Live): Bam 'Cho phep' (Allow) tren Chrome/Safari.")
        print("      - Che do B (Tuong thich 100% Zalo & HTTP): Chup anh / Quay video")
        print("        truc tiep bang Camera goc cua dien thoai ngay trong Zalo!")
        print("   4. Nhan Ctrl + C tai cua so nay de dong ung dung va huy tunnel.")
        print("=" * 72 + "\n")

    except Exception as e:
        print(f"\n[-] Loi khi mo Ngrok Tunnel: {e}")
        print("[!] Hay kiem tra ket noi Internet hoac kiem tra lai Ngrok Authtoken cua ban.\n")
        sys.exit(1)

    print("[*] Dang khoi chay Streamlit Web Server...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app_streamlit.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]

    streamlit_process = subprocess.Popen(streamlit_cmd)

    try:
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n[*] Nhan tin hieu dung (Ctrl+C). Dang dong Ngrok va Streamlit...")
    finally:
        try:
            if public_url:
                ngrok.disconnect(public_url)
            ngrok.kill()
        except Exception:
            pass
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
        print("[+] He thong da tat an toan. Tam biet!\n")

if __name__ == "__main__":
    main()
