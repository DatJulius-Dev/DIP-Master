# HE THONG GIAM SAT TAI XE THONG MINH & DA UNG DUNG MOC HINH HOC KHUON MAT
## DRIVER MONITORING SYSTEM & FACE LANDMARK MULTI-APPLICATIONS
- Hoc phan: Xu ly anh so (Digital Image Processing) - Dai hoc Ton Duc Thang
- Giang vien huong dan: TS. Pham Van Huy

---

## 1. CAU TRUC 6 PHAN HE CHUYEN SAU TREN STREAMLIT (app_streamlit.py)
### [TAB 1] Canh bao buon ngu cho tai xe (DMS - Driver Monitoring)
- Ban chat DIP: Hinh hoc giai tich 2D/3D, tinh ty le khia canh mat EAR (Soukupova & Cech, 2016), mieng MAR, bo loc lam muot thich nghi EMA (Exponential Moving Average) chong rung nhiễu va thuat toan uoc luong tu the dau cv2.solvePnP (Perspective-n-Point) voi mo hinh chuan 3D Canonical Face Model.
- Tinh nang: Tich hop chuan do luong vi giac ngu PERCLOS (NHTSA) kem cong chan Sanity Gate khi quay dau, phan ra 3 goc Euler (Pitch, Yaw, Roll) bang bien doi Rodrigues da chuan hoa chong nhay dau +-180 do, co che hieu chuan moc 0 chuan (Zero-Reference) va coi bao dong da luong bat dong bo.

### [TAB 2] Bai toan Diem danh & Chong gia mao (Face Attendance & Liveness)
- Ban chat DIP:
  + Face Alignment (Can chinh khuon mat): Dung toa do tam 2 mat de tinh goc xoay, ap dung ma tran bien doi Affine (cv2.getRotationMatrix2D + cv2.warpAffine) xoay thang khuon mat ve chuan truoc khi luu du lieu hoac trich xuat dac trung.
  + Active Liveness: Kiem tra phan xa chop mat va quay dau theo may trang thai huu han (FSM) ben vung trong session_state de chong viec dung anh chup man hinh dien thoai hoac video phat lai.
- Demo: Bat che do kiem tra Liveness tren Tab 2 -> Thuc hien thu thach truc tiep tren cam -> Diem danh hoan tat -> Xuat ngay anh mat cat vuong chuan da can chinh thang tap (256x256).

### [TAB 3] Thu kinh thuc te ao (Virtual Glasses Try-On)
- Ban chat DIP: Bien doi phoi canh Homography 3D (3x3) voi hieu ung co ngan phoi canh (Perspective Foreshortening) va neo song mui (landmark 168), ket hop tron kenh trong suot Alpha Blending tren vung ROI mat.
- Demo: Chon 4 mau kinh (Phi cong Aviator ma vang, Mat tron Retro, Kinh ram the thao Cyber, Kinh can gong vuong), tu dong xoay va bien dang phoi canh 3D bam sat theo khuon mat thoi gian thuc.

### [TAB 4] Trang diem ao (Virtual Makeup - Son moi & Ma hong)
- Ban chat DIP:
  + Mat na da giac & Lam mem bien tren vung ROI: Dung cv2.fillPoly tao mat na bo moi ngoai, tru di khoang mieng trong, dung cv2.GaussianBlur (Feathering) lam min bien son tu nhien.
  + Bao toan van moi voi khong gian mau CIELAB (L*a*b*) & HLS: Giu nguyen kenh do sang cam nhan L* va Lightness L de bao toan van moi va anh phan quang tu nhien, chi noi suy kenh sac do.
  + Ma hong ao: Dung 2 diem go ma (117, 346) ve hinh elip mo Gauss tren vung ROI hoa tron phan ma hong.
- Demo: Chon 5 bo mau son cao cap, tuy chon thuat toan khong gian mau (HYBRID / CIELAB / HLS), thanh truot dieu chinh do dam nhat Opacity va ma hong tu nhien.

### [TAB 5] Dieu khien tuong tac qua anh mat & khuon mat (Gaze & Facial Interaction)
- Ban chat DIP: Khai thac 10 diem mong mat (Iris Landmarks tu 468 den 477). Chieu vi tri tam con nguoi len he truc truc chuan noi 2 khoe mat de xac dinh Vector huong nhin (Gaze Vector) ngang va doc bat bien voi goc nghieng dau.
- Demo: Hien thi mui ten Vector huong nhin truc tiep tren mat, la ban radar 2D thoi gian thuc, liec mat sang trai/phai de tro the tuong tac va nhay mat chu dich (Blink Edge Trigger) de kich hoat lua chon (Virtual Click).

---

## 2. HUONG DAN CAI DAT VA KHOI CHAY
### Buoc 1: Cai dat thu vien
pip install -r requirements.txt

### Buoc 2: Khoi chay Web App Streamlit (5 Tabs day du)
streamlit run app_streamlit.py
(Hoac chay: python app_streamlit.py, hoac nhap dup vao file run_web.bat)

### Buoc 3: Khoi chay Desktop HUD ADAS OpenCV (Camera truc tiep)
python app_opencv.py
(Hoac nhap dup vao file run_desktop.bat)

### Buoc 4: Khoi chay Web App kem Ngrok Tunnel (Cho moi nguoi test tu xa)
python run_with_ngrok.py
(Hoac nhap dup vao file run_web_ngrok.bat)
- Tinh nang: Tu dong tao tunnel HTTPS an toan (https://xxxx.ngrok-free.app) de ban be hoac giang vien co the truy cap va trai nghiem he thong tu bat ky thiet bi nao qua mang Internet.


### [TAB 6] Bien hinh & Hoan doi khuon mat (Face Morphing & Face Swapping - Delaunay Triangulation)
- Ban chat DIP: Chia luoi tam giac Delaunay (cv2.Subdiv2D) tren 79 diem moc giai phau khuon mat, bien doi Affine tung manh tam giac (Piece-wise Affine Triangle Warping voi cv2.getAffineTransform va cv2.warpAffine), chuyen giao thong ke mau da Reinhard tren khong gian mau CIELAB (cv2.meanStdDev) va hoa tron bien muot ma qua mat na bao loi cv2.convexHull hoac tron mien dao ham Poisson cv2.seamlessClone.
- Tinh nang: Tuy chinh trong so bien hinh Alpha lien tuc tu 0.0 (Mat goc) -> 0.5 (Lai 50/50 hai khuon mat) -> 1.0 (Hoan doi 100% khuon mat), bat/tat luoi tam giac Delaunay truc quan tren ca Live Camera, Video, Anh tinh va WebRTC.
