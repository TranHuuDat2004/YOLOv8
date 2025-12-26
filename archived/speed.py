import cv2
import tempfile
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO

# 1. Cấu hình trang
st.set_page_config(page_title="AI Speed Estimation", layout="wide")
st.title("🚗 AI Speed Estimation (Đo tốc độ)")
st.markdown("""
**Nguyên lý:**
1. Tracking vị trí vật thể theo thời gian.
2. Tính khoảng cách di chuyển (pixel).
3. Chia cho thời gian (dựa vào FPS) để ra vận tốc.
""")

# 2. Load Model
@st.cache_resource
def load_model():
    # Dùng model lớn hơn chút (medium) để detect xe tốt hơn
    return YOLO('yolov8m.pt') 

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")
    st.stop()

# 3. Sidebar
st.sidebar.header("⚙️ Cấu hình đo đạc")
conf_threshold = st.sidebar.slider("Độ nhạy (Confidence)", 0.3, 1.0, 0.5)

# QUAN TRỌNG: Hệ số quy đổi Pixel -> Mét
# Bạn cần ước lượng: Chiều ngang đường thực tế là bao nhiêu mét? 
# Và trên video nó chiếm bao nhiêu pixel?
# Ví dụ: 1 chiếc xe dài 4.5m, trên video xe dài 100 pixel -> 1 mét = 22 pixel.
pixels_per_meter = st.sidebar.number_input("Số Pixel ứng với 1 Mét (Calibration)", min_value=1.0, value=20.0, step=1.0)

source_radio = st.sidebar.radio("Nguồn video:", ["📂 Upload Video", "📷 Webcam"])

# 4. Biến lưu trữ tốc độ
# Cấu trúc: {track_id: [last_x, last_y, last_time, current_speed_kmh]}
speed_tracker = {}

st_frame = st.empty()
cap = None

if source_radio == "📂 Upload Video":
    uploaded_file = st.file_uploader("Chọn video giao thông", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if st.sidebar.button("▶️ Bắt đầu đo tốc độ"):
            cap = cv2.VideoCapture(tfile.name)
elif source_radio == "📷 Webcam":
    if st.sidebar.button("🔴 Bật Camera"):
        cap = cv2.VideoCapture(0)

# 5. Xử lý chính
if cap is not None and cap.isOpened():
    stop_btn = st.sidebar.button("Dừng lại")
    
    # Lấy FPS video để tính thời gian
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Fallback nếu không đọc được FPS
    
    while cap.isOpened() and not stop_btn:
        success, frame = cap.read()
        if not success: break
        
        # Resize nhẹ để xử lý nhanh hơn nếu video 4K
        frame = cv2.resize(frame, (1280, 720))
        
        # Tracking (Xe hơi: class 2, Xe máy: class 3, Xe tải: class 7, Bus: 5)
        # Hoặc để trống classes=... để detect tất cả
        results = model.track(frame, classes=[2, 3, 5, 7], conf=conf_threshold, persist=True, verbose=False, tracker="bytetrack.yaml")
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                # Lấy điểm đáy (chân xe) để tính khoảng cách chính xác hơn tâm
                bottom_center = (cx, int(y2)) 
                
                current_time = time.time()
                
                speed_kmh = 0
                
                # Logic tính toán
                if track_id in speed_tracker:
                    prev_x, prev_y, prev_time, prev_speed = speed_tracker[track_id]
                    
                    # 1. Tính khoảng cách pixel (Euclidean distance)
                    pixel_dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    
                    # 2. Chỉ tính nếu xe di chuyển đủ nhiều (tránh rung lắc nhiễu)
                    if pixel_dist > 2: 
                        # 3. Quy đổi ra mét
                        real_dist_meters = pixel_dist / pixels_per_meter
                        
                        # 4. Tính thời gian trôi qua (giây)
                        # Cách 1: Dùng thời gian thực hệ thống (tốt cho webcam)
                        time_diff = current_time - prev_time
                        
                        # Cách 2: Dùng FPS video (tốt cho video upload) -> Chính xác hơn
                        # time_diff = 1 / fps 
                        
                        if time_diff > 0:
                            speed_ms = real_dist_meters / time_diff # Mét / giây
                            speed_kmh_raw = speed_ms * 3.6 # Đổi ra km/h
                            
                            # 5. Làm mượt số liệu (Moving Average) để số không nhảy loạn xạ
                            speed_kmh = 0.8 * prev_speed + 0.2 * speed_kmh_raw
                    else:
                        speed_kmh = prev_speed
                
                # Cập nhật vị trí mới
                speed_tracker[track_id] = [cx, cy, current_time, speed_kmh]
                
                # Vẽ lên hình
                label = f"ID:{track_id} {int(speed_kmh)} km/h"
                
                # Đổi màu theo tốc độ (Nhanh = Đỏ, Chậm = Xanh)
                color = (0, 255, 0)
                if speed_kmh > 40: color = (0, 165, 255) # Cam
                if speed_kmh > 70: color = (0, 0, 255)   # Đỏ
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Hiển thị
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", width=1200)

    cap.release()