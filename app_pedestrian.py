import cv2
import tempfile
import numpy as np
import streamlit as st
from collections import deque
from ultralytics import YOLO

# 1. Cấu hình trang
st.set_page_config(page_title="AI Pedestrian Analysis", layout="wide")
st.title("🚶 AI Pedestrian Counting")
st.markdown("Hệ thống đếm người đi bộ trong video (Đã lọc nhiễu ID).")

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")
    st.stop()

# 3. Sidebar Cấu hình
st.sidebar.header("⚙️ Cấu hình")
conf_threshold = st.sidebar.slider("Độ nhạy (Confidence)", 0.3, 1.0, 0.5)

# Thanh trượt quan trọng để lọc nhiễu
min_hits = st.sidebar.slider(
    "Số frame tối thiểu để đếm (Anti-Flicker)", 
    min_value=5, max_value=60, value=20, 
    help="Một người phải xuất hiện liên tục trong N frame thì mới được tính. Giúp loại bỏ rác hoặc nhận diện chập chờn."
)

# 4. Biến toàn cục
track_history = {} 
total_unique_ids = set() 
id_life_count = {} # Biến đếm tuổi thọ ID

metric_placeholder = st.empty()
st_frame = st.empty()

# 5. Giao diện Upload
uploaded_file = st.file_uploader("📂 Chọn video CCTV / Người đi bộ (mp4, avi)", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    # Lưu file tạm
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("▶️ Bắt đầu phân tích"):
        cap = cv2.VideoCapture(tfile.name)
        
        if cap.isOpened():
            stop_btn = st.button("Dừng lại")
            
            # Progress bar
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            frame_count = 0
            
            while cap.isOpened() and not stop_btn:
                success, frame = cap.read()
                if not success: break
                
                frame_count += 1
                if frame_count % 5 == 0: # Cập nhật thanh tiến trình mỗi 5 frame để đỡ lag
                    progress_bar.progress(frame_count / total_frames)

                overlay = frame.copy()
                
                # Tracking
                results = model.track(frame, classes=[0], conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        
                        # --- LOGIC CHỐNG NHIỄU (ANTI-FLICKER) ---
                        # 1. Tăng tuổi thọ ID
                        id_life_count[track_id] = id_life_count.get(track_id, 0) + 1
                        
                        color = (0, 255, 0) # Xanh (Chưa đếm)
                        status_text = "Tracking..."
                        
                        # 2. Chỉ ĐẾM khi ID tồn tại đủ lâu ( > min_hits)
                        if id_life_count[track_id] > min_hits:
                            total_unique_ids.add(track_id)
                            color = (0, 0, 255) # Đỏ (Đã đếm)
                            status_text = f"ID:{track_id}"
                            
                            # Vẽ đường đi (Heatmap)
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            if track_id not in track_history:
                                track_history[track_id] = deque(maxlen=40)
                            track_history[track_id].append((cx, cy))
                            
                            points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(overlay, [points], isClosed=False, color=(255, 255, 0), thickness=3)

                        # Vẽ Box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        # Hiển thị số frame đã tồn tại để debug dễ hơn
                        cv2.putText(frame, f"{status_text} ({id_life_count[track_id]})", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Gộp lớp phủ
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                # Hiển thị kết quả
                metric_placeholder.metric("👥 Tổng số người (Đã lọc nhiễu)", len(total_unique_ids))
                
                # Resize để hiển thị mượt hơn trên web
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", width=1000)

            cap.release()
            st.success("Đã phân tích xong video!")