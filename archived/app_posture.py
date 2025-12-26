import cv2
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Cấu hình trang
st.set_page_config(page_title="AI Posture Assistant", layout="centered")

st.title("🧘 AI Posture Corrector")
st.markdown("---")

# 2. Load Model (Cache để không load lại)
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")
    st.stop()

# 3. Hàm tính toán góc
def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def process_frame(frame, threshold):
    # Resize để tăng tốc độ nếu ảnh quá lớn
    # frame = cv2.resize(frame, (640, 480))
    
    results = model(frame, verbose=False, conf=0.5)
    annotated_frame = frame.copy()
    status = "Unknown"
    color = (200, 200, 200)

    if results[0].keypoints.has_visible:
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        
        # Chọn bên nào rõ hơn (Left vs Right)
        if keypoints[3][2] > keypoints[4][2]: # So sánh độ tin cậy của Tai
            ear, shoulder, hip = keypoints[3][:2], keypoints[5][:2], keypoints[11][:2]
        else:
            ear, shoulder, hip = keypoints[4][:2], keypoints[6][:2], keypoints[12][:2]

        angle = calculate_angle(ear, shoulder, hip)
        
        if angle < threshold:
            color = (0, 0, 255) # Red
            status = "BAD POSTURE"
        else:
            color = (0, 255, 0) # Green
            status = "GOOD"

        # Vẽ
        cv2.line(annotated_frame, (int(ear[0]), int(ear[1])), (int(shoulder[0]), int(shoulder[1])), (255, 255, 255), 3)
        cv2.line(annotated_frame, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (255, 255, 255), 3)
        cv2.circle(annotated_frame, (int(shoulder[0]), int(shoulder[1])), 10, color, -1)
        
        cv2.putText(annotated_frame, f"Angle: {int(angle)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated_frame, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return annotated_frame, status

# 4. Sidebar Cấu hình
st.sidebar.header("⚙️ Cài đặt")
mode = st.sidebar.radio("Chọn chế độ đầu vào:", ["📷 Sử dụng Webcam", "📂 Upload Video có sẵn"])
threshold = st.sidebar.slider("Ngưỡng cảnh báo (Góc lưng)", 50, 170, 140)
st_status_box = st.sidebar.empty()

# 5. Logic xử lý chính
st_frame_display = st.empty()

# --- CHẾ ĐỘ WEBCAM ---
if mode == "📷 Sử dụng Webcam":
    st.info("Nhấn nút bên dưới để kết nối Webcam.")
    start_cam = st.button("🔴 Bắt đầu Webcam", use_container_width=True)
    
    if start_cam:
        # Thử mở Webcam
        cap = cv2.VideoCapture(0)
        
        # KIỂM TRA NGAY: Nếu không mở được -> Báo lỗi & Chuyển sang Upload
        if not cap.isOpened():
            st.error("❌ LỖI: Thiết bị này không có Webcam hoặc không cho phép truy cập!")
            st.warning("⚠️ Đang chuyển sang chế độ Upload Video dự phòng...")
            
            # --- FALLBACK: Hiện chỗ upload ngay tại đây ---
            fallback_file = st.file_uploader("📂 Hãy chọn video mẫu để thay thế:", type=['mp4', 'avi', 'mov'])
            if fallback_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(fallback_file.read())
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.stop() # Dừng lại đợi upload
        
        # Nút dừng (dùng key để tránh trùng lặp)
        stop_btn = st.button("Dừng lại", key="stop_webcam")
        
        # Vòng lặp xử lý (Dù là Webcam thật hay Video fallback đều chạy ở đây)
        while cap.isOpened():
            if stop_btn: break
            
            success, frame = cap.read()
            if not success:
                st.warning("Mất tín hiệu video.")
                break
                
            # Xử lý AI
            processed_frame, status_text = process_frame(frame, threshold)
            
            # Hiển thị
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_frame_display.image(processed_frame, channels="RGB", use_container_width=True)
            
            # Cập nhật trạng thái sidebar
            if status_text == "BAD POSTURE":
                st_status_box.error(status_text)
            else:
                st_status_box.success(status_text)
                
        cap.release()

# --- CHẾ ĐỘ UPLOAD VIDEO (Chủ động) ---
elif mode == "📂 Upload Video có sẵn":
    uploaded_file = st.file_uploader("Kéo thả video vào đây", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        if st.button("▶️ Chạy Video", use_container_width=True):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            stop_btn = st.button("Dừng video")
            
            while cap.isOpened():
                if stop_btn: break
                success, frame = cap.read()
                if not success: break
                
                processed_frame, status_text = process_frame(frame, threshold)
                
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st_frame_display.image(processed_frame, channels="RGB", use_container_width=True)
                
                if status_text == "BAD POSTURE":
                    st_status_box.error(status_text)
                else:
                    st_status_box.success(status_text)

            cap.release()