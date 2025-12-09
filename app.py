import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

# --- CẤU HÌNH MEDIA PIPE ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- HÀM TÍNH TOÁN GÓC (LOGIC CỐT LÕI) ---
def calculate_angle(a, b, c):
    """
    Tính góc giữa 3 điểm a, b, c.
    a: vai, b: khuỷu tay, c: cổ tay
    """
    a = np.array(a) # Đầu mút 1
    b = np.array(b) # Đỉnh góc
    c = np.array(c) # Đầu mút 2
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# --- GIAO DIỆN STREAMLIT (UI) ---
st.set_page_config(layout="wide", page_title="AI Fitness Trainer")

# Tiêu đề và Sidebar
st.sidebar.image("https://mediapipe.dev/images/mobile/pose_tracking_example.gif", use_container_width=True)
st.sidebar.title("⚙️ Cài đặt")
target_reps = st.sidebar.number_input("Mục tiêu (Cái)", min_value=1, value=10)
confidence = st.sidebar.slider("Độ nhạy AI", 0.0, 1.0, 0.5)

st.title("💪 AI Personal Trainer - Bicep Curls")
st.write("Ứng dụng sử dụng Pose Estimation để đếm số lần tập luyện chuẩn xác.")

# Chia cột: Bên trái Video - Bên phải Thông số
col1, col2 = st.columns([0.7, 0.3])

with col2:
    st.markdown("### 📊 Thống kê thời gian thực")
    count_placeholder = st.empty() # Chỗ để hiện số đếm
    stage_placeholder = st.empty() # Chỗ để hiện trạng thái (Lên/Xuống)
    progress_bar = st.progress(0)  # Thanh tiến trình góc độ
    status_text = st.empty()       # Lời nhắc nhở

# Nút Start/Stop
run = st.checkbox('Bắt đầu Camera', value=True)
FRAME_WINDOW = col1.image([]) # Khung hình video

# --- XỬ LÝ CHÍNH ---
cap = cv2.VideoCapture(0) # Mở Webcam
counter = 0 
stage = None

# Khởi tạo Pose detection
with mp_pose.Pose(min_detection_confidence=confidence, min_tracking_confidence=confidence) as pose:
    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Không tìm thấy Camera!")
            break
        
        # 1. Chuẩn bị ảnh
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # 2. Đưa ảnh vào AI (MediaPipe)
        results = pose.process(image)
    
        # 3. Vẽ lại ảnh để hiển thị
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 4. Xử lý Logic đếm (Quan trọng nhất)
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Lấy tọa độ 3 điểm bên tay TRÁI (Vai - Khuỷu - Cổ tay)
            # Nếu muốn tay phải thì đổi LEFT thành RIGHT
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Tính góc
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Hiển thị góc lên màn hình video
            cv2.putText(image, str(int(angle)), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Logic đếm Reps
            if angle > 160:
                stage = "down" # Tay duỗi thẳng
            if angle < 30 and stage =='down':
                stage = "up" # Tay gập lại
                counter += 1
                
            # Cập nhật thanh Progress Bar theo góc (map góc 30-160 về 0-100)
            try:
                prog_val = np.interp(angle, [30, 160], [100, 0]) / 100
                progress_bar.progress(float(prog_val))
            except:
                pass

        except:
            pass
        
        # 5. Cập nhật giao diện Streamlit (Update UI)
        # Hiển thị số đếm to đùng
        count_placeholder.metric("Số lần tập (Reps)", counter, f"Mục tiêu: {target_reps}")
        
        # Hiển thị trạng thái
        if stage == 'up':
            stage_placeholder.info(f"Trạng thái: GẬP TAY (UP)")
        else:
            stage_placeholder.warning(f"Trạng thái: DUỖI TAY (DOWN)")

        # Cảnh báo nếu hoàn thành
        if counter >= target_reps:
            status_text.success("🎉 Đã hoàn thành mục tiêu!")
            
        # Vẽ bộ xương lên hình
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
        
        # Chuyển lại màu RGB để hiện lên Web
        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cap.release()