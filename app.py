import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- CẤU HÌNH MEDIA PIPE ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- HÀM TÍNH TOÁN GÓC ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(layout="wide", page_title="AI Fitness Trainer")

# Giấu Menu mặc định cho đẹp
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- SIDEBAR (THANH BÊN) ---
st.sidebar.title("⚙️ Cấu hình")
st.sidebar.info("Phiên bản Cloud (WebRTC)")
target_reps = st.sidebar.number_input("Mục tiêu (Cái)", min_value=1, value=10)
confidence = st.sidebar.slider("Độ nhạy AI", 0.0, 1.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.write("### 💡 Hướng dẫn:")
st.sidebar.write("1. Cho phép trình duyệt dùng Camera.")
st.sidebar.write("2. Chờ kết nối (có thể mất 10-20s).")
st.sidebar.write("3. Đứng xa để thấy nửa người trên.")

# --- CLASS XỬ LÝ VIDEO ---
class PoseDetector:
    def __init__(self):
        # Khởi tạo MediaPipe
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Xử lý ảnh
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Logic đếm
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Logic đếm Reps
            if angle > 160:
                self.stage = "down"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.counter += 1
                
        except Exception as e:
            pass

        # VẼ GIAO DIỆN LÊN VIDEO (Thay thế cho Chart bị lag trên Cloud)
        # 1. Vẽ hộp thông tin
        cv2.rectangle(image, (0,0), (250,80), (245,117,16), -1)
        
        # 2. Hiện số Reps
        cv2.putText(image, 'REPS', (15,25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), (10,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        # 3. Hiện trạng thái
        cv2.putText(image, 'STAGE', (90,25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.stage), (85,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

        # 4. Vẽ xương
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- PHẦN CHÍNH ---
st.title("💪 AI Personal Trainer - Bicep Curls")
st.write("Ứng dụng sử dụng Pose Estimation chạy trên Cloud.")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    # Cấu hình WebRTC với danh sách STUN Server mở rộng
    webrtc_streamer(
        key="visionfit-pose", 
        video_processor_factory=PoseDetector,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        }
    )

with col2:
    st.markdown("### 📊 Trạng thái")
    st.info("Đang chờ Camera...")
    st.write("Vì chạy trên Cloud nên sẽ có độ trễ nhất định so với chạy Local.")
    st.success(f"Mục tiêu: {target_reps} cái")