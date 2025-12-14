import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# --- CẤU HÌNH MEDIA PIPE HANDS ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- HÀM ĐẾM SỐ NGÓN TAY (LOGIC CỐT LÕI) ---
def count_fingers(image, hand_landmarks, hand_label):
    """
    Đếm số ngón tay đang mở dựa trên tọa độ các khớp.
    """
    count = 0
    # Danh sách các đầu ngón tay (Tip)
    # 4: Ngón cái, 8: Trỏ, 12: Giữa, 16: Áp út, 20: Út
    tip_ids = [4, 8, 12, 16, 20]
    
    # Lấy kích thước ảnh để vẽ text
    h, w, c = image.shape

    # --- 1. Xử lý 4 ngón dài (Trỏ, Giữa, Áp út, Út) ---
    # Logic: Nếu đầu ngón tay (Tip) nằm CAO HƠN khớp nối (PIP - khớp thứ 2 từ dưới lên)
    # Lưu ý: Trong ảnh, trục Y hướng xuống dưới, nên "cao hơn" nghĩa là giá trị Y nhỏ hơn.
    
    # Ngón trỏ đến ngón út (index 1 đến 4 trong tip_ids)
    if hand_landmarks.landmark[tip_ids[1]].y < hand_landmarks.landmark[tip_ids[1] - 2].y: # Ngón trỏ
        count += 1
    if hand_landmarks.landmark[tip_ids[2]].y < hand_landmarks.landmark[tip_ids[2] - 2].y: # Ngón giữa
        count += 1
    if hand_landmarks.landmark[tip_ids[3]].y < hand_landmarks.landmark[tip_ids[3] - 2].y: # Ngón áp út
        count += 1
    if hand_landmarks.landmark[tip_ids[4]].y < hand_landmarks.landmark[tip_ids[4] - 2].y: # Ngón út
        count += 1

    # --- 2. Xử lý riêng ngón cái (Thumb) ---
    # Ngón cái di chuyển theo trục ngang (X) là chủ yếu.
    # Logic phụ thuộc vào tay Trái hay Phải.
    
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x # Khớp dưới đầu ngón cái
    
    # Do camera thường bị lật ngược (Mirror), nên logic Left/Right có thể ngược lại tùy camera.
    # Logic dưới đây giả định camera đã được flip hoặc ở chế độ Selfie tiêu chuẩn.
    if hand_label == "Left": 
        # Tay trái: Mở khi đầu ngón cái nằm bên phải khớp (x lớn hơn)
        if thumb_tip_x > thumb_ip_x:
            count += 1
    else: 
        # Tay phải: Mở khi đầu ngón cái nằm bên trái khớp (x nhỏ hơn)
        if thumb_tip_x < thumb_ip_x:
            count += 1

    return count

# --- GIAO DIỆN STREAMLIT (UI) ---
st.set_page_config(layout="wide", page_title="AI Hand Tracking")

st.sidebar.image("https://mediapipe.dev/images/mobile/hand_tracking_3d_android_gpu.gif", use_container_width=True)
st.sidebar.title("⚙️ Cài đặt")
detection_confidence = st.sidebar.slider("Độ nhạy phát hiện", 0.0, 1.0, 0.7)
tracking_confidence = st.sidebar.slider("Độ nhạy theo dõi", 0.0, 1.0, 0.5)

st.title("✌️ AI Finger Counter")
st.write("Giơ tay lên trước camera để đếm số ngón tay.")

# Chia cột: Video và Kết quả
col1, col2 = st.columns([0.7, 0.3])

with col2:
    st.markdown("### 🔢 Kết quả")
    number_placeholder = st.empty() 
    hand_status = st.empty()

run = st.checkbox('Bắt đầu Camera', value=True)
FRAME_WINDOW = col1.image([])

# --- XỬ LÝ CHÍNH ---
cap = cv2.VideoCapture(0)

# Khởi tạo Hands detection
with mp_hands.Hands(
    max_num_hands=1, # Chỉ xử lý 1 tay để tránh rối
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence) as hands:
    
    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Không tìm thấy Camera!")
            break
        
        # Lật ngược ảnh (Mirror) để thao tác tự nhiên hơn (trái là trái, phải là phải)
        frame = cv2.flip(frame, 1)
        
        # 1. Chuẩn bị ảnh
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 2. Đưa ảnh vào AI
        results = hands.process(image)
        
        # 3. Vẽ lại
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        finger_count = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Lấy nhãn tay (Left/Right)
                label = handedness.classification[0].label 
                
                # Tính số ngón tay
                finger_count = count_fingers(image, hand_landmarks, label)
                
                # Vẽ khung xương bàn tay
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )
                
                # Hiển thị số ngón tay ngay trên video (Gần cổ tay)
                wrist_x = int(hand_landmarks.landmark[0].x * image.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * image.shape[0])
                
                cv2.putText(image, f"Count: {finger_count}", (wrist_x - 50, wrist_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Cập nhật UI bên phải
                hand_status.info(f"Tay phát hiện: {label}")

        # Hiển thị kết quả số to bên phải
        number_placeholder.metric("Số ngón tay", finger_count)
        
        # Render lên Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cap.release()