import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

# 1. Cấu hình trang - Chuyển sang layout "centered" để giao diện gọn gàng hơn
st.set_page_config(page_title="YOLOv8 Car Counter", layout="centered")

st.title("🚗 Traffic Counting System using YOLOv8")
st.markdown("---")

# 2. Sidebar chỉ giữ lại các thông số cấu hình phụ
st.sidebar.header("Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
st_count_sidebar = st.sidebar.empty() # Chỗ hiển thị số lượng bên tay trái nếu cần

# Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt')

model = load_model()

# 3. Khu vực chính: Upload file nằm ngay giữa
uploaded_file = st.file_uploader("📤 Drag and drop your video here", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Lưu file upload vào temp
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tạo các cột để căn giữa video và giới hạn kích thước
    # Cột giữa (col2) chiếm tỷ lệ 6, hai cột bên chiếm 1 -> Video sẽ chiếm khoảng 75% màn hình và nằm giữa
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        st_frame = st.empty() # Video sẽ hiển thị ở đây
        start_btn = st.button("🚀 Start Counting", use_container_width=True)
    
    if start_btn:
        line_y = int(height * 0.6)
        counter = 0
        counted_ids = set()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.success("✅ Video processing completed!")
                break

            # Tracking
            results = model.track(frame, persist=True, conf=confidence, classes=[2, 5, 7], tracker="bytetrack.yaml")

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)

                    # Vẽ tâm và ID
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID: {obj_id}", (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Logic đếm
                    if line_y - 10 < cy < line_y + 10:
                        if obj_id not in counted_ids:
                            counter += 1
                            counted_ids.add(obj_id)

            # Vẽ vạch kẻ và hiển thị số lượng
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
            cv2.putText(frame, f"Count: {counter}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Cập nhật số liệu đồng thời ở cả main UI và sidebar
            st_count_sidebar.metric("Total Vehicles", counter)

            # Hiển thị frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Dùng use_container_width=True để nó vừa khít với độ rộng của col2
            st_frame.image(frame, channels="RGB", use_container_width=True)

        cap.release()