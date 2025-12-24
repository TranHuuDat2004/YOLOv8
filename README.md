# 👁️ Vision AI Multi-Tool

> **Đồ án cuối kỳ môn Computer Vision**
>
> **Đề tài:** Ứng dụng tích hợp Đếm ngón tay (Finger Counting) và Đếm lưu lượng xe (Traffic Counting).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green)

## 📖 Giới thiệu
Dự án này là một bộ công cụ Thị giác máy tính (Computer Vision) all-in-one, bao gồm hai module chính phục vụ các mục đích khác nhau:

1.  **✌️ AI Finger Counter:** Sử dụng MediaPipe để nhận diện bàn tay và đếm số lượng ngón tay qua Webcam theo thời gian thực.
2.  **🚗 Traffic Counting System:** Sử dụng mô hình YOLOv8 mạnh mẽ để phát hiện, theo dõi và đếm lưu lượng phương tiện giao thông (xe hơi, xe tải, xe buýt) từ video tải lên.

---

## ✨ Tính năng chi tiết

### Module 1: Đếm Ngón Tay (Finger Counter)
*   **Công nghệ:** MediaPipe Hands.
*   **Input:** Webcam trực tiếp (Real-time).
*   **Chức năng:**
    *   Phát hiện bàn tay trái/phải.
    *   Vẽ khung xương bàn tay lên màn hình.
    *   Thuật toán logic đếm số ngón tay đang mở.
    *   Hiển thị kết quả ngay tức thì.

### Module 2: Đếm Lưu Lượng Xe (Traffic Counter)
*   **Công nghệ:** Ultralytics YOLOv8 (Tracking & Detection).
*   **Input:** Video tải lên (mp4, avi, mov...).
*   **Chức năng:**
    *   Tự động phát hiện các loại phương tiện: Xe hơi, Xe buýt, Xe tải, Xe máy.
    *   Tracking (theo dõi) đối tượng để tránh đếm trùng lặp.
    *   Đếm xe đi qua một vạch kẻ ảo (Virtual Line) trên đường.
    *   Hiển thị tổng số lượng xe đã đếm được.

---

## 🛠 Cài đặt và Chạy ứng dụng

### Bước 1: Clone dự án
```bash
git clone https://github.com/tranhuudat2004/VisionFit-App.git
cd VisionFit-App
```
*(Lưu ý: Tên thư mục có thể khác tùy vào nơi bạn lưu trữ)*

### Bước 2: Cài đặt thư viện
Yêu cầu máy tính đã cài đặt Python. Chạy lệnh sau:

```bash
pip install -r requirements.txt
```

### Bước 3: Chạy từng Module

#### 👉 Để chạy chức năng Đếm Ngón Tay:
```bash
streamlit run finger.py
```
*Sau khi chạy, cấp quyền truy cập Camera trên trình duyệt.*

#### 👉 Để chạy chức năng Đếm Lưu Lượng Xe:
```bash
streamlit run app.py
```
*Sau khi chạy, kéo thả file video giao thông vào giao diện để bắt đầu phân tích.*

---

## 📂 Cấu trúc dự án

```text
Project-Folder/
├── app.py              # Source code: Đếm lưu lượng xe (YOLOv8)
├── finger.py           # Source code: Đếm ngón tay (MediaPipe)
├── requirements.txt    # Danh sách thư viện
├── README.md           # Tài liệu hướng dẫn
└── ...
```

## 🧩 Công nghệ sử dụng
*   **Ngôn ngữ:** Python 3
*   **Giao diện:** Streamlit Framework
*   **AI Core:**
    *   **YOLOv8** (Object Detection & Tracking)
    *   **Google MediaPipe** (Hand Landmarks)
*   **Xử lý ảnh:** OpenCV

## 👥 Nhóm thực hiện
1.  Trần Hữu Đạt - 522H0081
2.  Dương Thị Thùy Linh - 522H0015

---
*Dự án phục vụ mục đích học tập môn Computer Vision.*
