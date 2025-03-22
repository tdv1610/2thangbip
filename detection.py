import cv2
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
import os
import numpy as np
from collections import deque

# Cấu hình thiết bị (GPU nếu có, ngược lại dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor và mô hình YOLO ViT
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device)

# Đường dẫn thư mục chứa video đã tiền xử lý
video_folder = "/Users/nhxtrxng/Desktop/NCKH_2/raw_vid"

# Nhập tên file video (không cần đuôi .avi)
video_name = input("Nhập tên file video (không cần đuôi .avi): ").strip()
video_filename = video_name if video_name.lower().endswith(".mp4") else video_name + ".mp4"
video_path = os.path.join(video_folder, video_filename)

# Kiểm tra file video có tồn tại không
if not os.path.isfile(video_path):
    print(f"Không tìm thấy file: {video_path}")
    exit()

# Nhập thời gian bắt đầu (giây)
start_time = input("Nhập thời gian bắt đầu (giây, mặc định 0): ").strip()
start_time = float(start_time) if start_time else 0

# Mở video bằng OpenCV
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Không thể mở video: {video_path}")
    exit()

# Đặt thời gian bắt đầu cho video
cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

# Lấy FPS của video để xử lý tua
fps = cap.get(cv2.CAP_PROP_FPS)
frame_jump = int(fps * 5)  # Nhảy 5 giây

print("\nCác phím điều khiển:")
print("  'q'  : Thoát video")
print("  'p'  : Tạm dừng/tiếp tục")
print("  ←    : Tua lùi 5 giây")
print("  →    : Tua tiến 5 giây\n")

paused = False  # Trạng thái tạm dừng

# Khởi tạo bộ lọc Kalman cho mỗi đối tượng
class KalmanBoxTracker:
    def __init__(self, bbox, id):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.bbox = bbox
        self.id = id

    def update(self, bbox):
        self.kalman.correct(np.array([[bbox[0]], [bbox[1]]], np.float32))
        self.bbox = bbox

    def predict(self):
        pred = self.kalman.predict()
        return [int(pred[0]), int(pred[1]), self.bbox[2], self.bbox[3]]

trackers = []
next_id = 0  # Bộ đếm ID cho các tracker

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Đã hết video!")
            break

        # Chuyển đổi frame từ BGR (OpenCV) sang RGB (mô hình yêu cầu)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tiền xử lý ảnh và chạy mô hình YOLO ViT
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # Chuyển kết quả dự đoán về tọa độ pixel dựa trên kích thước frame
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        # Cập nhật hoặc khởi tạo bộ lọc Kalman cho mỗi bounding box
        new_trackers = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.detach().cpu().numpy().astype(int)
            label_name = model.config.id2label[label.item()]
            text = f"{label_name}: {score:.2f} ({box[0]},{box[1]})-({box[2]},{box[3]})"

            # Tìm tracker gần nhất
            min_dist = float('inf')
            best_tracker = None
            for tracker in trackers:
                pred_box = tracker.predict()
                dist = np.linalg.norm(np.array(pred_box[:2]) - np.array(box[:2]))
                if dist < min_dist:
                    min_dist = dist
                    best_tracker = tracker

            if min_dist < 50:  # Ngưỡng khoảng cách để cập nhật tracker
                best_tracker.update(box)
                new_trackers.append(best_tracker)
            else:
                new_tracker = KalmanBoxTracker(box, next_id)
                next_id += 1
                new_trackers.append(new_tracker)

            # Vẽ bounding box, hiển thị nhãn, điểm số, tọa độ và ID lên frame
            pred_box = new_trackers[-1].predict()
            cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {new_trackers[-1].id} - {text}", (pred_box[0], pred_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        trackers = new_trackers

        # Hiển thị frame đã xử lý
        cv2.imshow("YOLO ViT Detection", frame)

    # Chờ phím nhấn (giảm độ trễ để macOS xử lý mượt hơn)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        print("Thoát video!")
        break

    elif key == ord('p'):
        paused = not paused
        print("Tạm dừng. Nhấn 'p' để tiếp tục..." if paused else "Tiếp tục.")

    elif key == 81:  # Mũi tên trái (←)
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = max(0, current_frame - frame_jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        print(f"Tua lùi đến {new_frame / fps:.2f} giây")

    elif key == 83:  # Mũi tên phải (→)
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = min(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + frame_jump)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        print(f"Tua tiến đến {new_frame / fps:.2f} giây")

cap.release()
cv2.destroyAllWindows()