import cv2
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
import os
import numpy as np
from collections import deque
import pickle

# ------------------------
# 1. CÀI ĐẶT YOLO VIẾT
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device)
model.eval()

# ------------------------
# 1.1. THÔNG SỐ CHO TÍNH KHOẢNG CÁCH
# ------------------------
focal_length = 550    # pixel
car_length = 2.5      # mét

# ------------------------
# 2. HÀM HỖ TRỢ
# ------------------------
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def estimate_distance(bbox, focal_length, car_length):
    pixel_length = bbox[2] - bbox[0]
    if pixel_length <= 0:
        return 0.0  # ✅ KHÔNG trả về inf nữa
    return (car_length * focal_length) / pixel_length

# ------------------------
# Lớp Kalman cho Tracking
# ------------------------
class KalmanBoxTracker:
    def __init__(self, bbox, id):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.bbox = bbox
        self.id = id

    def update(self, bbox):
        measurement = np.array([[bbox[0]], [bbox[1]]], np.float32)
        self.kalman.correct(measurement)
        self.bbox = bbox

    def predict(self):
        pred = self.kalman.predict()
        return [int(pred[0, 0]), int(pred[1, 0]), self.bbox[2], self.bbox[3]]

# ------------------------
# 3. THU THẬP DỮ LIỆU HUẤN LUYỆN TỪ TOÀN BỘ VIDEO TRONG THƯ MỤC
# ------------------------

video_folder = "archive 2/testing"
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".avi"))]

lag_length = 5  # Số frame làm input, frame thứ (lag_length+1) làm target

train_sequences = []
train_targets = []
tracker_histories = {}

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    print(f"\nĐang xử lý video: {video_file}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        continue

    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    trackers = []
    next_id = 0
    tracker_histories = {}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        new_trackers = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.detach().cpu().numpy().astype(int)
            min_dist = float('inf')
            best_tracker = None
            for tracker in trackers:
                t_box = tracker.predict()
                dist = np.linalg.norm(np.array(t_box[:2]) - np.array(box[:2]))
                if dist < min_dist:
                    min_dist = dist
                    best_tracker = tracker
            if min_dist < 50:
                best_tracker.update(box)
                new_trackers.append(best_tracker)
            else:
                new_tracker = KalmanBoxTracker(box, next_id)
                next_id += 1
                new_trackers.append(new_tracker)
        trackers = new_trackers

        for tracker in trackers:
            t_box = tracker.predict()
            distance = estimate_distance(t_box, focal_length, car_length)
            feature = t_box + [distance]
            tid = tracker.id
            if tid not in tracker_histories:
                tracker_histories[tid] = deque(maxlen=lag_length+1)
            tracker_histories[tid].append(feature)
            if len(tracker_histories[tid]) == lag_length + 1:
                history = list(tracker_histories[tid])
                train_sequences.append(history[:-1])
                train_targets.append(history[-1])

        progress = (frame_idx / total_frames) * 100
        print(f"\rVideo {video_file}: {progress:.2f}% đã xử lý", end="")

    cap.release()
    print(f"\nHoàn thành video: {video_file}")

print(f"\nTổng số mẫu thu thập: {len(train_sequences)}")

# ------------------------
# 4. LỌC DỮ LIỆU LỖI (inf, nan)
# ------------------------
clean_sequences = []
clean_targets = []

for seq, target in zip(train_sequences, train_targets):
    arr = np.array(seq + [target])  # shape (lag_length+1, 5)
    if not np.isfinite(arr).all():
        continue
    clean_sequences.append(seq)
    clean_targets.append(target)

print(f"Tổng số mẫu hợp lệ sau khi lọc: {len(clean_sequences)}")

with open("train_data.pkl", "wb") as f:
    pickle.dump((clean_sequences, clean_targets), f)

print("✅ Đã lưu dữ liệu sạch vào train_data.pkl")
