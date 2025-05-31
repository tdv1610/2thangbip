import cv2
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
import os
import numpy as np
import csv

# ------------------------
# 1. CÀI ĐẶT YOLO VIẾT
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device)

# ------------------------
# 1.1. CÀI ĐẶT THÔNG SỐ CHO TÍNH KHOẢNG CÁCH
# ------------------------
focal_length = 550    # tiêu cự camera (pixel)
car_length = 5   # chiều dài xe thực tế (mét)

# ------------------------
# 1.2. CẤU HÌNH MÔ PHỎNG EGO-MOTION
# ------------------------
# Giả sử xe gắn máy quay chạy ổn định 80 km/h
CAMERA_SPEED_KMH = 80

# ------------------------
# 2. HÀM HỖ TRỢ
# ------------------------
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=(255, 0, 0), thickness=3):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def estimate_distance(bbox, focal_length, car_length):
    pixel_length = bbox[2] - bbox[0]
    if pixel_length <= 0:
        return float('inf')
    return (car_length * focal_length) / pixel_length

# ------------------------
# 3. LỚP KALMAN CHO TRACKING (VỚI ƯỚC TÍNH VẬN TỐC TƯƠNG ĐỐI)
# ------------------------
class KalmanBoxTracker:
    def __init__(self, bbox, id, label):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        # Khởi tạo state với [x, y, dx, dy]
        self.kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.bbox = bbox
        self.id = id
        self.label = label  # lưu nhãn của đối tượng
        self.velocity_history = []  # Lưu lịch sử vận tốc để làm mượt

    def update(self, bbox, label):
        self.kalman.correct(np.array([[bbox[0]], [bbox[1]]], np.float32))
        self.bbox = bbox
        self.label = label

    def predict(self):
        pred = self.kalman.predict()
        return [int(pred[0]), int(pred[1]), self.bbox[2], self.bbox[3]]

    def get_velocity(self, fps, car_length, ego_motion):
        # Ước tính vận tốc tương đối từ Kalman (đơn vị pixel/frame)
        velocity = self.kalman.statePost[2:4].flatten()
        # Hiệu chỉnh: loại bỏ chuyển động của máy quay (ego-motion)
        corrected_velocity = velocity - ego_motion
        pixel_width = self.bbox[2] - self.bbox[0]
        if pixel_width <= 0:
            rel_v_mps = 0.0
        else:
            scale = car_length / pixel_width  # chuyển đổi pixel -> mét
            rel_v_mps = np.linalg.norm(corrected_velocity) * fps * scale

        # Làm mượt: trung bình 5 giá trị cuối
        self.velocity_history.append(rel_v_mps)
        if len(self.velocity_history) > 5:
            self.velocity_history.pop(0)
        smoothed_rel_v = np.mean(self.velocity_history)
        
        # Kết hợp với tốc độ của xe gắn máy quay (chuyển từ km/h sang m/s)
        camera_speed_mps = CAMERA_SPEED_KMH / 3.6
        absolute_speed_mps = smoothed_rel_v + camera_speed_mps
        # Trả về km/h
        return absolute_speed_mps * 3.6

# ------------------------
# 4. XỬ LÝ VIDEO VÀ GHI THÔNG TIN RA CSV
# ------------------------
video_folder = "archive 2/testing"
video_name = input("Nhập tên file video (không cần đuôi .avi): ").strip()
video_filename = video_name if video_name.lower().endswith(".mp4") else video_name + ".mp4"
video_path = os.path.join(video_folder, video_filename)

if not os.path.isfile(video_path):
    print(f"Không tìm thấy file: {video_path}")
    exit()
# Lưu đường dẫn video vào file tạm
temp_video_path_file = "video_path.txt"
with open(temp_video_path_file, "w") as f:
    f.write(video_path)

print(f"Đã lưu đường dẫn video vào file: {temp_video_path_file}")
start_time = input("Nhập thời gian bắt đầu (giây, mặc định 0): ").strip()
start_time = float(start_time) if start_time else 0

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Không thể mở video: {video_path}")
    exit()

cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_jump = int(fps * 5)
paused = False
trackers = []
next_id = 0

prev_gray = None

all_records = []  # Danh sách lưu thông tin các vật thể qua các frame
frame_counter = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("")
            break

        frame_counter += 1
        # Tính thời gian (giây) dựa trên frame_counter và fps
        current_second = frame_counter / fps

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------------
        # Ước tính chuyển động của máy quay (ego-motion) từ optical flow
        # ------------------------
        ego_motion = np.array([0, 0], dtype=np.float32)
        if prev_gray is not None:
            features_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            if features_prev is not None:
                features_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, features_prev, None)
                if features_curr is not None and status is not None:
                    good_prev = features_prev[status.flatten() == 1]
                    good_curr = features_curr[status.flatten() == 1]
                    if len(good_prev) > 0:
                        motion_vectors = good_curr - good_prev
                        ego_motion = np.mean(motion_vectors, axis=0)
        prev_gray = current_gray.copy()

        # ------------------------
        # 4.1. YOLO VIẾT: Nhận diện xe
        # ------------------------
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        new_trackers = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.detach().cpu().numpy().astype(int)
            label_name = model.config.id2label[label.item()]
            text = f"{label_name}: {score:.2f} ({box[0]},{box[1]})-({box[2]},{box[3]})"

            # Kiểm tra và cập nhật tracker dựa trên khoảng cách giữa bbox
            min_dist = float('inf')
            best_tracker = None
            for tracker in trackers:
                pred_box = tracker.predict()
                dist = np.linalg.norm(np.array(pred_box[:2]) - np.array(box[:2]))
                if dist < min_dist:
                    min_dist = dist
                    best_tracker = tracker

            if min_dist < 50 and best_tracker is not None:
                best_tracker.update(box, label_name)
                new_trackers.append(best_tracker)
            else:
                new_tracker = KalmanBoxTracker(box, next_id, label_name)
                next_id += 1
                new_trackers.append(new_tracker)

            pred_box = new_trackers[-1].predict()
            cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {new_trackers[-1].id} - {text}",
                        (pred_box[0], pred_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        trackers = new_trackers

        # ------------------------
        # ------------------------
        # 4.3. Ước tính khoảng cách và vận tốc (kết hợp vận tốc tương đối và tốc độ camera)
        # ------------------------
        for tracker in trackers:
            pred_box = tracker.predict()  # [x1, y1, x2, y2]
            center = get_center(pred_box)
            distance_m = estimate_distance(pred_box, focal_length, car_length)
            speed_kmh = tracker.get_velocity(fps, car_length, ego_motion)
            cv2.putText(frame, f"Dist: {distance_m:.1f} m", center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (pred_box[0], pred_box[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Lưu thông tin của từng đối tượng trong frame này
            record = {
                "frame": frame_counter,
                "second": round(current_second, 2),
                "id": tracker.id,
                "label": tracker.label,
                "bbox": f"{pred_box[0]},{pred_box[1]},{pred_box[2]},{pred_box[3]}",
                "distance_m": round(distance_m, 2),
                "speed_kmh": round(speed_kmh, 2)
            }
            all_records.append(record)

              # Hiển thị video
        cv2.imshow("YOLO ViT Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Thoát video!")
            break

cap.release()
cv2.destroyAllWindows()

# ------------------------
# Sau khi xử lý video, sắp xếp và ghi thông tin ra file CSV.
# File CSV sẽ được lưu trong folder "data" với tên '<tênfile>_data.csv'
# ------------------------
sorted_records = sorted(all_records, key=lambda x: (x["second"], x["frame"], x["id"], x["label"]))

# Tạo folder "data" nếu chưa tồn tại
output_folder = "data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tạo tên file CSV từ video_filename (loại bỏ phần mở rộng)
base_name = os.path.splitext(video_filename)[0]
output_file = "datavideo.csv"
with open(output_file, mode="w", newline="") as csvfile:
    fieldnames = ["frame", "second", "id", "label", "bbox", "distance_m", "speed_kmh"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for rec in sorted_records:
        writer.writerow(rec)
import subprocess
subprocess.run(["python", "train_bbox_predictor.py"])