import cv2
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
import os
import numpy as np
from collections import deque

# ------------------------
# 1. CÀI ĐẶT YOLO VIẾT
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device)

# ------------------------
# 2. ĐỊNH NGHĨA HÀM HỖ TRỢ
# ------------------------
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines, color=(255, 0, 0), thickness=3):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def distance_to_line(point, line):
    # Tính khoảng cách từ điểm point (x0, y0) đến đường thẳng xác định bởi 2 điểm line = [x1, y1, x2, y2]
    x0, y0 = point
    x1, y1, x2, y2 = line
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / (den + 1e-6)

def get_center(box):
    # Box ở định dạng [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Lớp Kalman cho tracking (như cũ)
class KalmanBoxTracker:
    def __init__(self, bbox, id):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.bbox = bbox
        self.id = id

    def update(self, bbox):
        self.kalman.correct(np.array([[bbox[0]], [bbox[1]]], np.float32))
        self.bbox = bbox

    def predict(self):
        pred = self.kalman.predict()
        # Giả sử bbox định dạng [x1, y1, x2, y2], ta giữ nguyên kích thước hiện tại
        return [int(pred[0]), int(pred[1]), self.bbox[2], self.bbox[3]]

# ------------------------
# 3. XỬ LÝ VIDEO VÀ TÍCH HỢP
# ------------------------
video_folder = "/Users/nhxtrxng/Desktop/NCKH_2/raw_vid"
video_name = input("Nhập tên file video (không cần đuôi .avi): ").strip()
video_filename = video_name if video_name.lower().endswith(".mp4") else video_name + ".mp4"
video_path = os.path.join(video_folder, video_filename)

if not os.path.isfile(video_path):
    print(f"Không tìm thấy file: {video_path}")
    exit()

start_time = input("Nhập thời gian bắt đầu (giây, mặc định 0): ").strip()
start_time = float(start_time) if start_time else 0

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Không thể mở video: {video_path}")
    exit()

cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_jump = int(fps * 5)  # Nhảy 5 giây

print("\nCác phím điều khiển:")
print("  'q'  : Thoát video")
print("  'p'  : Tạm dừng/tiếp tục")
print("  ←    : Tua lùi 5 giây")
print("  →    : Tua tiến 5 giây\n")

paused = False
trackers = []
next_id = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Đã hết video!")
            break

        # ------------------------
        # 3.1. YOLO VIẾT: Nhận diện xe
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

            # Cập nhật hoặc khởi tạo tracker
            min_dist = float('inf')
            best_tracker = None
            for tracker in trackers:
                pred_box = tracker.predict()
                dist = np.linalg.norm(np.array(pred_box[:2]) - np.array(box[:2]))
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

            pred_box = new_trackers[-1].predict()
            cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {new_trackers[-1].id} - {text}",
                        (pred_box[0], pred_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        trackers = new_trackers

        # ------------------------
        # 3.2. Lane Detection: Dùng Canny + Hough để phát hiện làn đường
        # ------------------------
        gray_lane = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_lane = cv2.GaussianBlur(gray_lane, (5, 5), 0)
        edges = cv2.Canny(blur_lane, 50, 150)
        h, w = edges.shape
        roi_vertices = np.array([[(0, h),
                                  (w * 0.45, h * 0.6),
                                  (w * 0.55, h * 0.6),
                                  (w, h)]], dtype=np.int32)
        masked_edges = region_of_interest(edges, roi_vertices)
        lane_lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180,
                                     threshold=50, minLineLength=100, maxLineGap=50)
        # Vẽ làn đường (màu xanh dương)
        draw_lines(frame, lane_lines, color=(255, 0, 0), thickness=3)

        # ------------------------
        # 3.3. Phân tích khoảng cách giữa xe và lane
        # ------------------------
        for tracker in trackers:
            pred_box = tracker.predict()  # [x1, y1, x2, y2]
            center = get_center(pred_box)
            if lane_lines is not None:
                distances = []
                for line in lane_lines:
                    for x1, y1, x2, y2 in line:
                        d = distance_to_line(center, [x1, y1, x2, y2])
                        distances.append(d)
                if distances:
                    min_distance = min(distances)
                    cv2.putText(frame, f"Dist: {min_distance:.1f}", center,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # Bạn có thể thêm logic: nếu min_distance < một ngưỡng nào đó, xe đang nằm trong lane
                    if min_distance < 50:
                        cv2.circle(frame, center, 5, (0, 255, 255), -1)

        # Hiển thị kết quả
        cv2.imshow("YOLO ViT + Lane Detection", frame)

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
