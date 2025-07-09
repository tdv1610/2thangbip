import threading
import pickle
import cv2
import torch
import torch.nn as nn
import numpy as np
from queue import Queue
from transformers import YolosImageProcessor, YolosForObjectDetection
from collections import defaultdict, deque
from utils import draw_boxes, multi_level_lag_scheme, normalize_input, denormalize_output
from predictor import LSTMPredictor, BoundingBoxPredictor

# ======================
# Load YOLO-ViT model
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
yolo_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device)

# Load scaler & LSTM model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
lstm_model = LSTMPredictor(input_size=5, hidden_size=64, num_layers=1, output_size=5).to(device)
lstm_model.load_state_dict(torch.load("bbox_predictor.pth", map_location=device))
lstm_model.eval()

# Queues
frame_queue = Queue()
result_queue = Queue()

# History for LSTM
history_buffer = defaultdict(lambda: deque(maxlen=7))
video_path = "archive 2/testing/challenge_video.mp4"
cap = cv2.VideoCapture(video_path)

# ======================
# Tracking ID Assignment (simple IoU or center distance)
# ======================
next_track_id = [0]
active_tracks = {}  # track_id -> last_box
MAX_DISTANCE = 50

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def assign_track_id(new_box):
    new_cx, new_cy = get_center(new_box)
    for tid, last_box in active_tracks.items():
        cx, cy = get_center(last_box)
        if abs(new_cx - cx) < MAX_DISTANCE and abs(new_cy - cy) < MAX_DISTANCE:
            active_tracks[tid] = new_box
            return tid
    # New ID
    tid = next_track_id[0]
    next_track_id[0] += 1
    active_tracks[tid] = new_box
    return tid

# ========== Thread 1: YOLO Detection ==========
def detection_worker():
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
        outputs = yolo_model(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]

        bboxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [int(x.item()) for x in box]
            label_name = yolo_model.config.id2label[label.item()]
            speed_kmh = 0  # chưa dùng

            # Gán track_id cho mỗi đối tượng
            track_id = assign_track_id([x1, y1, x2, y2])

            bboxes.append({
                "frame_id": frame_id,
                "box": [x1, y1, x2, y2],
                "label": label_name,
                "id": track_id,
                "speed_kmh": speed_kmh
            })

            # Cập nhật history riêng theo ID
            history_buffer[track_id].append([x1, y1, x2, y2, speed_kmh])

        frame_queue.put((frame, bboxes))
        frame_id += 1

# ========== Thread 2: Dự đoán LSTM ==========
def prediction_worker():
    while True:
        try:
            frame, bboxes = frame_queue.get(timeout=5)
        except:
            break
        pred_boxes = []

        for obj in bboxes:
            track_id = obj["id"]
            label = obj["label"]
            box = obj["box"]
            x1, y1, x2, y2 = box

            # Vẽ bounding box thực tế
            pred_boxes.append((box, (0, 255, 0), f"Real ID:{track_id}"))

            history = list(history_buffer[track_id])
            if len(history) < 7:
                continue

            seq = np.array(history)
            seq_input = multi_level_lag_scheme(seq)
            norm_input = normalize_input(seq_input, scaler)
            input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = lstm_model(input_tensor).squeeze(0).cpu().numpy()
            pred_box = denormalize_output(pred, scaler)
            pred_x1, pred_y1, pred_x2, pred_y2 = [int(p) for p in pred_box[:4]]
            pred_boxes.append(([pred_x1, pred_y1, pred_x2, pred_y2], (0, 0, 255), f"Pred ID:{track_id}"))

        result_queue.put((frame, pred_boxes))

# ========== Thread 3: Display ==========
def display_worker():
    while True:
        try:
            frame, boxes = result_queue.get(timeout=5)
        except:
            break
        draw_boxes(frame, boxes)
        cv2.imshow("YOLO + LSTM Prediction", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ========== Run Threads ==========
t1 = threading.Thread(target=detection_worker)
t2 = threading.Thread(target=prediction_worker)
t3 = threading.Thread(target=display_worker)

t1.start()
t2.start()
t3.start()
t1.join()
t2.join()
t3.join()
