import os
import sys
import cv2
import torch
import pickle
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection
from collections import deque, defaultdict

# Ensure repository root is on sys.path when running this script directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from project.models.predictor import LSTMSeq2SeqPredictor, BoundingBoxPredictor
from project.utils.utils import normalize_input, denormalize_output, draw_boxes, KalmanBoxTracker, get_center, compute_iou

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_WIDTH = 60
MIN_HEIGHT = 50
YOLO_SCORE_THRES = 0.85
KALMAN_DIST_THRES = 55
FOCAL_LENGTH = 550
CAR_LENGTH = 2.5
PRED_LENGTH = 10  # Số bước dự đoán

# Artifact directories
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
ARRAYS_DIR = os.path.join(ARTIFACTS_DIR, "arrays")
WEIGHTS_DIR = os.path.join(ARTIFACTS_DIR, "weights")

processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
yolo_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(DEVICE)

with open(os.path.join(ARRAYS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

lstm_model = LSTMSeq2SeqPredictor(
    input_size=9, hidden_size=64, num_layers=1, output_size=9, pred_length=PRED_LENGTH
).to(DEVICE)

# Load trọng số đã fine-tune
lstm_model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "finetune.pth"), map_location=DEVICE))
lstm_model.eval()

# Predictor wrapper
predictor = BoundingBoxPredictor(lstm_model, scaler, pred_length=PRED_LENGTH)


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    trackers = []
    next_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    obj_histories = defaultdict(dict)
    future_preds = {}  # (tracker.id, frame_idx_tplus10) -> bbox
    mae_list, iou_list, cnt = [], [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to(DEVICE)
        outputs = yolo_model(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]], device=outputs.logits.device)
        results = processor.post_process_object_detection(
            outputs, threshold=YOLO_SCORE_THRES, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                continue
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": yolo_model.config.id2label[label.item()],
                "score": float(score)
            })

        # Predict step Kalman cho mỗi tracker
        for t in trackers:
            t.predict()

        new_trackers = []
        used = set()
        for det in detections:
            min_dist = float('inf')
            best_tracker = None
            for tracker in trackers:
                dist = np.linalg.norm(
                    np.array(get_center(tracker.bbox)) - np.array(get_center(det["box"]))
                )
                if dist < min_dist:
                    min_dist = dist
                    best_tracker = tracker
            if min_dist < KALMAN_DIST_THRES and best_tracker is not None:
                best_tracker.update(det["box"], det["label"])
                new_trackers.append(best_tracker)
                used.add(best_tracker.id)
            else:
                t = KalmanBoxTracker(det["box"], next_id, det["label"])
                next_id += 1
                new_trackers.append(t)
                used.add(t.id)
        for tracker in trackers:
            if tracker.id not in used:
                tracker.missed += 1
                if tracker.missed < 10:
                    new_trackers.append(tracker)
        trackers = new_trackers

        boxes_to_draw = []
        # Cập nhật history cho từng tracker
        for tracker in trackers:
            bbox = tracker.bbox
            obj_histories[tracker.id][frame_idx] = bbox

            state = tracker.kalman.statePost.flatten()
            vx1, vy1, vx2, vy2 = float(state[4]), float(state[5]), float(state[6]), float(state[7])
            speed_kmh = tracker.get_velocity(fps=fps, car_length=CAR_LENGTH, ego_motion=np.array([0.0, 0.0]))

            if not hasattr(tracker, "history"):
                tracker.history = deque(maxlen=7)
            tracker.history.append([
                bbox[0], bbox[1], bbox[2], bbox[3],
                vx1, vy1, vx2, vy2, speed_kmh
            ])

        # Nếu đủ 7 bước lịch sử thì dự đoán 10 bước tiếp theo
        for tracker in trackers:
            if len(tracker.history) == 7:
                seq = np.array(tracker.history)
                denorm_preds = predictor.predict_from_array(seq, use_mls=False)
                pred_box = denorm_preds[-1]

                # Giữ nguyên kích thước khung YOLO gốc
                x1, y1, x2, y2 = tracker.bbox
                w, h = x2 - x1, y2 - y1

                cx = (pred_box[0] + pred_box[2]) / 2
                cy = (pred_box[1] + pred_box[3]) / 2

                new_x1 = int(cx - w / 2)
                new_y1 = int(cy - h / 2)
                new_x2 = int(cx + w / 2)
                new_y2 = int(cy + h / 2)
                pred_box_int = [new_x1, new_y1, new_x2, new_y2]

                future_preds[(tracker.id, frame_idx + PRED_LENGTH)] = pred_box_int

        # Vẽ bbox thực tế hiện tại (xanh lá)
        for tracker in trackers:
            bbox = tracker.bbox
            boxes_to_draw.append((bbox, (0, 255, 0), f"ID {tracker.id}-{tracker.label}"))

        # Vẽ bbox dự đoán t+10 (đỏ) và tính MAE/IoU
        for tracker in trackers:
            pred_box = future_preds.get((tracker.id, frame_idx), None)
            if pred_box is not None:
                boxes_to_draw.append((pred_box, (0, 0, 255), f"Pred t+10 ID:{tracker.id}"))
                real_box = obj_histories[tracker.id].get(frame_idx, None)
                if real_box is not None:
                    mae = np.mean(np.abs(np.array(real_box) - np.array(pred_box)))
                    iou = compute_iou(real_box, pred_box)
                    mae_list.append(mae)
                    iou_list.append(iou)
                    cnt += 1

        draw_boxes(frame, boxes_to_draw)
        cv2.imshow("YOLO+Kalman+LSTM (Green: Real, Red: t+10 Predict - fixed size)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if cnt > 0:
        print("\n===== ĐÁNH GIÁ TRAJECTORY PREDICTION (t+10, khung cố định) =====")
        print(f"- MAE trung bình (pixel): {np.mean(mae_list):.2f}")
        print(f"- IoU trung bình: {np.mean(iou_list):.4f}")
    else:
        print("Không có đủ dữ liệu để tính MAE/IoU!")


if __name__ == "__main__":
    default_video = os.path.join(ROOT_DIR, "archive 2", "test", "test1.mp4")
    main(default_video)
