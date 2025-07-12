import os
import cv2
import torch
import pickle
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection
from collections import deque, defaultdict
from predictor import LSTMPredictor, BoundingBoxPredictor
from utils import multi_level_lag_scheme, normalize_input, denormalize_output, draw_boxes, KalmanBoxTracker, get_center, compute_iou

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_WIDTH = 50
MIN_HEIGHT = 40
YOLO_SCORE_THRES = 0.85
KALMAN_DIST_THRES = 55
FOCAL_LENGTH = 550
CAR_LENGTH = 2.5

processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
yolo_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(DEVICE)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
lstm_model = LSTMPredictor(input_size=9, hidden_size=64, num_layers=1, output_size=9).to(DEVICE)
lstm_model.load_state_dict(torch.load("bbox_predictor.pth", map_location=DEVICE))
lstm_model.eval()
predictor = BoundingBoxPredictor(lstm_model, scaler)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    trackers = []
    next_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    obj_histories = defaultdict(dict)
    future_preds = {}
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
        results = processor.post_process_object_detection(outputs, threshold=YOLO_SCORE_THRES, target_sizes=target_sizes)[0]

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
        for tracker in trackers:
            bbox = tracker.bbox  # Lấy bbox mới nhất
            obj_histories[tracker.id][frame_idx] = bbox

            state = tracker.kalman.statePost.flatten()
            vx1, vy1, vx2, vy2 = float(state[4]), float(state[5]), float(state[6]), float(state[7])
            speed_kmh = 0  # relative_speed nếu muốn

            if not hasattr(tracker, "history"):
                tracker.history = deque(maxlen=7)
            tracker.history.append([
                bbox[0], bbox[1], bbox[2], bbox[3],
                vx1, vy1, vx2, vy2, speed_kmh
            ])

            if len(tracker.history) == 7:
                seq = np.array(tracker.history)
                input_seq = multi_level_lag_scheme(seq)
                norm_input = normalize_input(input_seq, scaler)
                input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = lstm_model(input_tensor).squeeze(0).cpu().numpy()
                pred_box = denormalize_output(pred, scaler)
                pred_x1, pred_y1, pred_x2, pred_y2 = [int(p) for p in pred_box[:4]]
                future_preds[(tracker.id, frame_idx+1)] = [pred_x1, pred_y1, pred_x2, pred_y2]

            boxes_to_draw.append((bbox, (0, 255, 0), f"ID {tracker.id}-{tracker.label}"))
        for tracker in trackers:
            pred_box = future_preds.get((tracker.id, frame_idx), None)
            if pred_box is not None:
                boxes_to_draw.append((pred_box, (0, 0, 255), f"Pred ID:{tracker.id}"))
                real_box = obj_histories[tracker.id][frame_idx]
                mae = np.mean(np.abs(np.array(real_box) - np.array(pred_box)))
                iou = compute_iou(real_box, pred_box)
                mae_list.append(mae)
                iou_list.append(iou)
                cnt += 1

        draw_boxes(frame, boxes_to_draw)
        cv2.imshow("YOLO+Kalman+LSTM (Green: Real, Red: Predict)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if cnt > 0:
        print("\n===== ĐÁNH GIÁ TRAJECTORY PREDICTION =====")
        print(f"- MAE trung bình (pixel): {np.mean(mae_list):.2f}")
        print(f"- IoU trung bình: {np.mean(iou_list):.4f}")
    else:
        print("Không có đủ dữ liệu để tính MAE/IoU!")

if __name__ == "__main__":
    main("archive 2/testing/challenge.mp4")
