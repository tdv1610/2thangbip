# create_csv_from_video.py
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import cv2
import numpy as np
import csv
from transformers import YolosImageProcessor, YolosForObjectDetection
from project.utils.utils import KalmanBoxTracker, get_center, estimate_distance
import torch

def create_csv_from_video(
    video_folder=r"D:\New folder (4)\2thangbip\archive 2\testing",
    output_folder="../data",
    focal_length=550,
    car_length=2.5,
    camera_speed_kmh=80,
    yolo_model="hustvl/yolos-tiny",
    min_area=800, min_width=10, min_height=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = YolosImageProcessor.from_pretrained(yolo_model)
    model = YolosForObjectDetection.from_pretrained(yolo_model).to(device)

    # --- Nhập tên video ---
    video_name = input("Nhập tên file video (không cần đuôi .mp4): ").strip()
    video_filename = video_name if video_name.lower().endswith(".mp4") else video_name + ".mp4"
    video_path = os.path.join(video_folder, video_filename)

    if not os.path.isfile(video_path):
        print(f"Không tìm thấy file: {video_path}")
        return None

    start_time = input("Nhập thời gian bắt đầu (giây, mặc định 0): ").strip()
    start_time = float(start_time) if start_time else 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    trackers = []
    next_id = 0
    prev_gray = None
    all_records = []
    frame_counter = 0

    print("=== Đang xử lý video, vui lòng đợi... ===")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Đã hết video!")
            break

        frame_counter += 1
        percent = frame_counter / total_frames * 100
        if frame_counter % 10 == 0 or frame_counter == 1:
            print(f"\rTiến độ: {frame_counter}/{total_frames} ({percent:.2f}%)", end='')

        current_second = frame_counter / fps
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Estimate ego-motion
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

        # YOLO detect
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]], device=outputs.logits.device)
        results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        new_trackers = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.detach().cpu().numpy().astype(int)
            label_name = model.config.id2label[label.item()]
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area < min_area or (x2 - x1) < min_width or (y2 - y1) < min_height:
                continue

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
                tracker_used = best_tracker
                new_trackers.append(best_tracker)
            else:
                new_tracker = KalmanBoxTracker(box, next_id, label_name)
                tracker_used = new_tracker
                next_id += 1
                new_trackers.append(new_tracker)

            pred_box = tracker_used.predict()
            # ====== Lấy vận tốc Kalman 8 trạng thái ======
            vx1 = float(tracker_used.kalman.statePost[4, 0])
            vy1 = float(tracker_used.kalman.statePost[5, 0])
            vx2 = float(tracker_used.kalman.statePost[6, 0])
            vy2 = float(tracker_used.kalman.statePost[7, 0])

            object_speed_kmh = tracker_used.get_velocity(fps, car_length, ego_motion, camera_speed_kmh)
            relative_speed_kmh = object_speed_kmh - camera_speed_kmh

            record = {
                "frame": frame_counter,
                "second": round(current_second, 2),
                "id": tracker_used.id,
                "label": tracker_used.label,
                "x1": int(pred_box[0]),
                "y1": int(pred_box[1]),
                "x2": int(pred_box[2]),
                "y2": int(pred_box[3]),
                "vx1": vx1,
                "vy1": vy1,
                "vx2": vx2,
                "vy2": vy2,
                "distance_m": round(estimate_distance(pred_box, focal_length, car_length), 2),
                "speed_kmh": round(relative_speed_kmh, 2),
            }
            all_records.append(record)
        trackers = new_trackers

        # Muốn hiển thị video thì bật dòng dưới
        # cv2.imshow("Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    print("\nXử lý xong video.")

    # Save CSV
    sorted_records = sorted(all_records, key=lambda x: (x["second"], x["frame"], x["id"], x["label"]))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_name = os.path.splitext(video_filename)[0]
    output_file = os.path.join(output_folder, f"{base_name}_data.csv")
    with open(output_file, mode="w", newline="") as csvfile:
        fieldnames = [
            "frame", "second", "id", "label",
            "x1", "y1", "x2", "y2",
            "vx1", "vy1", "vx2", "vy2",
            "distance_m", "speed_kmh"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted_records:
            writer.writerow(rec)
    print(f"Đã xuất thông tin nhận diện ra file: {output_file}")
    return output_file

if __name__ == "__main__":
    create_csv_from_video()
