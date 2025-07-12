# utils.py
import cv2
import numpy as np

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

class KalmanBoxTracker:
    def __init__(self, bbox, id, label):
        # 8 trạng thái: x1, y1, x2, y2, vx1, vy1, vx2, vy2
        self.kalman = cv2.KalmanFilter(8, 4)
        # Đo lường: chỉ đo được x1, y1, x2, y2
        self.kalman.measurementMatrix = np.zeros((4, 8), np.float32)
        self.kalman.measurementMatrix[:4, :4] = np.eye(4)
        # Ma trận chuyển trạng thái
        self.kalman.transitionMatrix = np.eye(8, dtype=np.float32)
        self.velocity_history = []
        self.missed = 0 
        dt = 1.0
        for i in range(4):
            self.kalman.transitionMatrix[i, i+4] = dt
        # Nhiễu
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1
        # Trạng thái ban đầu
        state = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], np.float32)
        self.kalman.statePre = state.reshape(8, 1)
        self.kalman.statePost = state.reshape(8, 1)
        self.bbox = bbox[:]
        self.id = id
        self.label = label
        self.velocity_history = []

    def update(self, bbox, label):
        measurement = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], np.float32).reshape(4, 1)
        self.kalman.correct(measurement)
        self.bbox = bbox[:]
        self.label = label

    def predict(self):
        pred = self.kalman.predict()
        x1, y1, x2, y2 = [int(pred[i, 0]) for i in range(4)]
        self.bbox = [x1, y1, x2, y2]
        return [x1, y1, x2, y2]

    def get_velocity(self, fps, car_length, ego_motion, camera_speed_kmh=80):
        # Lấy vx1, vy1, vx2, vy2
        velocity = self.kalman.statePost[4:8, 0]
        vx = (velocity[0] + velocity[2]) / 2
        vy = (velocity[1] + velocity[3]) / 2
        corrected_velocity = np.array([vx, vy]) - ego_motion
        pixel_width = self.bbox[2] - self.bbox[0]
        if pixel_width <= 0:
            rel_v_mps = 0.0
        else:
            scale = car_length / pixel_width
            rel_v_mps = np.linalg.norm(corrected_velocity) * fps * scale
        self.velocity_history.append(rel_v_mps)
        if len(self.velocity_history) > 5:
            self.velocity_history.pop(0)
        smoothed_rel_v = np.mean(self.velocity_history)
        camera_speed_mps = camera_speed_kmh / 3.6
        absolute_speed_mps = smoothed_rel_v + camera_speed_mps
        return absolute_speed_mps * 3.6

# ==== Các hàm sequence cho LSTM ====
def multi_level_lag_scheme(seq, lags=[1, 2, 3]):
    n = seq.shape[0]
    if n < max(lags):
        return seq
    lagged = [seq[n - lag - 1:n - lag] for lag in lags]
    return np.concatenate(lagged, axis=0)

def normalize_input(seq, scaler):
    num_features = scaler.mean_.shape[0]
    flat_seq = seq.reshape(-1, num_features)
    normalized = scaler.transform(flat_seq)
    return normalized.reshape(seq.shape)

def denormalize_output(pred, scaler):
    indices = [0, 1, 2, 3]
    pred_norm = pred[indices]
    mean = scaler.mean_[indices]
    scale = scaler.scale_[indices]
    return pred_norm * scale + mean

def draw_boxes(frame, boxes):
    for box in boxes:
        if len(box) == 2:
            (x1, y1, x2, y2), color = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        elif len(box) == 3:
            (x1, y1, x2, y2), color, text = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, text, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def is_valid_bbox(box, min_area=800, min_width=10, min_height=10):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    width = x2 - x1
    height = y2 - y1
    return area >= min_area and width >= min_width and height >= min_height

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def draw_dashed_rectangle(img, box, color, thickness=2, dash_length=8):
    x1, y1, x2, y2 = map(int, box)
    # 4 cạnh
    for (pt1, pt2) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                       ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        draw_dashed_line(img, pt1, pt2, color, thickness, dash_length)

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=8):
    dist = int(np.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]))
    for i in range(0, dist, dash_length*2):
        start = (
            int(pt1[0] + (pt2[0]-pt1[0])*i/dist),
            int(pt1[1] + (pt2[1]-pt1[1])*i/dist)
        )
        end = (
            int(pt1[0] + (pt2[0]-pt1[0])*min(i+dash_length, dist)/dist),
            int(pt1[1] + (pt2[1]-pt1[1])*min(i+dash_length, dist)/dist)
        )
        cv2.line(img, start, end, color, thickness)
