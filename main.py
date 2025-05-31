import os
import cv2
import pandas as pd
import torch
import pickle
from predictor import LSTMPredictor, BoundingBoxPredictor
from utils import draw_boxes

# --- Cấu hình ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load scaler ---
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Khởi tạo mô hình LSTM ---
# LSTMPredictor được định nghĩa với input_size=5, output_size=5
model = LSTMPredictor(input_size=5, hidden_size=64, num_layers=1, output_size=5).to(DEVICE)
# Load checkpoint state_dict
state_dict = torch.load("bbox_predictor.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# --- Tạo đối tượng predictor ---
predictor = BoundingBoxPredictor(model, scaler)

# --- Đọc dữ liệu CSV ---
csv_path = "datavideo.csv"  # CSV phải có các cột: frame, id, bbox, speed_kmh, label, … 
df = pd.read_csv(csv_path)

# Tách cột 'bbox' thành các cột: x1, y1, x2, y2
df[['x1', 'y1', 'x2', 'y2']] = df['bbox'].str.split(',', expand=True).astype(float)

# Tạo dictionary detection_boxes: key = frame, value = list các (box, color) nhận diện (màu xanh lá)
detection_boxes = {}
for idx, row in df.iterrows():
    frame_num = int(row['frame'])
    # Box nhận diện: lấy [x1,y1,x2,y2]
    box = [row['x1'], row['y1'], row['x2'], row['y2']]
    # Gán màu xanh lá cho detection
    detection_boxes.setdefault(frame_num, []).append((box, (0, 255, 0)))

# --- Tạo dự đoán cho mỗi đối tượng ---
# Group theo "id" (mỗi đối tượng có dữ liệu theo thời gian)
object_groups = df.groupby("id")
all_predictions = {}
for object_id, group in object_groups:
    preds = predictor.predict(group)
    all_predictions[object_id] = preds

# --- Mở video đầu vào (đường dẫn chính xác) ---
# Đọc đường dẫn video từ file tạm
temp_video_path_file = "video_path.txt"
if not os.path.isfile(temp_video_path_file):
    print(f"Không tìm thấy file lưu đường dẫn video: {temp_video_path_file}")
    exit()

with open(temp_video_path_file, "r") as f:
    video_path = f.read().strip()

if not os.path.isfile(video_path):
    print(f"Không tìm thấy video: {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Không thể mở video: {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS của video: {fps}")
out = None

# Dùng frame index để đối chiếu với dữ liệu (giả sử frame trong CSV phù hợp với thứ tự frame trong video)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy detection boxes từ CSV theo frame
    det_boxes = detection_boxes.get(frame_idx + 1, [])  # giả sử frame trong CSV bắt đầu từ 1
    # Tích hợp vào danh sách vẽ (màu xanh lá)
    boxes_to_draw = det_boxes.copy()
    
    # Với mỗi đối tượng, nếu có dự đoán cho frame hiện tại, thêm vào danh sách (vẽ màu đỏ)
    for object_id, predictions in all_predictions.items():
        # Lưu ý: predictor.predict tạo ra danh sách dự đoán cho frame từ SEQUENCE_LENGTH đến cuối
        # Do đó, nếu frame_idx+1 < SEQUENCE_LENGTH, dự đoán có thể None.
        if frame_idx < len(predictions):
            pred_box = predictions[frame_idx]
            if pred_box is not None:
                # Chỉ lấy 4 giá trị nếu cần so sánh trực quan: [x1, y1, x2, y2]
                pred_box = [float(x) for x in pred_box[:4]]
                boxes_to_draw.append((pred_box, (0, 0, 255)))  # màu đỏ cho dự đoán

    # Vẽ cả detection (xanh lá) và dự đoán (đỏ)
    draw_boxes(frame, boxes_to_draw)
    cv2.imshow("Detection (Green) & Prediction (Red)", frame)

    if out is None:
        out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    out.write(frame)

    # Sử dụng waitKey với thời gian dựa trên fps để giữ tốc độ video gốc
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
