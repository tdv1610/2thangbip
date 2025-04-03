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
# Load state_dict từ file checkpoint
state_dict = torch.load("bbox_predictor.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# --- Tạo đối tượng predictor ---
predictor = BoundingBoxPredictor(model, scaler)

# --- Đọc dữ liệu CSV ---
csv_path = "data/challenge_data.csv"  # CSV cần có các cột: frame, id, bbox, speed_kmh, label, … 
df = pd.read_csv(csv_path)

# Tách cột 'bbox' thành các cột: x1, y1, x2, y2
df[['x1', 'y1', 'x2', 'y2']] = df['bbox'].str.split(',', expand=True).astype(float)

# Tạo dictionary detection_boxes: key = frame, value = list các (box, color, text)
detection_boxes = {}
for idx, row in df.iterrows():
    frame_num = int(row['frame'])
    box = [row['x1'], row['y1'], row['x2'], row['y2']]
    text = f"ID {int(row['id'])} - {row['label']}"
    detection_boxes.setdefault(frame_num, []).append((box, (0, 255, 0), text))

# Tạo dictionary chứa nhãn cho mỗi đối tượng
object_labels = {}
for object_id, group in df.groupby("id"):
    object_labels[object_id] = group['label'].iloc[0]

# --- Tạo dự đoán cho mỗi đối tượng ---
# Group theo "id"
object_groups = df.groupby("id")
all_predictions = {}
for object_id, group in object_groups:
    preds = predictor.predict(group)
    all_predictions[object_id] = preds

# --- Mở video đầu vào ---
video_path = "/Users/nhxtrxng/2thangbip/archive 2/testing/challenge.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Không thể mở video: {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS của video: {fps}")
out = None

# Danh sách để lưu thông tin dự đoán (CSV)
predicted_data = []

# Rewind video (giả sử frame trong CSV bắt đầu từ 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tính thời gian (giây) dựa trên frame index và fps
    second = round((frame_idx + 1) / fps, 2)
    
    # --- Vẽ bounding box nhận diện (màu xanh lá) ---
    det_boxes = detection_boxes.get(frame_idx + 1, [])
    boxes_to_draw = []
    for box, color, text in det_boxes:
        boxes_to_draw.append((box, color))
        cv2.putText(frame, text, (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Vẽ bounding box dự đoán (màu đỏ) và lưu thông tin dự đoán ---
    for object_id, predictions in all_predictions.items():
        if frame_idx < len(predictions):
            pred_box = predictions[frame_idx]
            if pred_box is not None:
                # Lấy 4 giá trị để so sánh trực quan: [x1, y1, x2, y2]
                pred_box_coords = [float(x) for x in pred_box[:4]]
                boxes_to_draw.append((pred_box_coords, (0, 0, 255)))
                label_text = f"ID {object_id} - {object_labels.get(object_id, '')}"
                cv2.putText(frame, label_text, (int(pred_box_coords[0]), int(pred_box_coords[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Lưu thông tin dự đoán vào danh sách (dạng dictionary)
                predicted_data.append({
                    "frame": frame_idx + 1,
                    "second": second,
                    "id": object_id,
                    "label": object_labels.get(object_id, ""),
                    "bbox": ",".join([str(int(x)) for x in pred_box_coords])
                })

    # Vẽ tất cả bounding box lên frame
    draw_boxes(frame, boxes_to_draw)
    cv2.imshow("Detection (Green) & Prediction (Red)", frame)

    if out is None:
        out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    out.write(frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

# --- Xuất dữ liệu dự đoán ra CSV ---
predicted_df = pd.DataFrame(predicted_data, columns=["frame", "second", "id", "label", "bbox"])
predicted_df.to_csv("predicted_data.csv", index=False)
print("Đã xuất dữ liệu dự đoán ra file predicted_data.csv")
