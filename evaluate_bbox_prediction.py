import pandas as pd
import numpy as np
import math

def parse_bbox(bbox_str):
    """Chuyển chuỗi bbox "x1,y1,x2,y2" thành list float."""
    return list(map(float, bbox_str.split(',')))

def compute_euclidean_error(gt_box, pred_box):
    """Tính MEE: trung bình khoảng cách Euclidean giữa top-left và bottom-right."""
    d1 = math.sqrt((gt_box[0] - pred_box[0])**2 + (gt_box[1] - pred_box[1])**2)
    d2 = math.sqrt((gt_box[2] - pred_box[2])**2 + (gt_box[3] - pred_box[3])**2)
    return (d1 + d2) / 2

def compute_mae(gt_box, pred_box):
    """Tính Mean Absolute Error cho 4 tọa độ."""
    error = (abs(gt_box[0]-pred_box[0]) + abs(gt_box[1]-pred_box[1]) +
             abs(gt_box[2]-pred_box[2]) + abs(gt_box[3]-pred_box[3])) / 4
    return error

def compute_rmse(gt_box, pred_box):
    """Tính Root Mean Squared Error cho 4 tọa độ."""
    mse = ((gt_box[0]-pred_box[0])**2 + (gt_box[1]-pred_box[1])**2 +
           (gt_box[2]-pred_box[2])**2 + (gt_box[3]-pred_box[3])**2) / 4
    return math.sqrt(mse)

def compute_bbox_diagonal(box):
    """Tính đường chéo của bounding box."""
    return math.sqrt((box[2]-box[0])**2 + (box[3]-box[1])**2)

def calculate_errors(ground_truth_csv, predicted_csv):
    # Đọc file CSV
    gt_df = pd.read_csv(ground_truth_csv)
    pred_df = pd.read_csv(predicted_csv)
    
    # Hiển thị tên cột để kiểm tra
    print("Cột ground truth:", gt_df.columns.tolist())
    print("Cột predicted:", pred_df.columns.tolist())
    
    # Chuyển đổi cột bbox từ chuỗi sang list float
    gt_df['bbox_parsed'] = gt_df['bbox'].apply(parse_bbox)
    pred_df['bbox_parsed'] = pred_df['bbox'].apply(parse_bbox)
    
    # Đổi tên cột đã parse để dùng trong merge
    gt_df = gt_df.rename(columns={"bbox_parsed": "bbox_gt"})
    pred_df = pred_df.rename(columns={"bbox_parsed": "bbox_pred"})
    
    # Loại bỏ cột gốc "bbox" để tránh trùng lặp
    gt_df = gt_df[['frame', 'second', 'id', 'label', 'bbox_gt']]
    pred_df = pred_df[['frame', 'second', 'id', 'label', 'bbox_pred']]
    
    # Vì predicted bắt đầu từ frame 7, lọc ground truth chỉ từ frame 7 trở đi
    gt_df = gt_df[gt_df['frame'] >= 7]
    
    # Đảm bảo id có cùng kiểu
    gt_df["id"] = gt_df["id"].astype(str)
    pred_df["id"] = pred_df["id"].astype(str)
    
    # Merge theo frame và id
    merged_df = pd.merge(gt_df, pred_df, on=["frame", "id"], suffixes=("_gt", "_pred"))
    
    if merged_df.empty:
        print("Không có dữ liệu sau khi merge. Kiểm tra lại cột 'frame' và 'id'.")
        return
    
    mee_list = []
    mae_list = []
    rmse_list = []
    diag_list = []
    
    for idx, row in merged_df.iterrows():
        gt_box = row["bbox_gt"]  # list [x1, y1, x2, y2]
        pred_box = row["bbox_pred"]
        diag = compute_bbox_diagonal(gt_box)
        diag_list.append(diag)
        mee = compute_euclidean_error(gt_box, pred_box)
        mae = compute_mae(gt_box, pred_box)
        rmse = compute_rmse(gt_box, pred_box)
        mee_list.append(mee)
        mae_list.append(mae)
        rmse_list.append(rmse)
    
    MEE = np.mean(mee_list)
    MAE = np.mean(mae_list)
    RMSE = np.mean(rmse_list)
    avg_diag = np.mean(diag_list)
    
    # Chuyển đổi sai số thành % độ chính xác
    Accuracy_MEE = 100 - (MEE / avg_diag * 100)
    Accuracy_MAE = 100 - (MAE / avg_diag * 100)
    Accuracy_RMSE = 100 - (RMSE / avg_diag * 100)
    
    print(f"MEE (Mean Euclidean Error): {MEE:.4f}")
    print(f"MAE (Mean Absolute Error): {MAE:.4f}")
    print(f"RMSE (Root Mean Squared Error): {RMSE:.4f}")
    print(f"Average Bounding Box Diagonal: {avg_diag:.4f}")
    print(f"Accuracy (MEE-based): {Accuracy_MEE:.2f}%")
    print(f"Accuracy (MAE-based): {Accuracy_MAE:.2f}%")
    print(f"Accuracy (RMSE-based): {Accuracy_RMSE:.2f}%")

if __name__ == "__main__":
    # Đường dẫn file CSV ground truth và predicted (chỉnh sửa cho phù hợp)
    gt_csv = "/Users/nhxtrxng/2thangbip/data/challenge_data.csv"
    pred_csv = "/Users/nhxtrxng/2thangbip/predicted_data.csv"
    calculate_errors(gt_csv, pred_csv)
