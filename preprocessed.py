import cv2
import os

def resize_and_convert_video(input_path, output_path, width=854, height=480, output_format="avi"):
    """
    Giảm độ phân giải video xuống 480p và chuyển đổi sang định dạng mong muốn (.avi).
    
    - input_path: Đường dẫn file video gốc
    - output_path: Đường dẫn file video sau khi xử lý (không cần đuôi)
    - width, height: Kích thước mong muốn (mặc định 854x480 cho 480p)
    - output_format: Định dạng đầu ra (chỉ hỗ trợ "avi" ở đây)
    """
    
    # Kiểm tra định dạng đầu ra hợp lệ
    if output_format not in ["avi"]:
        raise ValueError("Định dạng không hợp lệ! Chỉ hỗ trợ 'avi'.")
    
    cap = cv2.VideoCapture(input_path)
    # Lấy FPS gốc của video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sử dụng codec XVID cho định dạng AVI
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_file = output_path if output_path.endswith(".avi") else output_path + ".avi"
    
    out = cv2.VideoWriter(output_file, fourcc, original_fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame về 480p (854x480)
        frame_resized = cv2.resize(frame, (width, height))
        out.write(frame_resized)

    cap.release()
    out.release()
    print(f"✅ Video đã được xử lý và lưu tại: {output_file}")

def process_folder(input_folder, output_folder):
    """
    Duyệt qua tất cả các file MP4 trong input_folder, xử lý và lưu kết quả vào output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(input_folder):
        if file.lower().endswith(".mp4"):
            input_path = os.path.join(input_folder, file)
            # Đặt tên file đầu ra giống tên file gốc nhưng chuyển đuôi thành .avi
            output_file_name = os.path.splitext(file)[0] + ".avi"
            output_path = os.path.join(output_folder, output_file_name)
            
            resize_and_convert_video(input_path, output_path, width=854, height=480, output_format="avi")

# Đường dẫn thư mục gốc chứa video và thư mục đầu ra
input_folder = "/Users/nhxtrxng/Desktop/NCKH_2/raw_vid"
output_folder = "preprocessed_vid"

# Thực hiện chuyển đổi
process_folder(input_folder, output_folder)
