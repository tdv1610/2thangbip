import cv2
import os

def split_video_opencv(input_path, output_dir, segment_duration=10):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Không mở được video.")
        return

    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    segment_frames = segment_duration * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    part = 0
    frame_count = 0
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % segment_frames == 0:
            if out:
                out.release()
            output_file = os.path.join(output_dir, f'clip_{part:03d}.mp4')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"Đang xuất: {output_file}")
            part += 1

        out.write(frame)
        frame_count += 1

    if out:
        out.release()
    cap.release()
    print("Hoàn tất tách video.")

# === Cấu hình ===
input_video_path = r"D:\NCKH_2\2thangbip\archive 2\testing\solidYellowLeft.mp4"  # Đường dẫn video gốc
output_folder = "output_clips_cv2"        # Thư mục chứa các đoạn
duration_per_clip = 10                    # Độ dài mỗi đoạn (giây)

split_video_opencv(input_video_path, output_folder, duration_per_clip)
