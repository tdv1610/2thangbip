from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def split_video(input_path, output_folder, segment_length=5, min_last_keep=8):
    # Tạo folder output nếu chưa có
    os.makedirs(output_folder, exist_ok=True)

    # Load video
    video = VideoFileClip(input_path)
    duration = video.duration  # Tổng thời lượng video (giây)

    part = 1
    start = 0

    while start < duration:
        end = start + segment_length

        # Nếu còn lại ít hơn min_last_keep thì giữ nguyên đoạn cuối
        if duration - start < min_last_keep:
            end = duration  

        output_path = os.path.join(output_folder, f"{part}.mp4")

        # Cắt đoạn (MoviePy 2.x dùng subclipped)
        subclip = video.subclipped(start, end)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"✅ Đã tạo: {output_path}")

        # Nếu end đã chạm đến cuối video thì dừng
        if end >= duration:
            break

        start += segment_length
        part += 1

    video.close()

# Ví dụ chạy
input_video = os.path.join("archive 2", "testing", "1.mp4")   # video đầu vào
output_folder = os.path.join("archive 2", "Test")                 # thư mục đầu ra

split_video(input_video, output_folder, segment_length=5, min_last_keep=8)
