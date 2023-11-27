import cv2

# 调整视频的分辨率
def modify_resolution(input_video_path, output_video_path, new_width=0, new_height=0, ratio=0.8):
    cap = cv2.VideoCapture(input_video_path)

    # 获取原视频的帧率、宽度和高度
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置新的宽度和高度
    if ratio > 0:
        new_width = int(width * ratio)
        new_height = int(height * ratio)

    # 创建 VideoWriter 对象，用于写入新的视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择合适的编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 调整帧的分辨率
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 写入调整后的帧
        out.write(resized_frame)

    # 释放资源
    cap.release()
    out.release()

# 示例使用：将视频的分辨率调整为新的宽度和高度
input_video_path = "data/video/fjdc.mp4"  # 替换为你的输入视频文件路径
output_video_path = "data/video/fjdc2.mp4"  # 替换为输出视频文件路径

# 调整视频的分辨率为原来的 0.5 倍
modify_resolution(input_video_path, output_video_path, ratio=0.5)
