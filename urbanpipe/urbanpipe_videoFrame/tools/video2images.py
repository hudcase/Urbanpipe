import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_folder, num_frames=16):
    """
    从视频中提取指定数量的帧并保存到输出文件夹中。

    参数:
        video_path (str): 视频文件的路径。
        output_folder (str): 输出文件夹的路径。
        num_frames (int): 要提取的帧的数量，默认为16。

    返回:
        无返回值，但会将帧图像保存到指定的输出文件夹中。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 计算视频帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 检查视频帧数是否足够
    if total_frames < num_frames:
        print(f"Warning: Video '{video_path}' has insufficient frames.")
        return

    # 计算选取的帧数
    selected_frames = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    # 提取并保存帧图像
    index = 0
    for frame_num in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            # 生成帧图像的输出路径
            output_path = os.path.join(output_folder, f"frame_{index}.jpg")
            # 将帧图像保存到指定路径
            cv2.imwrite(output_path, frame)
            index += 1

    # 释放视频对象
    cap.release()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--input_folder", type=str, default="dataset/tony_data_video", help="Input folder containing videos.")
    parser.add_argument("--output_folder", type=str, default="dataset/tony_data_videoFrame", help="Output folder to save extracted frames.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to extract from each video.")
    args = parser.parse_args()

    # 确保输出文件夹存在
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # 处理每个视频文件
    file_names = os.listdir(args.input_folder)
    file_names.sort()
    file_names.reverse()
    for filename in tqdm(file_names):
        if filename.endswith(".mp4"):  # 只处理视频文件
            video_path = os.path.join(args.input_folder, filename)
            output_subfolder = os.path.join(args.output_folder, os.path.splitext(filename)[0])  # 为每个视频创建一个子文件夹
            extract_frames(video_path, output_subfolder, args.num_frames)



