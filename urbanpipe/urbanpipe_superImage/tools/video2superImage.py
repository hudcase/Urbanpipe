import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_interval=9):
    """
    从视频中提取帧并保存到输出文件夹中。

    参数:
        video_path (str): 视频文件的路径。
        output_folder (str): 输出文件夹的路径。
        frame_interval (int): 提取帧的间隔，默认为9。

    返回:
        无返回值，但会将帧图像保存到指定的输出文件夹中。
    """
    # 确保输出文件夹存在，如果不存在则创建该文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件，返回一个VideoCapture对象，用于后续的视频操作
    cap = cv2.VideoCapture(video_path)

    # 计算视频的总帧数，使用cv2.CAP_PROP_FRAME_COUNT属性获取
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算视频的帧率，使用cv2.CAP_PROP_FPS属性获取
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 检查视频的总帧数是否小于指定的帧间隔
    if total_frames < frame_interval:
        # 如果帧数不足，打印警告信息
        print(f"Warning: Video '{video_path}' has insufficient frames for the specified frame interval.")
        return

    # 计算需要选取的帧数，使用np.linspace函数在3到总帧数 - 2的范围内均匀选取9个整数作为帧编号
    selected_frames = np.linspace(3, total_frames - 2, num=9, dtype=int)

    # 遍历选取的帧编号
    for frame_num in selected_frames:
        # 设置视频捕获对象的当前帧位置为指定的帧编号
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # 读取当前帧，ret为布尔值，表示是否成功读取帧，frame为读取到的帧图像
        ret, frame = cap.read()
        if ret:
            # 如果成功读取帧，生成帧图像的输出路径
            output_path = os.path.join(output_folder, f"frame_{frame_num}.jpg")
            # 将帧图像保存到指定路径
            cv2.imwrite(output_path, frame)

    # 释放视频捕获对象，避免资源占用
    cap.release()

    
def create_grid_image(input_folder, output_path):
    """
    从文件夹中读取帧图像，并将它们组合成九宫格图像后保存。

    参数:
        input_folder (str): 包含帧图像的文件夹的路径。
        output_path (str): 九宫格图像的输出路径。

    返回:
        无返回值，但会将生成的九宫格图像保存到指定的输出路径。
    """
    # 初始化一个空列表，用于存储调整大小后的帧图像
    images_resized = []
    # 获取输入文件夹中的所有文件和文件夹名称
    image_names = os.listdir(input_folder)
    # 循环9次，读取前9个帧图像
    for i in range(9):
        # 生成当前帧图像的完整路径
        frame_path = os.path.join(input_folder, image_names[i])
        # 检查当前帧图像文件是否存在
        if os.path.exists(frame_path):
            # 读取当前帧图像
            frame = cv2.imread(frame_path)
            if frame is not None:
                # 如果成功读取帧图像，将其大小调整为原始大小的一半，使用cv2.resize函数
                frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                # 将调整大小后的帧图像添加到列表中
                images_resized.append(frame_resized)

    # 检查是否有足够的帧图像（即9张）用于创建九宫格图像
    if len(images_resized) < 9:
        # 如果帧数不足，打印错误信息
        print(input_folder, len(images_resized), "Error: Not enough frames available.")
        return

    # 将所有帧图像组合成九宫格图像
    # 先将前3张图像水平拼接，再将中间3张图像水平拼接，最后将后3张图像水平拼接
    # 然后将这三组水平拼接后的图像垂直拼接成一个大的九宫格图像
    grid_image = np.vstack([np.hstack(images_resized[:3]), np.hstack(images_resized[3:6]), np.hstack(images_resized[6:9])])

    # 保存九宫格图像到指定的输出路径
    cv2.imwrite(output_path, grid_image)


if __name__ == "__main__":
    # 创建一个命令行参数解析器，用于解析用户输入的命令行参数
    parser = argparse.ArgumentParser(description="Extract frames from videos and create grid images.")
    # 添加一个命令行参数，用于指定包含视频文件的输入文件夹，默认值为"./dataset/video"
    parser.add_argument("--input_folder", type=str, default="../dataset/tony_data_video", help="Input folder containing videos.")
    # 添加一个命令行参数，用于指定保存提取帧图像的输出文件夹，默认值为"./dataset/videoFrame"
    parser.add_argument("--output_videoFrame_folder", type=str, default="./dataset/videoFrame", help="Output folder to save extracted frames")
    # 添加一个命令行参数，用于指定保存九宫格图像的输出文件夹，默认值为"./dataset/superImage"
    parser.add_argument("--output_superImage_folder", type=str, default="./dataset/superImage", help="Output folder to save grid images.")
    # 解析命令行参数
    args = parser.parse_args()

    # 确保保存提取帧图像的输出文件夹存在，如果不存在则创建该文件夹
    if not os.path.exists(args.output_videoFrame_folder):
        os.makedirs(args.output_videoFrame_folder)

    # 确保保存九宫格图像的输出文件夹存在，如果不存在则创建该文件夹
    if not os.path.exists(args.output_superImage_folder):
        os.makedirs(args.output_superImage_folder)

    # 使用tqdm库显示处理进度条，遍历输入文件夹中的所有文件和文件夹
    for filename in tqdm(os.listdir(args.input_folder)):
        # 只处理扩展名为.mp4的视频文件
        if filename.endswith(".mp4"):
            # 生成当前视频文件的完整路径
            video_path = os.path.join(args.input_folder, filename)
            # 为每个视频创建一个子文件夹，用于保存提取的帧图像
            output_subfolder = os.path.join(args.output_videoFrame_folder, os.path.splitext(filename)[0])
            # 调用extract_frames函数从当前视频中提取帧并保存到子文件夹中
            extract_frames(video_path, output_subfolder)
            # 调用create_grid_image函数将子文件夹中的帧图像组合成九宫格图像并保存到指定的输出文件夹中
            create_grid_image(output_subfolder, os.path.join(args.output_superImage_folder, f"{os.path.splitext(filename)[0]}_superImage.jpg"))

            