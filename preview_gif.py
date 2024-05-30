from PIL import Image
import os

def save_gif_frames(gif_path, output_folder, resize=False):
    # 打开GIF文件
    with Image.open(gif_path) as img:
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 初始化帧计数器
        frame_count = 0

        # 遍历GIF中的每一帧
        while True:
            try:
                # 将当前帧保存为PNG文件
                img.seek(frame_count)
                frame_path = os.path.join(output_folder, f'frame_{frame_count}.png')
                if resize:
                    img1 = img.resize((512, 512))
                else:
                    img1 = img
                img1.save(frame_path)
                print(f'Saved {frame_path}')
                frame_count += 1
            except EOFError:
                # 到达GIF的末尾
                break

# 示例使用
gif_path = "C://Users//wangxiaodong//Downloads//arxiv-Final-3D-instance-20240527T134209Z-001//arxiv-Final-3D-instance//A motocross bike flies through the air, kicking up a huge cloud of dust_masked__16764468080_1_16768280152_novel_view.gif"  # 替换为你的GIF文件路径
output_folder = "C://Users//wangxiaodong//Downloads//arxiv-Final-3D-instance-20240527T134209Z-001//arxiv-Final-3D-instance/motor1_512"  # 替换为你希望保存帧的文件夹
save_gif_frames(gif_path, output_folder, resize=True)
