from PIL import Image
import numpy as np
import os

def apply_color_map(image_path, output_path):
    """
    根据颜色映射修改图像像素值。

    Args:
        image_path (str): 输入图像路径。
        output_path (str): 输出图像路径。
    """
    # 定义颜色映射字典
    color_map = {
        0: (0, 0, 0),  # 其他
        1: (153, 76, 0),  # 泥滩
        2: (0, 102, 51),  # 芦荻
        3: (0, 204, 0),  # 苔草
        4: (0, 102, 204),  # 水体
        5: (0, 153, 153),  # 浮叶植物
        6: (255, 255, 51),  # 菰
        7: (153, 255, 153),  # 蓼子草
        8: (255, 255, 255),  # 背景
    }

    # 打开图像
    img = Image.open(image_path)

    # 将图像转换为 NumPy 数组
    img_array = np.array(img)

    # 创建一个新的 RGB 图像
    # height, width = img_array.shape
    # rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
    height, width = img_array.shape
    rgb_img = np.full((height, width, 3), 255, dtype=np.uint8)
    # 应用颜色映射
    for class_id, color in color_map.items():
        rgb_img[img_array == class_id] = color

    # 将 NumPy 数组转换回 PIL 图像
    colored_img = Image.fromarray(rgb_img)

    # 保存修改后的图像
    colored_img.save(output_path)


def batch_apply_color_map(input_folder, output_folder):
    """
    批量处理文件夹中的图片并应用颜色映射。

    Args:
        input_folder (str): 输入图像文件夹路径。
        output_folder (str): 输出图像文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.tif', '.png', '.jpg', '.jpeg')):  # 可以根据需要添加或删除文件扩展名
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"colored_{filename}")

            print(f"Processing: {filename}")
            apply_color_map(input_path, output_path)

    print("Batch processing completed.")



