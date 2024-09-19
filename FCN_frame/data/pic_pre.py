import cv2
import os

"""
降分辨率
"""
def resize_image(input_path, output_path, scale_percent):
    # 读取图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to load image {input_path}")
        return

    # 获取原始尺寸
    original_dimensions = image.shape[:2]  # (height, width)

    # 计算新的尺寸
    new_dimensions = (int(original_dimensions[1] * scale_percent / 100),
                      int(original_dimensions[0] * scale_percent / 100))

    # 调整图像大小
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    # 保存调整后的图像
    cv2.imwrite(output_path, resized_image)
    print(f"Saved resized image to {output_path}")

def batch_resize_images(input_folder, output_folder, scale_percent):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, scale_percent)


"""
标签图RGB转换单通道
"""
def convert_rgb_to_gray(input_path, output_path):
    # 读取图像
    image_rgb = cv2.imread(input_path)
    if image_rgb is None:
        print(f"Failed to load image {input_path}")
        return

    # 检查图像是否为 RGB 格式
    if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
        # 假设图像是 RGB 格式且每个通道的值相同
        # 直接使用任意一个通道作为灰度图像
        gray_image = image_rgb[:, :, 0]  # 取 R 通道，G 或 B 通道也可以

        # 保存灰度图像
        cv2.imwrite(output_path, gray_image)
        print(f"Saved gray image to {output_path}")
    else:
        print(f"Image {input_path} is not in RGB format")

def batch_convert_rgb_to_gray(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            convert_rgb_to_gray(input_path, output_path)
