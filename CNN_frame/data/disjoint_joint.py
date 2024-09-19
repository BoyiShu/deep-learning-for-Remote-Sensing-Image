import rasterio
import numpy as np
import os

def split_data(optics_path, labels_path, output_dir,num_rows,num_cols):
    """
    将影像数据和标签数据分割成子图并保存到指定文件夹。

    Args:
        optics_path (str): 影像数据路径。
        labels_path (str): 标签数据路径。
        output_dir (str): 输出文件夹路径。

    Returns:
        None
    """

    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

    # 读取影像数据
    with rasterio.open(optics_path) as src:
        optics_data = src.read()
        optics_meta = src.meta

    # 读取标签数据
    with rasterio.open(labels_path) as src:
        labels_data = src.read()
        labels_meta = src.meta

    # 计算子图大小
    height, width = optics_data.shape[1:]
    sub_height = height // num_rows
    sub_width = width // num_cols

    # 划分图像
    for i in range(num_rows):
        for j in range(num_cols):
            # 提取子图数据
            optics_sub_data = optics_data[:, i * sub_height:(i + 1) * sub_height, j * sub_width:(j + 1) * sub_width]
            labels_sub_data = labels_data[:, i * sub_height:(i + 1) * sub_height, j * sub_width:(j + 1) * sub_width]

            # 更新子图元数据
            sub_meta1 = optics_meta.copy()
            sub_meta2 = labels_meta.copy()
            sub_meta1['height'] = sub_height
            sub_meta1['width'] = sub_width
            sub_meta2['height'] = sub_height
            sub_meta2['width'] = sub_width
            # 更新子图元数据的通道数
            sub_meta1['count'] = optics_sub_data.shape[0]

            # 保存子图
            with rasterio.open(os.path.join(output_dir, f"9_optics_sub_{i}_{j}.tif"), 'w', **sub_meta1) as dst:
                dst.write(optics_sub_data)
                print("optics_sub_data", optics_sub_data.shape)
                np.save(os.path.join(output_dir, f"9_optics_sub_{i}_{j}.npy"), optics_sub_data)

            with rasterio.open(os.path.join(output_dir, f"9_labels_sub_{i}_{j}.tif"), 'w', **sub_meta2) as dst:
                dst.write(labels_sub_data)
                print("labels_sub_data", labels_sub_data.shape)


def stitch_subimages(output_dir, sub_height, sub_width, num_rows, num_cols):
    """
    将多个子图像拼接成一个完整图像。

    Args:
        output_dir (str): 子图像所在的目录。
        sub_height (int): 子图像的高度。
        sub_width (int): 子图像的宽度。
        num_rows (int): 子图像的行数。
        num_cols (int): 子图像的列数。
    """

    # 计算完整图像的尺寸
    height = sub_height * num_rows
    width = sub_width * num_cols

    # 读取子图像数据并检查一致性
    sub_data = []
    for i in range(num_rows):
        for j in range(num_cols):
            with rasterio.open(f"{output_dir}pyh_densenet_result_{i}_{j}.tif") as src:
                sub_data.append(src.read())
                # 检查子图像尺寸是否符合预期
                if src.height != sub_height or src.width != sub_width:
                    raise ValueError(f"Sub-image {i}_{j} has incorrect dimensions. Expected ({sub_height}, {sub_width}), found ({src.height}, {src.width})")

    # 从参考子图像获取元数据
    with rasterio.open(f"{output_dir}pyh_densenet_result_0_0.tif") as src:
        sub_meta = src.meta

    # 更新完整图像的元数据
    sub_meta['height'] = height
    sub_meta['width'] = width
    # 删除不需要的地理空间信息
    sub_meta.pop('transform', None)
    sub_meta.pop('crs', None)

    # 创建完整图像数组
    data = np.zeros((sub_meta['count'], height, width), dtype=sub_data[0].dtype)

    # 将子图像拼接在一起
    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            data[:, i * sub_height:(i + 1) * sub_height, j * sub_width:(j + 1) * sub_width] = sub_data[k]
            k += 1

    # 保存完整图像
    with rasterio.open(f"{output_dir}modified_image.tif", 'w', **sub_meta) as dst:
        dst.write(data)