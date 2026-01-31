"""
将LSMI数据集整理为Gehler_Shi数据集格式的脚本

功能：
1. 创建相机文件夹（galaxy, sony, nikon）
2. 为每个相机创建dng和png文件夹
3. 从RAW文件（Sony->.arw, Nikon->.nef, Galaxy->.dng）转换为16-bit线性RGB PNG
4. 将RAW文件复制到dng文件夹（供后续脚本读取CCM和黑电平元数据）
5. 生成coordinates文件夹和坐标文件
6. 生成real_illum_568.mat文件（从meta.json提取Light数据）

路径说明：
- Sony 原始数据：/mnt/sda2/SMY/dataset/sony（使用 --sony_root 指定）
- Galaxy 原始数据：/mnt/sda2/SMY/dataset/galaxy（使用 --galaxy_root 指定）
- Nikon 原始数据：/mnt/sda1/SMY/project/dataset/LSMI/nikon（使用 --lsmi_root 指定）

输出结构（与Gehler_Shi格式一致）：
LSMI/
├── galaxy/
│   ├── dng/          (空文件夹)
│   └── png/          (PNG图像)
├── sony/
│   ├── dng/          (空文件夹)
│   └── png/          (PNG图像)
├── nikon/
│   ├── dng/          (空文件夹)
│   └── png/          (PNG图像)
├── coordinates/      (Macbeth色卡坐标文件)
└── real_illum_568.mat (Ground truth illuminant数据)
"""

import os
import json
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
from pathlib import Path
import rawpy
import shutil

# Macbeth色卡的标准坐标（24个色块，6x4排列）
CELLCHART = np.float32([
    [0.25, 0.25], [2.75, 0.25], [2.75, 2.75], [0.25, 2.75],  # 0
    [3.00, 0.25], [5.50, 0.25], [5.50, 2.75], [3.00, 2.75],  # 1
    [5.75, 0.25], [8.25, 0.25], [8.25, 2.75], [5.75, 2.75],  # 2
    [8.50, 0.25], [11.00, 0.25], [11.00, 2.75], [8.50, 2.75],  # 3
    [11.25, 0.25], [13.75, 0.25], [13.75, 2.75], [11.25, 2.75],  # 4
    [14.00, 0.25], [16.50, 0.25], [16.50, 2.75], [14.00, 2.75],  # 5
    [0.25, 3.00], [2.75, 3.00], [2.75, 5.50], [0.25, 5.50],  # 6
    [3.00, 3.00], [5.50, 3.00], [5.50, 5.50], [3.00, 5.50],  # 7
    [5.75, 3.00], [8.25, 3.00], [8.25, 5.50], [5.75, 5.50],  # 8
    [8.50, 3.00], [11.00, 3.00], [11.00, 5.50], [8.50, 5.50],  # 9
    [11.25, 3.00], [13.75, 3.00], [13.75, 5.50], [11.25, 5.50],  # 10
    [14.00, 3.00], [16.50, 3.00], [16.50, 5.50], [14.00, 5.50],  # 11
    [0.25, 5.75], [2.75, 5.75], [2.75, 8.25], [0.25, 8.25],  # 12
    [3.00, 5.75], [5.50, 5.75], [5.50, 8.25], [3.00, 8.25],  # 13
    [5.75, 5.75], [8.25, 5.75], [8.25, 8.25], [5.75, 8.25],  # 14
    [8.50, 5.75], [11.00, 5.75], [11.00, 8.25], [8.50, 8.25],  # 15
    [11.25, 5.75], [13.75, 5.75], [13.75, 8.25], [11.25, 8.25],  # 16
    [14.00, 5.75], [16.50, 5.75], [16.50, 8.25], [14.00, 8.25],  # 17
    [0.25, 8.50], [2.75, 8.50], [2.75, 11.00], [0.25, 11.00],  # 18
    [3.00, 8.50], [5.50, 8.50], [5.50, 11.00], [3.00, 11.00],  # 19
    [5.75, 8.50], [8.25, 8.50], [8.25, 11.00], [5.75, 11.00],  # 20
    [8.50, 8.50], [11.00, 8.50], [11.00, 11.00], [8.50, 11.00],  # 21
    [11.25, 8.50], [13.75, 8.50], [13.75, 11.00], [11.25, 11.00],  # 22
    [14.00, 8.50], [16.50, 8.50], [16.50, 11.00], [14.00, 11.00],  # 23
])

MCCBOX = np.float32([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])


def calculate_patch_centers(mcc_coord_4points, img_width, img_height):
    """
    从MCC的4个角点坐标计算24个色块的中心坐标
    
    Args:
        mcc_coord_4points: 4个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        24个色块的中心坐标列表
    """
    # 将4个角点转换为numpy数组（处理字符串格式的坐标）
    src_points = np.array([[float(p[0]), float(p[1])] for p in mcc_coord_4points], dtype=np.float32)
    
    # 目标坐标（标准MCC框的4个角点）
    dst_points = MCCBOX * 20  # 放大20倍
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 将CELLCHART重塑为24个色块，每个色块4个角点
    cellchart = CELLCHART.reshape(24, 4, 2) * 20
    
    # 计算每个色块的中心坐标
    patch_centers = []
    for i in range(24):
        # 获取色块的4个角点
        corners = cellchart[i]
        # 计算中心点（在变换后的坐标系中）
        center = np.mean(corners, axis=0)
        # 应用逆变换得到原始图像中的坐标
        center_homogeneous = np.array([center[0], center[1], 1.0])
        center_transformed = M @ center_homogeneous
        center_transformed = center_transformed[:2] / center_transformed[2]
        
        patch_centers.append([float(center_transformed[0]), float(center_transformed[1])])
    
    return patch_centers


def generate_coordinates_file(mcc_coords_dict, img_width, img_height, output_path):
    """
    生成coordinates文件
    
    Args:
        mcc_coords_dict: MCC坐标字典，包含mcc1, mcc2, mcc3等
        img_width: 图像宽度
        img_height: 图像高度
        output_path: 输出文件路径
    
    格式：
    - 第一行：图像宽度 高度
    - 第2-5行：MCC的4个角点坐标（用于生成mask）
    - 第6-29行：24个色块的中心坐标
    """
    # 使用第一个MCC的坐标（如果有多个MCC，使用mcc1）
    if 'mcc1' in mcc_coords_dict:
        mcc_coord = mcc_coords_dict['mcc1']
    else:
        # 如果没有mcc1，使用第一个可用的MCC
        first_key = list(mcc_coords_dict.keys())[0]
        mcc_coord = mcc_coords_dict[first_key]
    
    # 计算24个色块的中心坐标
    patch_centers = calculate_patch_centers(mcc_coord, img_width, img_height)
    
    # 写入文件
    with open(output_path, 'w') as f:
        # 第一行：图像宽度 高度
        f.write(f"{int(img_width)} {int(img_height)}\n")
        # 第2-5行：MCC的4个角点坐标（用于生成mask）
        for corner in mcc_coord:
            # 确保坐标是浮点数（处理字符串格式的坐标）
            x = float(corner[0]) if isinstance(corner[0], str) else float(corner[0])
            y = float(corner[1]) if isinstance(corner[1], str) else float(corner[1])
            f.write(f"{x} {y}\n")
        # 第6-29行：24个色块的中心坐标
        for center in patch_centers:
            f.write(f"{center[0]} {center[1]}\n")


def get_image_size_from_raw(raw_path):
    """从RAW文件获取图像尺寸"""
    try:
        with rawpy.imread(raw_path) as raw:
            # 获取RAW图像尺寸
            return raw.sizes.width, raw.sizes.height
    except Exception as e:
        print(f"Warning: Failed to read RAW {raw_path}: {e}")
    return None, None


def convert_raw_to_linear_png(raw_path, output_png_path, camera_name):
    """
    将RAW文件转换为16-bit线性RGB PNG
    
    Args:
        raw_path: RAW文件路径（.arw, .nef, .dng）
        output_png_path: 输出PNG文件路径
        camera_name: 相机名称（用于错误处理）
    
    Returns:
        bool: 转换是否成功
    """
    try:
        with rawpy.imread(raw_path) as raw:
            # 关键处理：使用 postprocess 生成“线性、无相机白平衡”的 16-bit 图像
            #
            # 重要说明：
            # - 对于“Raw/Linear 域”的后续处理（例如再用相机矩阵/CCM 做建模），这里不应把
            #   相机拍摄时的 AWB/Camera WB 烘进图像，否则会把场景光照信息部分抵消掉。
            # - 同时需要禁用 gamma（sRGB 非线性），保持线性响应。
            #
            # 参数要点：
            # - gamma=(1,1): 关闭 Gamma 校正，保持线性
            # - no_auto_bright=True: 关闭自动亮度调整
            # - use_camera_wb=False + user_wb=[1,1,1,1]: 不应用相机白平衡（近似“单位WB”）
            # - output_color=rawpy.ColorSpace.raw: 输出保持在相机 RAW 颜色空间（避免转 sRGB）
            # - output_bps=16: 输出 16-bit
            rgb_16bit = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0],
                output_color=rawpy.ColorSpace.raw,
                output_bps=16
            )
            
            # rgb_16bit是numpy数组，shape为[H, W, 3]，dtype为uint16
            # 直接保存为PNG（OpenCV会自动处理16-bit PNG）
            # 注意：OpenCV使用BGR格式，需要转换
            rgb_16bit_bgr = cv2.cvtColor(rgb_16bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_png_path, rgb_16bit_bgr)
            
            return True
    except Exception as e:
        print(f"Error: Failed to convert RAW {raw_path} to PNG: {e}")
        return False


def find_raw_files(place_path, camera_name):
    """
    根据相机品牌查找对应的RAW文件
    
    Args:
        place_path: Place文件夹路径
        camera_name: 相机名称（sony, nikon, galaxy）
    
    Returns:
        list: RAW文件路径列表
    """
    raw_extensions = {
        'sony': ['.arw'],
        'nikon': ['.nef'],
        'galaxy': ['.dng']
    }
    
    extensions = raw_extensions.get(camera_name.lower(), ['.dng', '.arw', '.nef'])
    raw_files = []
    
    for ext in extensions:
        files = [f for f in os.listdir(place_path) if f.lower().endswith(ext)]
        raw_files.extend([os.path.join(place_path, f) for f in files])
    
    return sorted(raw_files)


def process_camera(camera_name, camera_root, output_root, meta_data, coordinates_output_dir, all_illuminants):
    """
    处理单个相机的数据，重组为Gehler_Shi格式
    
    Args:
        camera_name: 相机名称（如sony, nikon, galaxy）
        camera_root: 相机数据根目录
        output_root: 输出根目录
        meta_data: meta.json的内容
        coordinates_output_dir: coordinates文件夹输出目录
        all_illuminants: 用于收集所有illuminant数据的列表
    """
    print(f"\n处理相机: {camera_name}")
    
    # 创建输出目录结构
    cam_output_dir = os.path.join(output_root, camera_name)
    cam_dng_dir = os.path.join(cam_output_dir, "dng")
    cam_png_dir = os.path.join(cam_output_dir, "png")
    
    # 创建dng文件夹（用于存放RAW文件，供后续脚本读取元数据）
    os.makedirs(cam_dng_dir, exist_ok=True)
    # 创建png文件夹
    os.makedirs(cam_png_dir, exist_ok=True)
    os.makedirs(coordinates_output_dir, exist_ok=True)
    
    # 处理每个Place文件夹
    place_folders = [d for d in os.listdir(camera_root) 
                     if os.path.isdir(os.path.join(camera_root, d)) and d.startswith("Place")]
    place_folders.sort()
    
    processed_count = 0
    skipped_count = 0
    
    for place_name in tqdm(place_folders, desc=f"处理 {camera_name}"):
        place_path = os.path.join(camera_root, place_name)
        
        if place_name not in meta_data:
            print(f"Warning: {place_name} 不在meta.json中，跳过")
            skipped_count += 1
            continue
        
        place_meta = meta_data[place_name]
        mcc_coords = place_meta.get("MCCCoord", {})
        num_lights = place_meta.get("NumOfLights", 1)
        
        if not mcc_coords:
            print(f"Warning: {place_name} 没有MCC坐标，跳过")
            skipped_count += 1
            continue
        
        # 获取该Place的所有RAW文件
        raw_files = find_raw_files(place_path, camera_name)
        
        if not raw_files:
            print(f"Warning: {place_name} 中没有找到RAW文件（{camera_name}），跳过")
            skipped_count += 1
            continue
        
        for raw_path in raw_files:
            # 获取基础文件名（不含扩展名）
            raw_filename = os.path.basename(raw_path)
            base_name = os.path.splitext(raw_filename)[0]
            
            # [Critical Fix] 检查RAW文件大小，如果为0则删除并跳过
            try:
                file_size = os.path.getsize(raw_path)
                if file_size == 0:
                    print(f"Warning: 检测到损坏的RAW文件（大小为0）: {raw_filename}，正在删除...")
                    try:
                        os.remove(raw_path)
                        print(f"  已删除损坏文件: {raw_path}")
                    except Exception as e:
                        print(f"  删除文件失败: {e}")
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"Warning: 无法获取文件大小 {raw_filename}: {e}，跳过")
                skipped_count += 1
                continue
            
            # 获取图像尺寸（从RAW文件）
            img_width, img_height = get_image_size_from_raw(raw_path)
            if img_width is None:
                print(f"Warning: 无法获取RAW图像尺寸 {raw_filename}，跳过")
                skipped_count += 1
                continue
            
            # 注意：根据最新要求，dng 文件夹只用于占位/目录结构，不再向其中写入 RAW 文件
            # 因此，这里不再执行 RAW → dng 目录的复制操作，保持 dng 目录为空
            
            # 对于多光源场景，为每个光源生成一个图像副本
            for light_idx in range(1, num_lights + 1):
                light_key = f"Light{light_idx}"
                if light_key not in place_meta:
                    continue
                
                gt_illum = np.array(place_meta[light_key], dtype=np.float32)
                # 确保归一化（G=1.0）
                if gt_illum[1] != 0:
                    gt_illum = gt_illum / gt_illum[1]
                
                # 收集illuminant数据
                all_illuminants.append(gt_illum.tolist())
                
                # 为每个光源生成单独的文件名
                if num_lights > 1:
                    suffix = f"_light{light_idx}"
                    output_png_light = os.path.join(cam_png_dir, f"{camera_name}_{place_name}_{base_name}{suffix}.png")
                    coord_filename = f"{camera_name}_{place_name}_{base_name}{suffix}_macbeth.txt"
                else:
                    output_png_light = os.path.join(cam_png_dir, f"{camera_name}_{place_name}_{base_name}.png")
                    coord_filename = f"{camera_name}_{place_name}_{base_name}_macbeth.txt"
                
                # [Skip Logic] 检查PNG文件是否已存在，如果存在则跳过
                if os.path.exists(output_png_light):
                    # 检查文件是否有效（大小>0）
                    try:
                        png_size = os.path.getsize(output_png_light)
                        if png_size > 0:
                            # PNG文件已存在且有效，跳过处理
                            continue
                        else:
                            # PNG文件存在但大小为0，删除并重新处理
                            print(f"Warning: 检测到损坏的PNG文件（大小为0）: {os.path.basename(output_png_light)}，将重新生成...")
                            os.remove(output_png_light)
                    except Exception:
                        # 如果无法获取文件大小，也尝试删除并重新处理
                        try:
                            os.remove(output_png_light)
                        except Exception:
                            pass
                
                # [Skip Logic] 检查coordinates文件是否已存在
                coord_path = os.path.join(coordinates_output_dir, coord_filename)
                if os.path.exists(coord_path):
                    # coordinates文件已存在，跳过生成（但继续处理PNG）
                    pass
                else:
                    # 生成coordinates文件
                    generate_coordinates_file(mcc_coords, img_width, img_height, coord_path)
                
                # 将RAW转换为16-bit线性PNG并保存
                if convert_raw_to_linear_png(raw_path, output_png_light, camera_name):
                    processed_count += 1
                else:
                    skipped_count += 1
    
    print(f"完成处理 {camera_name}: 处理 {processed_count} 张图像，跳过 {skipped_count} 张")
    print(f"  - DNG文件夹: {cam_dng_dir} (空文件夹，仅用于目录结构)")
    print(f"  - PNG文件夹: {cam_png_dir} ({processed_count} 张16-bit线性PNG图像)")
    print(f"  - Coordinates: {processed_count} 个文件")


def generate_real_illum_mat(all_illuminants, output_path):
    """
    生成real_illum_568.mat文件
    
    Args:
        all_illuminants: 所有illuminant数据的列表
        output_path: 输出文件路径
    """
    # 转换为numpy数组
    real_rgb = np.array(all_illuminants, dtype=np.float32)
    
    # 保存为.mat文件（格式与Gehler_Shi一致）
    scipy.io.savemat(output_path, {'real_rgb': real_rgb})
    print(f"\n已生成 real_illum_568.mat: {len(all_illuminants)} 个illuminant数据")


def main():
    parser = argparse.ArgumentParser(description="将LSMI数据集整理为Gehler_Shi格式")
    # 注意：Sony 和 Galaxy 原始数据目录已单独移动到 /mnt/sda2/SMY/dataset/ 下，
    # Nikon 仍在 LSMI 根目录下。
    parser.add_argument(
        "--lsmi_root",
        type=str,
        default="/mnt/sda1/SMY/project/dataset/LSMI",
        help="LSMI数据集根目录（包含 nikon/ 等子目录）",
    )
    parser.add_argument(
        "--sony_root",
        type=str,
        default="/mnt/sda2/SMY/dataset/sony",
        help="Sony 原始数据根目录（包含各个 Place 子目录）",
    )
    parser.add_argument(
        "--galaxy_root",
        type=str,
        default="/mnt/sda2/SMY/dataset/galaxy",
        help="Galaxy 原始数据根目录（包含各个 Place 子目录）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/mnt/sda2/SMY/ccmnet/LSMI",
        help="输出根目录（整理后的GS格式数据，将写入 /mnt/sda2/SMY/ccmnet/LSMI 下）",
    )
    parser.add_argument("--coordinates_dir", type=str, default=None,
                       help="coordinates文件夹输出目录（默认在output_root下）")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                       choices=["nikon"],
                       help="指定要处理的相机列表（例如：--cameras nikon galaxy）。如果不指定，默认处理所有相机。")
    
    args = parser.parse_args()
    
    lsmi_root = args.lsmi_root
    sony_root = args.sony_root
    galaxy_root = args.galaxy_root
    output_root = args.output_root
    
    # 如果coordinates_dir未指定，使用output_root下的coordinates文件夹
    if args.coordinates_dir is None:
        coordinates_output_dir = os.path.join(output_root, "coordinates")
    else:
        coordinates_output_dir = args.coordinates_dir
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(coordinates_output_dir, exist_ok=True)
    
    # 读取每个相机的meta.json
    # 如果指定了--cameras参数，只处理指定的相机；否则默认只处理nikon
    if args.cameras is not None:
        cameras = [c.lower() for c in args.cameras]  # 转换为小写
        print(f"指定处理的相机: {cameras}")
    else:
        cameras = ["nikon"]  # 默认只处理nikon
        print(f"默认处理相机: {cameras}")
    
    all_illuminants = []  # 收集所有illuminant数据
    
    for camera_name in cameras:
        # Sony 和 Galaxy 相机的原始数据目录单独配置，Nikon 仍使用 lsmi_root/nikon
        if camera_name.lower() == "sony":
            camera_root = sony_root
        elif camera_name.lower() == "galaxy":
            camera_root = galaxy_root
        else:
            camera_root = os.path.join(lsmi_root, camera_name)
        if not os.path.exists(camera_root):
            print(f"Warning: 相机目录不存在 {camera_root}，跳过")
            continue
        
        # 读取meta.json
        # 首先尝试从separate_json_files目录读取（仍位于 lsmi_root 下）
        separate_json_dir = os.path.join(lsmi_root, "separate_json_files", camera_name)
        meta_path = None
        
        # 检查separate_json_files目录
        if os.path.exists(separate_json_dir):
            separate_meta_path = os.path.join(separate_json_dir, f"{camera_name}_meta.json")
            if os.path.exists(separate_meta_path):
                meta_path = separate_meta_path
                print(f"使用separate_json_files中的meta.json: {meta_path}")
        
        # 如果separate_json_files中没有，尝试从相机目录读取
        if meta_path is None:
            meta_path = os.path.join(camera_root, "meta.json")
        
        if not os.path.exists(meta_path):
            print(f"Warning: meta.json不存在 {meta_path}，跳过")
            continue
        
        # 检查文件是否为空
        file_size = os.path.getsize(meta_path)
        if file_size == 0:
            print(f"Error: meta.json文件为空 {meta_path}，跳过")
            continue
        
        # 读取并解析JSON
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"Error: meta.json文件内容为空 {meta_path}，跳过")
                    continue
                meta_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error: meta.json格式错误 {meta_path}: {e}")
            print(f"文件大小: {file_size} bytes")
            print(f"文件前100个字符: {content[:100] if 'content' in locals() else '无法读取'}")
            continue
        except Exception as e:
            print(f"Error: 读取meta.json失败 {meta_path}: {e}")
            continue
        
        # 处理相机数据
        process_camera(camera_name, camera_root, output_root, meta_data, 
                      coordinates_output_dir, all_illuminants)
    
    # 生成real_illum_568.mat文件
    if all_illuminants:
        real_illum_path = os.path.join(output_root, "real_illum_568.mat")
        generate_real_illum_mat(all_illuminants, real_illum_path)
    else:
        print("Warning: 没有收集到任何illuminant数据，无法生成real_illum_568.mat")
    
    print("\n" + "="*60)
    print("数据整理完成！")
    print("="*60)
    print(f"\n输出目录: {output_root}")
    print(f"Coordinates目录: {coordinates_output_dir}")
    print(f"real_illum_568.mat: {os.path.join(output_root, 'real_illum_568.mat')}")
    print("\n目录结构（与Gehler_Shi格式一致）：")
    print(f"{output_root}/")
    print("├── galaxy/")
    print("│   ├── dng/        (RAW文件，.dng格式)")
    print("│   └── png/        (16-bit线性RGB PNG图像)")
    print("├── sony/")
    print("│   ├── dng/        (RAW文件，.arw格式)")
    print("│   └── png/        (16-bit线性RGB PNG图像)")
    print("├── nikon/")
    print("│   ├── dng/        (RAW文件，.nef格式)")
    print("│   └── png/        (16-bit线性RGB PNG图像)")
    print("├── coordinates/    (Macbeth色卡坐标文件)")
    print("└── real_illum_568.mat (Ground truth illuminant数据)")
    print("\n新增的文件夹：")
    print("  - galaxy/dng/")
    print("  - galaxy/png/")
    print("  - sony/dng/")
    print("  - sony/png/")
    print("  - nikon/dng/")
    print("  - nikon/png/")
    print("  - coordinates/")
    print("  - real_illum_568.mat (文件)")


if __name__ == "__main__":
    main()

