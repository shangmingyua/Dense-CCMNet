"""
LSMI数据集预处理脚本（基于Gehler_Shi预处理流程）
[Final Production Version - Fixed]
1. 修复 single_img_meta 未定义错误
2. 严格使用 metadata.json 生成 GT
"""

import os
import json
import sys
import scipy.io
import numpy as np
import cv2
import rawpy
import logging
from exiftool import ExifToolHelper
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import re

# 添加mcc_utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '_deprecated', 'data_scripts'))
try:
    from mcc_utils import get_binary_mask
except ImportError:
    def get_binary_mask(h, w, point_list):
        mask = np.ones((h, w), dtype=np.uint8)
        pts = np.array(point_list, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (0))
        return mask


def norm_matrix(mat):
    return mat / np.sum(mat, axis=1, keepdims=True)


def exif_to_nparray(exif_matrix):
    matrix_str = exif_matrix.split(' ')
    matrix = np.array([float(x) for x in matrix_str])
    if len(matrix) == 9:
        matrix = matrix.reshape((3, 3))
    elif len(matrix) == 3:
        matrix = np.diag(matrix)
    return matrix


def get_camrgb_for_colortemp(calibration_dict, illum_colortemp, calillum1_cct=2856, calillum2_cct=6504,
                             return_calibration_matrices=False):
    import colour
    assert illum_colortemp != 0
    ab = calibration_dict["ab"]
    cm1 = calibration_dict["cm1"]
    cm2 = calibration_dict["cm2"]
    fm1 = calibration_dict["fm1"]
    fm2 = calibration_dict["fm2"]

    illum_xy = colour.temperature.CCT_to_xy(illum_colortemp, method='Kang 2002')
    illum_XYZ = colour.xy_to_XYZ(illum_xy)

    g = (1 / illum_colortemp - 1 / calillum2_cct) / (1 / calillum1_cct - 1 / calillum2_cct)
    g = np.clip(g, 0, 1)
    cm = g * cm1 + (1 - g) * cm2
    fm = g * fm1 + (1 - g) * fm2

    WBCam2XYZ = fm
    XYZ2WBCam = np.linalg.inv(WBCam2XYZ)
    XYZ2Cam = cm
    Cam2XYZ = np.linalg.inv(XYZ2Cam)
    cam_neutral = np.dot(XYZ2Cam, illum_XYZ)

    if return_calibration_matrices:
        return cam_neutral, WBCam2XYZ, XYZ2WBCam, Cam2XYZ, XYZ2Cam

    return cam_neutral


def mix_chroma(mixmap, chroma_list, illum_count):
    if len(mixmap.shape) == 2:
        mixmap = mixmap[:, :, np.newaxis]
    elif len(mixmap.shape) == 3 and mixmap.shape[2] == 1:
        pass

    ret = np.zeros((mixmap.shape[0], mixmap.shape[1], 3), dtype=np.float32)

    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i]) - 1
        if illum_idx < 0 or illum_idx > 2:
            continue
        if illum_idx < len(chroma_list) and chroma_list[illum_idx] is not None:
            if i >= mixmap.shape[2]:
                continue
            mixmap_channel = mixmap[:, :, i]
            mixmap_3ch = np.stack((mixmap_channel,) * 3, axis=2)
            ret += mixmap_3ch * np.array(chroma_list[illum_idx])

    return ret


def process_single_raw_image(args_tuple):
    (raw_path, cam_name, dng_black_levels_passed, dng_white_level_passed,
     camera_illum_rgb_list,
     camera_wbraw2xyz_list, camera_xyz2wbraw_list, camera_xyz2raw_list, camera_raw2xyz_list,
     img_target_size, cam_target_dir, cam_resized_dir, mcc_data_root,
     do_save_preprocess, do_save_original,
     camera_cm1_list, camera_cm2_list, camera_fm1_list, camera_fm2_list,
     lsmi_meta_data, accurate_gt_data) = args_tuple

    fname = os.path.basename(raw_path).split('.')[0]

    # 输出文件名定义
    target_name = f'{fname}_sensorname_{cam_name}.png'
    target_metadata_name = f'{fname}_sensorname_{cam_name}_metadata.json'
    target_resized_path = os.path.join(cam_resized_dir, target_name)
    target_metadata_path = os.path.join(cam_resized_dir, target_metadata_name)

    # [优化] 跳过逻辑
    if do_save_original:
        if os.path.exists(target_resized_path) and os.path.exists(target_metadata_path):
            if os.path.getsize(target_resized_path) > 0 and os.path.getsize(target_metadata_path) > 0:
                return fname, None

    # 读取图像
    try:
        raw_img_np = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        return fname, None

    if raw_img_np is None:
        return fname, None

    if raw_img_np.ndim == 2:
        raw_img_np = raw_img_np[:, :, None]

    # BGR -> RGB
    raw_img_np = raw_img_np[:, :, ::-1].astype(np.float32)

    # 检测输入类型
    file_ext = os.path.splitext(raw_path)[1].lower()
    is_png_input = file_ext == '.png'

    # 预处理黑电平
    if is_png_input:
        black_level_val = 0
        sat_level_val = 65535
        max_level = sat_level_val - black_level_val
        raw_img_processed = raw_img_np
        raw_img_processed = np.clip(raw_img_processed, 0, max_level)
    else:
        if isinstance(dng_black_levels_passed, (list, tuple, np.ndarray)):
            black_level_val = dng_black_levels_passed[0]
        else:
            black_level_val = dng_black_levels_passed
        max_level = dng_white_level_passed - black_level_val
        raw_img_processed = np.clip(raw_img_np - black_level_val, 0, max_level)

    # =========================================================================
    # [Fixed] 必须在此处初始化 single_img_meta，防止后续 return 报错
    # =========================================================================
    # 模拟白平衡参数 (使用索引0作为默认值，因为我们不依赖这些值进行 GT 计算)
    nearest_idx = 0
    kelvin_temp = int(2500 + nearest_idx)

    # 确保列表不为空
    if camera_xyz2wbraw_list and len(camera_xyz2wbraw_list) > 0:
        single_img_meta = {
            'closest_kelvin_temp': kelvin_temp,
            'xyz2wbraw': camera_xyz2wbraw_list[nearest_idx],
            'wbraw2xyz': camera_wbraw2xyz_list[nearest_idx],
            'xyz2raw': camera_xyz2raw_list[nearest_idx],
            'raw2xyz': camera_raw2xyz_list[nearest_idx],
        }
    else:
        # 如果 calibration 数据缺失，给个空字典防止报错
        single_img_meta = {}

    # -------------------------------------------------------------------------
    # [MASKING] 基于 meta.json 的遮挡
    # -------------------------------------------------------------------------
    mask = np.ones((raw_img_np.shape[0], raw_img_np.shape[1], 1), dtype=np.uint8)
    place_name = None
    match = re.search(r'(Place\d+)', fname)
    if match:
        place_name = match.group(1)

    if place_name and lsmi_meta_data and place_name in lsmi_meta_data:
        place_info = lsmi_meta_data[place_name]
        if "MCCCoord" in place_info:
            mcc_dict = place_info["MCCCoord"]
            for mcc_key, coords_raw in mcc_dict.items():
                try:
                    points_raw = np.array(coords_raw, dtype=np.float32)
                    max_coord_val = np.max(points_raw)
                    max_img_dim = max(raw_img_np.shape[:2])
                    scale = 1.0
                    if max_coord_val > max_img_dim * 1.5:
                        scale = 0.5
                    points = (points_raw * scale).astype(np.int32)
                    cv2.fillPoly(mask, [points], (0))
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # [CORE] GT Illumination & Mixture Map
    # -------------------------------------------------------------------------
    if do_save_original:
        raw_img_masked = raw_img_processed * mask
        if mask.sum() == 0:
            raw_img_masked = raw_img_processed

        raw_img_resized = cv2.resize(raw_img_masked, img_target_size, interpolation=cv2.INTER_AREA)
        if is_png_input:
            raw_img_resized_norm = np.clip(raw_img_resized, 0, 65535.0)
        else:
            raw_img_resized_norm = np.clip(raw_img_resized / max_level * 65535.0, 0, 65535.0)

        gt_map = None
        gt_map_normalized = None

        if lsmi_meta_data is not None:
            # 解析 illum_count (1, 12, 123)
            illum_count_str = None
            fname_clean = fname.split('_sensorname_')[0] if '_sensorname_' in fname else fname

            match = re.search(r'_(\d{1,3})(?:_light|$)', fname_clean)
            if match:
                illum_count_str = match.group(1)
            elif '_light' in fname_clean:
                parts = fname_clean.split('_light')
                if len(parts) > 1 and parts[1]:
                    illum_count_str = parts[1][0]
            else:
                match = re.search(r'_(\d+)$', fname_clean)
                if match:
                    illum_count_str = match.group(1)

            if place_name and place_name in lsmi_meta_data:
                place_info = lsmi_meta_data[place_name]
                num_lights = place_info.get('NumOfLights', 1)

                # -----------------------------------------------------------
                # [GT Logic] 从 accurate_gt_data 获取真实光源值
                # -----------------------------------------------------------
                chroma_list = [None, None, None]
                lights_needed = []

                if num_lights == 1:
                    lights_needed = [1]
                else:
                    if illum_count_str:
                        lights_needed = [int(c) for c in illum_count_str]
                    else:
                        lights_needed = list(range(1, num_lights + 1))

                # 查找 GT
                valid_gt_found = True
                for l_idx in lights_needed:
                    gt_key = f"{cam_name}_{place_name}_{fname_clean}_light{l_idx}"

                    found_light = False
                    if accurate_gt_data and gt_key in accurate_gt_data:
                        if "gt_illum" in accurate_gt_data[gt_key]:
                            chroma_list[l_idx-1] = accurate_gt_data[gt_key]["gt_illum"]
                            found_light = True

                    if not found_light:
                        valid_gt_found = False
                        break

                if not valid_gt_found:
                    return fname, None  # 严格模式：缺 GT 则跳过

                # -----------------------------------------------------------
                # 构建 GT Map
                # -----------------------------------------------------------

                # 单光源情况
                if num_lights == 1 or (illum_count_str and len(illum_count_str) == 1):
                    target_idx = lights_needed[0] - 1
                    if chroma_list[target_idx] is not None:
                        gt_map = np.ones((img_target_size[1], img_target_size[0], 3), dtype=np.float32)
                        gt_map = gt_map * np.array(chroma_list[target_idx])
                        gt_map_normalized = gt_map / (gt_map[:, :, 1:2] + 1e-8)
                        gt_map = gt_map_normalized.copy()
                    else:
                        return fname, None

                # 多光源情况 (需要 Mixture Map)
                else:
                    mixmap = None
                    raw_path_dir = os.path.dirname(raw_path)
                    cam_base_dir = os.path.dirname(raw_path_dir)
                    place_dir = os.path.join(cam_base_dir, place_name)
                    if not os.path.exists(place_dir):
                        place_dir = os.path.join(mcc_data_root, cam_name, place_name)

                    possible_mixmap_names = [
                        f'{fname_clean}.npy',
                        f'{place_name}_{illum_count_str}.npy',
                        f'{place_name}_12.npy',
                        f'{place_name}_13.npy',
                        f'{place_name}_123.npy',
                    ]

                    for mix_name in possible_mixmap_names:
                        p1 = os.path.join(place_dir, mix_name)
                        if os.path.exists(p1):
                            try: mixmap = np.load(p1); break
                            except: pass
                        p2 = os.path.join(raw_path_dir, mix_name)
                        if os.path.exists(p2):
                            try: mixmap = np.load(p2); break
                            except: pass

                    if mixmap is not None:
                        # Resize Mixmap
                        if len(mixmap.shape) == 2:
                            mixmap_resized = cv2.resize(mixmap, (img_target_size[0], img_target_size[1]), interpolation=cv2.INTER_LINEAR)
                            mixmap_resized = mixmap_resized[:, :, np.newaxis]
                        elif len(mixmap.shape) == 3:
                            mixmap_resized = np.zeros((img_target_size[1], img_target_size[0], mixmap.shape[2]), dtype=np.float32)
                            for c in range(mixmap.shape[2]):
                                mixmap_resized[:, :, c] = cv2.resize(mixmap[:, :, c], (img_target_size[0], img_target_size[1]), interpolation=cv2.INTER_LINEAR)
                        else:
                            mixmap_resized = mixmap

                        ZERO_MASK = -1
                        mixmap_resized = np.where(mixmap_resized == ZERO_MASK, 0, mixmap_resized)
                        illum_map = mix_chroma(mixmap_resized, chroma_list, illum_count_str)

                        zero_mask_positions = np.any(mixmap_resized == 0, axis=2) if len(mixmap_resized.shape) == 3 else (mixmap_resized == 0)
                        illum_map[zero_mask_positions] = [1.0, 1.0, 1.0]

                        gt_map = illum_map.copy()
                        gt_map_normalized = gt_map / (gt_map[:, :, 1:2] + 1e-8)
                    else:
                        return fname, None

        if gt_map is None:
            return fname, None

        # 计算并保存全局光照到 single_img_meta
        gt_map_mean = np.mean(gt_map, axis=(0, 1))
        single_img_meta['gt_illum'] = gt_map_mean.tolist()  # [Key Update]

        # 保存 GT Map
        if gt_map_normalized is not None:
            gt_map_to_save = gt_map_normalized.copy()
        else:
            gt_map_to_save = gt_map.copy()
            gt_map_to_save = gt_map_to_save / (gt_map_to_save[:, :, 1:2] + 1e-8)

        gt_map_to_save = np.clip(gt_map_to_save, 0.0, 2.0)
        gt_map_16bit = (gt_map_to_save / 2.0 * 65535.0).astype(np.uint16)
        gt_map_name = target_name.replace('.png', '_gt_map.png')
        cv2.imwrite(os.path.join(cam_resized_dir, gt_map_name), gt_map_16bit[:, :, ::-1])

        # 保存 Image
        cv2.imwrite(os.path.join(cam_resized_dir, target_name), raw_img_resized_norm[:, :, ::-1].astype(np.uint16))

        # 保存 Metadata
        gt_map_mean_normalized = np.mean(gt_map_normalized, axis=(0, 1)) if gt_map_normalized is not None else gt_map_mean
        resized_meta = {
            'illuminant_color_raw': gt_map_mean_normalized.tolist(),
            'gt_map_path': gt_map_name,
            'cm1': camera_cm1_list, 'cm2': camera_cm2_list,
            'fm1': camera_fm1_list, 'fm2': camera_fm2_list
        }
        with open(os.path.join(cam_resized_dir, target_name.replace('.png', '_metadata.json')), 'w') as f:
            json.dump(resized_meta, f, indent=4)

    return fname, single_img_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="预处理LSMI数据集（GS格式）")
    parser.add_argument("--data_root", type=str, default="/mnt/sda2/SMY/ccmnet/LSMI")
    parser.add_argument("--preprocessed_base_dir", type=str, default="/mnt/sda2/SMY/ccmnet/preprocessed_for_augmentation/")
    parser.add_argument("--resized_dir", type=str, default="/mnt/sda2/SMY/ccmnet/original_resized/LSMI/")
    parser.add_argument("--target_size", type=int, nargs=2, default=[384, 256])

    # 增加 metadata.json 路径参数
    parser.add_argument("--metadata_filename", type=str, default="metadata.json", help="包含准确GT的文件名")

    args = parser.parse_args()
    target_size = tuple(args.target_size)

    # [Config]
    SAVE_PREPROCESS = False
    SAVE_ORIGINAL = True

    data_root = args.data_root
    preprocessed_base_dir = args.preprocessed_base_dir
    lsmi_resized_dir = args.resized_dir
    metadata_filename = args.metadata_filename

    os.makedirs(preprocessed_base_dir, exist_ok=True)
    os.makedirs(lsmi_resized_dir, exist_ok=True)

    cams = ['galaxy', 'nikon']

    # 加载 Calibration Data
    calibration_metadata_file = os.path.join(preprocessed_base_dir, 'calibration_metadata.json')
    if os.path.exists(calibration_metadata_file):
        all_cam_calibration_data = json.load(open(calibration_metadata_file))
    else:
        all_cam_calibration_data = {}

    # 加载 meta.json (用于遮挡) 和 metadata.json (用于 GT)
    lsmi_meta_data_all = {}
    accurate_gt_data_all = {}

    # 定义 metadata.json 路径
    camera_gt_paths = {
        'sony': f'/mnt/sda2/SMY/dataset/sony/{metadata_filename}',
        'galaxy': f'/mnt/sda2/SMY/dataset/galaxy/{metadata_filename}',
        'nikon': f'/mnt/sda1/SMY/project/dataset/LSMI/nikon/{metadata_filename}'
    }

    # 定义 meta.json 路径
    camera_meta_paths = {
        'sony': '/mnt/sda2/SMY/dataset/sony/meta.json',
        'galaxy': '/mnt/sda2/SMY/dataset/galaxy/meta.json',
        'nikon': '/mnt/sda1/SMY/project/dataset/LSMI/nikon/meta.json'
    }

    for cam in cams:
        # 1. 加载 meta.json
        cam_meta_path = camera_meta_paths.get(cam)
        if cam_meta_path and os.path.exists(cam_meta_path):
            try:
                with open(cam_meta_path, 'r') as f:
                    lsmi_meta_data_all[cam] = json.load(f)
            except:
                lsmi_meta_data_all[cam] = None
        else:
            lsmi_meta_data_all[cam] = None

        # 2. 加载 accurate metadata.json
        gt_path = camera_gt_paths.get(cam)
        # 回退查找
        if not gt_path or not os.path.exists(gt_path):
            gt_path = os.path.join(data_root, cam, metadata_filename)

        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r') as f:
                    accurate_gt_data_all[cam] = json.load(f)
                print(f"Loaded accurate GT for {cam} from {gt_path}")
            except Exception as e:
                print(f"Error loading GT for {cam}: {e}")
                accurate_gt_data_all[cam] = {}
        else:
            print(f"[Warning] Accurate GT file ({metadata_filename}) not found for {cam}!")
            accurate_gt_data_all[cam] = {}

    # 开始处理
    with ExifToolHelper() as et:
        for cam in tqdm(cams, desc="Processing Cameras"):
            # 准备参数
            target_cam_dir_for_preprocess = os.path.join(preprocessed_base_dir, cam)
            os.makedirs(target_cam_dir_for_preprocess, exist_ok=True)

            cam_dng_dir = os.path.join(data_root, cam, 'dng')
            raw_extensions = {
                'sony': ['.arw', '.dng'],
                'nikon': ['.nef', '.dng'],
                'galaxy': ['.dng']
            }
            extensions = raw_extensions.get(cam.lower(), ['.dng', '.arw', '.nef'])
            dng_list = []

            if not os.path.exists(cam_dng_dir):
                continue

            try:
                files_in_dir = os.listdir(cam_dng_dir)
            except Exception as e:
                continue

            for ext in extensions:
                dng_list.extend([os.path.join(cam_dng_dir, f) for f in files_in_dir
                                 if f.lower().endswith(ext.lower())])

            dng_list = list(set(dng_list))

            if not dng_list:
                continue

            first_dng_file = dng_list[0]
            try:
                dng_raw_obj_temp = rawpy.imread(first_dng_file)
                dng_black_levels_raw = dng_raw_obj_temp.black_level_per_channel
                dng_white_level_raw = dng_raw_obj_temp.white_level
                del dng_raw_obj_temp

                if not isinstance(dng_black_levels_raw, np.ndarray):
                    dng_black_levels_raw = np.array(dng_black_levels_raw, dtype=np.uint16)

                if cam.lower() == 'sony':
                    dng_black_levels = np.array([128] * len(dng_black_levels_raw), dtype=dng_black_levels_raw.dtype)
                    dng_white_level = 4095
                else:
                    dng_black_levels = dng_black_levels_raw.copy() if isinstance(dng_black_levels_raw,
                                                                                 np.ndarray) else np.array(
                        dng_black_levels_raw, dtype=np.uint16)
                    dng_white_level = dng_white_level_raw
                    if len(dng_black_levels) > 0:
                        dng_black_levels = np.array([min(dng_black_levels)] * len(dng_black_levels_raw),
                                                    dtype=dng_black_levels_raw.dtype)
            except Exception as e:
                continue

            meta_dict = et.get_metadata(first_dng_file)[0]

            try:
                cm1 = exif_to_nparray(meta_dict["EXIF:ColorMatrix1"])
                cm2 = exif_to_nparray(meta_dict["EXIF:ColorMatrix2"])
                fm1 = exif_to_nparray(meta_dict["EXIF:ForwardMatrix1"])
                fm2 = exif_to_nparray(meta_dict["EXIF:ForwardMatrix2"])
            except KeyError as e:
                continue

            illum1 = meta_dict.get("EXIF:CalibrationIlluminant1", 17)
            illum2 = meta_dict.get("EXIF:CalibrationIlluminant2", 21)

            if "EXIF:AnalogBalance" in meta_dict:
                ab = exif_to_nparray(meta_dict["EXIF:AnalogBalance"])
            else:
                ab = np.eye(3, dtype=np.float32)

            calibration_dict_internal = {"ab": ab, "cm1": cm1, "cm2": cm2, "fm1": fm1, "fm2": fm2}

            cam_illum_rgb_list, cam_xyz2wbraw_list, cam_wbraw2xyz_list, cam_xyz2raw_list, cam_raw2xyz_list = [], [], [], [], []
            for colortemp_val in range(2500, 7501):
                illumrgb, wbraw2xyz_m, xyz2wbraw_m, raw2xyz_m, xyz2raw_m = get_camrgb_for_colortemp(
                    calibration_dict_internal, colortemp_val, return_calibration_matrices=True)
                illumrgb = illumrgb / illumrgb[1]
                cam_illum_rgb_list.append(illumrgb.tolist())
                cam_xyz2wbraw_list.append(wbraw2xyz_m.tolist())
                cam_wbraw2xyz_list.append(wbraw2xyz_m.tolist())
                cam_xyz2raw_list.append(xyz2raw_m.tolist())
                cam_raw2xyz_list.append(raw2xyz_m.tolist())

            all_cam_calibration_data[cam] = {
                'black_level': int(dng_black_levels[0]),
                'white_level': int(dng_white_level),
                'CalibrationIlluminant1': int(illum1),
                'CalibrationIlluminant2': int(illum2),
                'ColorMatrix1': cm1.tolist(),
                'ColorMatrix2': cm2.tolist(),
                'ForwardMatrix1': fm1.tolist(),
                'ForwardMatrix2': fm2.tolist(),
                'AnalogBalance': ab.tolist(),
                'IlluminantRGB': cam_illum_rgb_list,
            }

            cam_raw_img_dir = os.path.join(data_root, cam, 'png')
            if not os.path.exists(cam_raw_img_dir):
                logging.warning(f"PNG dir not found for {cam}: {cam_raw_img_dir}")
                continue

            cam_raw_img_paths = sorted(
                [os.path.join(cam_raw_img_dir, f) for f in os.listdir(cam_raw_img_dir) if f.endswith('.png')])

            # -------------------------------------------------------------
            # [Core Modification] 获取当前相机的 GT 和 Meta
            # -------------------------------------------------------------
            lsmi_meta_data = lsmi_meta_data_all.get(cam)
            accurate_gt_data = accurate_gt_data_all.get(cam) # [New]

            if not lsmi_meta_data or not accurate_gt_data:
                print(f"Skipping {cam} due to missing meta/gt data.")
                continue

            tasks_for_pool = []
            for raw_path_item in cam_raw_img_paths:
                task_args = (
                    raw_path_item, cam, dng_black_levels, dng_white_level,
                    cam_illum_rgb_list,
                    cam_wbraw2xyz_list, cam_xyz2wbraw_list, cam_xyz2raw_list, cam_raw2xyz_list,
                    target_size, target_cam_dir_for_preprocess, lsmi_resized_dir,
                    data_root,
                    SAVE_PREPROCESS, SAVE_ORIGINAL,
                    cm1.tolist(), cm2.tolist(), fm1.tolist(), fm2.tolist(),
                    lsmi_meta_data,
                    accurate_gt_data # [New] 传递准确的 GT 数据
                )
                tasks_for_pool.append(task_args)

            per_cam_meta_results = {}
            if tasks_for_pool:
                num_processes = 16
                print(f"🚀 启动生产模式: {cam} (Processes: {num_processes})")

                with Pool(processes=num_processes) as pool:
                    results_iterator = pool.imap(process_single_raw_image, tasks_for_pool)

                    skipped_count = 0
                    processed_count = 0

                    for fname_res, single_img_meta_res in tqdm(results_iterator, total=len(tasks_for_pool),
                                                               desc=f"Processing {cam}"):
                        if single_img_meta_res is None:
                            skipped_count += 1
                            continue
                        per_cam_meta_results[fname_res] = single_img_meta_res
                        processed_count += 1

                    if skipped_count > 0:
                        logging.info(f"\n[{cam}] Skipped {skipped_count} images (already exist or error).")

            with open(os.path.join(target_cam_dir_for_preprocess, 'metadata.json'), 'w') as f:
                json.dump(per_cam_meta_results, f, indent=4)

    with open(calibration_metadata_file, 'w') as f:
        json.dump(all_cam_calibration_data, f, indent=4)

    logging.info('Done')