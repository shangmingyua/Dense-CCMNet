import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import random

# 配置 logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(message)s')


class Data(Dataset):
    def __init__(self, imgfiles, mode='training', input_size=256, load_hist=False):
        """
        Args:
            imgfiles: 图像文件路径列表
            mode: 'training' 或 'testing'
            input_size: 输入到网络的图像块大小 (Patch Size)
        """
        self.imgfiles = imgfiles
        self.mode = mode
        self.patch_size = input_size

        # 定义 LSMI 数据集的默认黑电平
        # NUS 数据集经过 rawpy 处理后，黑电平通常为 0
        self.default_black_levels = {
            'galaxy': 0,
            'sony': 128,
            'nikon': 0,
            'canon': 2048,  # LSMI Canon 默认为 2048
            'fuji': 0,
            'olympus': 0,
            'panasonic': 0
        }
        # 默认饱和度 (用于归一化)
        self.saturation = 65535.0

        logging.info(f'Creating dataset with {len(self.imgfiles)} examples. Mode: {mode}')

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, i):
        max_attempts = 100

        for attempt in range(max_attempts):
            idx = (i + attempt) % len(self.imgfiles)
            img_file = self.imgfiles[idx]
            file_name = os.path.basename(img_file)
            base_name = os.path.splitext(file_name)[0]
            dir_name = os.path.dirname(img_file)

            # [策略 1] 强制黑名单 (针对 LSMI)
            bad_places = ["Place326", "Place443", "Place444", "Place583",
                          "Place264", "Place274", "Place257", "Place267"]
            if any(bp in img_file for bp in bad_places):
                continue

            if not os.path.exists(img_file):
                continue

            # =======================================================
            # 1. 优先读取元数据 (Metadata)
            # =======================================================
            meta_file = os.path.join(dir_name, base_name + '_metadata.json')

            metadata = {}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass

            # =======================================================
            # 2. 读取与预处理输入图像
            # =======================================================
            try:
                # 读取 16-bit 图像
                img = cv2.imread(img_file, -1)
                if img is None: continue
                if img.max() == 0: continue  # Check 全黑

                # -----------------------------------------------------------
                # [核心修复] 智能黑电平判断逻辑
                # -----------------------------------------------------------
                black_level = 0.0

                if 'black_level' in metadata:
                    # 情况 A: 元数据里明确写了
                    black_level = float(metadata['black_level'])
                else:
                    # 情况 B: 元数据里没写，靠文件名猜
                    f_lower = file_name.lower()

                    # [关键修复] 区分 NUS (通常无需减黑电平) 和 LSMI
                    # NUS 文件名特征: "Canon EOS...", "Nikon D...", 包含空格或特定型号
                    is_nus_naming = ('canon eos' in f_lower) or ('nikon d' in f_lower) or \
                                    ('fujifilm' in f_lower) or ('samsung' in f_lower) or \
                                    ('panasonic' in f_lower) or ('olympus' in f_lower) or \
                                    ('sony slt' in f_lower)

                    if is_nus_naming:
                        # NUS 数据集经过 rawpy 预处理，黑电平已经是 0
                        black_level = 0.0
                    else:
                        # LSMI 数据集逻辑 (原有逻辑)
                        if 'galaxy' in f_lower:
                            black_level = self.default_black_levels['galaxy']
                        elif 'sony' in f_lower:
                            black_level = self.default_black_levels['sony']
                        elif 'nikon' in f_lower:
                            black_level = self.default_black_levels['nikon']
                        elif 'canon' in f_lower:
                            black_level = self.default_black_levels['canon']

                # 执行减法
                img = img.astype(np.float32)
                img = np.maximum(img - black_level, 0)

                # 颜色空间转换 BGR -> RGB
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 归一化
                saturation = float(metadata.get('saturation', self.saturation))
                img = img / (saturation - black_level + 1e-6)
                img = np.clip(img, 0.0, 1.0)

                # [策略 2] 亮度过滤 (阈值 1%)
                img_mean_intensity = img.mean()
                if img_mean_intensity < 0.01:
                    # 只有当它是 GT 相关文件时才不警告，普通图片太黑要警告
                    if attempt == 0 and '_gt' not in file_name:
                        # logging.warning(f"[DARK] Skipping dark img: {file_name} ({img_mean_intensity:.4f})")
                        pass
                    continue

                # Resize
                img_np = cv2.resize(img, (self.patch_size, self.patch_size))

            except Exception as e:
                continue

            # =======================================================
            # 3. 读取 GT Map
            # =======================================================
            cm1 = torch.eye(3).float().flatten()
            cm2 = torch.eye(3).float().flatten()
            num_illuminants = 0

            if 'cm1' in metadata: cm1 = torch.tensor(metadata['cm1'], dtype=torch.float32).flatten()
            if 'cm2' in metadata: cm2 = torch.tensor(metadata['cm2'], dtype=torch.float32).flatten()

            for k in ['Light1', 'Light2', 'Light3']:
                if k in metadata: num_illuminants += 1
            if num_illuminants == 0 and 'gt_ill' in metadata: num_illuminants = 1

            # 3.1 寻找 GT 文件
            gt_map_np = np.ones_like(img_np) * 0.5
            gt_path = None

            # (A) Metadata 指定
            if 'gt_map_path' in metadata:
                temp_path = metadata['gt_map_path']
                if not os.path.isabs(temp_path):
                    temp_path = os.path.join(dir_name, temp_path)
                if os.path.exists(temp_path):
                    gt_path = temp_path

            # (B) 自动搜索
            if gt_path is None:
                candidates = [
                    base_name + '_gt_map.png',
                    base_name + '_gt.tiff',
                    base_name + '_gt.png',
                    base_name + '.npy'
                ]
                for cand in candidates:
                    p = os.path.join(dir_name, cand)
                    if os.path.exists(p):
                        gt_path = p
                        break

            gt_map_loaded = False
            if gt_path is not None:
                try:
                    if gt_path.endswith('.npy'):
                        gt_img = np.load(gt_path)
                    else:
                        gt_img = cv2.imread(gt_path, -1)
                        if gt_img is not None:
                            if len(gt_img.shape) == 2:
                                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
                            else:
                                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                    if gt_img is not None:
                        gt_img = gt_img.astype(np.float32)
                        # 归一化 GT
                        if gt_img.max() > 1.0:
                            gt_img = gt_img / (gt_img.max() + 1e-8)

                        gt_map_np = cv2.resize(gt_img, (self.patch_size, self.patch_size))
                        gt_map_np = np.clip(gt_map_np, 0.0, 10.0)
                        gt_map_loaded = True
                except Exception:
                    pass

            # (C) 回退方案：单光源填充
            if not gt_map_loaded:
                if 'gt_ill' in metadata:
                    gt_vec = np.array(metadata['gt_ill'], dtype=np.float32)
                    if gt_vec.max() > 10.0: gt_vec = gt_vec / (gt_vec.max() + 1e-8)
                    gt_map_np = np.tile(gt_vec, (self.patch_size, self.patch_size, 1))
                    gt_map_np = np.clip(gt_map_np, 0.0, 2.0)

            # =======================================================
            # 4. 数据增强
            # =======================================================
            if self.mode == 'training':
                if random.random() > 0.5:
                    img_np = cv2.flip(img_np, 1)
                    gt_map_np = cv2.flip(gt_map_np, 1)
                if random.random() > 0.5:
                    img_np = cv2.flip(img_np, 0)
                    gt_map_np = cv2.flip(gt_map_np, 0)
                k = random.randint(0, 3)
                if k > 0:
                    img_np = np.rot90(img_np, k).copy()
                    gt_map_np = np.rot90(gt_map_np, k).copy()

            # =======================================================
            # 5. 转换为 Tensor
            # =======================================================
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            gt_map = torch.from_numpy(gt_map_np).permute(2, 0, 1)

            # 暗部强制白平衡
            brightness = img_tensor.mean(dim=0)
            mask_dark = brightness < 0.01
            neutral_color = torch.tensor([1.0, 1.0, 1.0], device=gt_map.device).view(3, 1, 1)
            mask_expanded = mask_dark.unsqueeze(0).expand_as(gt_map)
            gt_map = torch.where(mask_expanded, neutral_color, gt_map)

            # 异常 GT 过滤
            try:
                mean_gt = gt_map.mean(dim=(1, 2))
                g = mean_gt[1].item()
                denom = g if abs(g) > 1e-8 else 1e-8
                mean_gt_norm = (mean_gt / denom).detach().cpu()

                if mean_gt_norm[0].item() > 5.0 or mean_gt_norm[2].item() > 5.0:
                    if attempt == 0:
                        logging.warning(f"[DIRTY_GT] Extreme Gain: {file_name} {mean_gt_norm.tolist()}")
                    continue
            except Exception:
                continue

            return {
                'image': img_tensor,
                'gt_map': gt_map,
                'cm1': cm1,
                'cm2': cm2,
                'num_illuminants': num_illuminants,
                'file_name': file_name,
                'path': img_file
            }

        raise RuntimeError(f"Failed to find valid sample after {max_attempts} attempts")

    @staticmethod
    def load_files(img_dir, sub_dirs=None, exclude_list=None):
        logging.info(f'Loading images from {img_dir}...')
        imgfiles = []
        search_dirs = [img_dir]
        if sub_dirs:
            search_dirs = [os.path.join(img_dir, sub) for sub in sub_dirs]

        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.dng', '.bmp')):
                        # 过滤 GT 文件
                        if '_gt' in file.lower() or 'mask' in file.lower(): continue
                        # 过滤 Metadata 文件
                        if 'metadata' in file.lower(): continue

                        if exclude_list and any(ex in file for ex in exclude_list): continue
                        imgfiles.append(os.path.join(root, file))

        logging.info(f'Found {len(imgfiles)} images in {img_dir}.')
        return imgfiles