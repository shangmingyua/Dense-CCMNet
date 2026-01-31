# # import sys
# # import os
# # import re
# # import cv2
# # import numpy as np
# # import json
# # import time
# # import random  # [新增] 用于打乱数据
# # from datetime import datetime
# # from collections import defaultdict
# #
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # 获取项目根目录 (.../CCMNet-main)
# # project_root = os.path.dirname(current_dir)
# # # 将根目录加入 Python 搜索路径
# # if project_root not in sys.path:
# #     sys.path.append(project_root)
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from src.ccmnet.ccmnet import network
# # from src import dataset
# # import math
# # import argparse
# # import matplotlib
# #
# # matplotlib.use('Agg')  # 使用非交互式后端
# # import matplotlib.pyplot as plt
# #
# #
# # def angular_error_map(pred, gt):
# #     """ 计算像素级角误差 (Robust Version) - 强制归一化版本 """
# #     # 1. 强制归一化
# #     pred = nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
# #     gt = nn.functional.normalize(gt, p=2, dim=1, eps=1e-8)
# #
# #     # 2. 点积与截断
# #     dot = torch.sum(pred * gt, dim=1)
# #     dot = torch.clamp(dot, -0.999999, 0.999999)
# #
# #     # 3. 计算角度
# #     angle = torch.acos(dot) * (180.0 / math.pi)
# #
# #     # 4. 安全检查
# #     angle = torch.clamp(angle, 0.0, 180.0)
# #
# #     return angle
# #
# #
# # def total_variation_loss(pred_map):
# #     """ Total Variation (TV) Loss """
# #     batch_size = pred_map.size(0)
# #     diff_x = pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]
# #     tv_x = torch.abs(diff_x).sum()
# #     diff_y = pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]
# #     tv_y = torch.abs(diff_y).sum()
# #     H, W = pred_map.size(2), pred_map.size(3)
# #     tv_loss = (tv_x + tv_y) / (batch_size * H * W)
# #     return tv_loss
# #
# #
# # def save_debug_images(img, pred, gt, aux_map=None, epoch=None, img_count=None, save_dir='debug_imgs', prefix='epoch'):
# #     """ 保存可视化结果 """
# #     os.makedirs(save_dir, exist_ok=True)
# #
# #     if img.dim() == 4:
# #         img = img[0].detach().cpu().permute(1, 2, 0).numpy()
# #         pred = pred[0].detach().cpu().permute(1, 2, 0).numpy()
# #         gt = gt[0].detach().cpu().permute(1, 2, 0).numpy()
# #         if aux_map is not None:
# #             aux_map = aux_map[0].detach().cpu().permute(1, 2, 0).numpy()
# #     else:
# #         img = img.detach().cpu().permute(1, 2, 0).numpy()
# #         pred = pred.detach().cpu().permute(1, 2, 0).numpy()
# #         gt = gt.detach().cpu().permute(1, 2, 0).numpy()
# #         if aux_map is not None:
# #             aux_map = aux_map.detach().cpu().permute(1, 2, 0).numpy()
# #
# #     if img.size == 0 or img.max() == 0.0:
# #         img = np.ones((256, 256, 3), dtype=np.float32) * 0.5
# #
# #     if img.max() > 1.0: img = img / img.max()
# #     img = np.clip(img, 0.0, 1.0)
# #     img_vis = np.clip(np.power(img, 1.0 / 2.2), 0, 1) * 255
# #     img_vis = img_vis.astype(np.uint8)
# #
# #     def to_heatmap(illum_map, colormap='coolwarm', force_show_variation=False):
# #         intensity = np.linalg.norm(illum_map, axis=2)
# #         intensity_min, intensity_max = intensity.min(), intensity.max()
# #         intensity_range = intensity_max - intensity_min
# #
# #         if intensity_range < 1e-6 and not force_show_variation:
# #             rgb_vis = illum_map.copy()
# #             rgb_abs_max = np.abs(rgb_vis).max()
# #             if rgb_abs_max < 1e-6:
# #                 rgb_vis = np.ones_like(illum_map) * 0.5
# #             else:
# #                 rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
# #                 rgb_vis = np.clip(rgb_vis, 0, 1)
# #             heatmap = (rgb_vis * 255).astype(np.uint8)
# #         else:
# #             if intensity_range < 0.01:
# #                 rgb_vis = illum_map.copy()
# #                 rgb_abs_max = np.abs(rgb_vis).max()
# #                 if rgb_abs_max < 1e-6:
# #                     rgb_vis = np.ones_like(illum_map) * 0.5
# #                 else:
# #                     rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
# #                     rgb_vis = np.clip(rgb_vis, 0, 1)
# #                 heatmap = (rgb_vis * 255).astype(np.uint8)
# #             else:
# #                 intensity_vis = (intensity - intensity_min) / (intensity_range + 1e-8)
# #                 cmap = plt.get_cmap(colormap)
# #                 heatmap = cmap(intensity_vis)[:, :, :3]
# #                 heatmap = (heatmap * 255).astype(np.uint8)
# #         return heatmap
# #
# #     pred_heatmap = to_heatmap(pred, 'coolwarm', force_show_variation=True)
# #     gt_heatmap = to_heatmap(gt, 'coolwarm', force_show_variation=True)
# #
# #     target_h, target_w = pred_heatmap.shape[:2]
# #     img_vis_resized = cv2.resize(img_vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
# #
# #     images_to_concat = [img_vis_resized, pred_heatmap, gt_heatmap]
# #
# #     if aux_map is not None:
# #         aux_heatmap = to_heatmap(aux_map, 'coolwarm', force_show_variation=True)
# #         aux_heatmap_resized = cv2.resize(aux_heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
# #         images_to_concat.append(aux_heatmap_resized)
# #
# #     concat = np.hstack(images_to_concat)
# #     concat_bgr = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
# #
# #     if img_count is not None:
# #         filename = f'{prefix}_{img_count:06d}.png'
# #     elif epoch is not None:
# #         filename = f'{prefix}_{epoch:03d}.png'
# #     else:
# #         filename = f'{prefix}_unknown.png'
# #
# #     try:
# #         cv2.imwrite(os.path.join(save_dir, filename), concat_bgr)
# #     except Exception as e:
# #         print(f"[WARNING] Failed to save debug image: {e}")
# #
# #
# # def evaluate(net, val_loader, device):
# #     """
# #     在验证集上评估模型
# #     [Update] 增加了截断逻辑：计算 Clean Mean (剔除最差30%) 以应对脏数据 GT
# #     """
# #     print(f"[Validation] Starting evaluation on {len(val_loader)} batches...", flush=True)
# #     net.eval()
# #     errors = []
# #
# #     with torch.no_grad():
# #         for batch_idx, batch in enumerate(val_loader):
# #             if batch_idx % 10 == 0 or batch_idx < 3:
# #                 print(f"[Validation] Processing batch {batch_idx + 1}/{len(val_loader)}...", flush=True)
# #             img = batch['image'].to(device)
# #             gt_map = batch['gt_map'].to(device)
# #             cm1 = batch['cm1'].to(device)
# #             cm2 = batch['cm2'].to(device)
# #
# #             if img.min().item() == 0.0 and img.max().item() == 0.0: continue
# #
# #             img_input = torch.clamp(img, 1e-8, 1.0)
# #             img_input = torch.pow(img_input, 1.0 / 2.2)
# #
# #             decoder_output = net(img_input, cm1, cm2)
# #
# #             if isinstance(decoder_output, dict):
# #                 pred_map = decoder_output['final']
# #             else:
# #                 pred_map = decoder_output
# #
# #             # 验证集始终使用 Angular Error
# #             err_map = angular_error_map(pred_map, gt_map)
# #
# #             gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
# #             zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
# #             err_map = err_map * (~zero_mask).float()
# #
# #             # 计算每张图的平均误差，并收集到 errors 列表中
# #             valid_pixels = torch.sum((~zero_mask).float().view(err_map.size(0), -1), dim=1) + 1e-8
# #             sum_errors = torch.sum(err_map.view(err_map.size(0), -1), dim=1)
# #             batch_errors = sum_errors / valid_pixels
# #
# #             errors.extend(batch_errors.cpu().numpy())
# #
# #     # 转为 numpy 数组
# #     errors = np.array(errors)
# #
# #     if len(errors) == 0:
# #         return {'mean_error': 0, 'median_error': 0, 'clean_mean_error': 0, 'std_error': 0, 'min_error': 0,
# #                 'max_error': 0}
# #
# #     # ====================================================
# #     # [核心修改] 计算截断后的指标 (Clean Metric)
# #     # 截断改为 30% (即保留 70%)
# #     # ====================================================
# #     # 1. 排序
# #     errors_sorted = np.sort(errors)
# #
# #     # 2. 截断：保留误差最小的 70%
# #     keep_ratio = 1.0  # [User Request] 暂停截断，改为 1.0
# #     num_keep = int(len(errors) * keep_ratio)
# #     num_keep = max(1, num_keep)  # 保护逻辑
# #
# #     clean_errors = errors_sorted[:num_keep]
# #     dirty_errors = errors_sorted[num_keep:]
# #
# #     clean_mean = np.mean(clean_errors) if len(clean_errors) > 0 else 0
# #     original_mean = np.mean(errors)
# #
# #     print(
# #         f"\n[Validation Trimmed Stat] Total: {len(errors)} | Keep Top {int(keep_ratio * 100)}%: {num_keep} | Discard Worst: {len(dirty_errors)}")
# #     if len(dirty_errors) > 0:
# #         print(f"  Dirty Samples Min Error: {dirty_errors.min():.4f}° | Max Error: {dirty_errors.max():.4f}°")
# #     print(f"  Clean Mean: {clean_mean:.4f}° vs Original Mean: {original_mean:.4f}°")
# #
# #     metrics = {
# #         'mean_error': float(original_mean),  # 原始均值 (用于参考)
# #         'clean_mean_error': float(clean_mean),  # [新] 清洗后均值 (用于 Loss 和 Scheduler)
# #         'median_error': float(np.median(errors)),  # 中位数
# #         'std_error': float(np.std(errors)),
# #         'min_error': float(np.min(errors)),
# #         'max_error': float(np.max(errors)),
# #         'percentile_25': float(np.percentile(errors, 25)),
# #         'percentile_75': float(np.percentile(errors, 75)),
# #     }
# #     return metrics
# #
# #
# # def load_checkpoint(net, checkpoint_path, optimizer=None, strict=False):
# #     if not os.path.exists(checkpoint_path):
# #         print(f"Checkpoint not found: {checkpoint_path}")
# #         return 0, float('inf')
# #
# #     print(f"Loading checkpoint from: {checkpoint_path}")
# #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
# #     model_state_dict = checkpoint.get('model_state_dict', checkpoint)
# #
# #     if isinstance(net, torch.nn.DataParallel):
# #         net.module.load_state_dict(model_state_dict, strict=False)
# #     else:
# #         net.load_state_dict(model_state_dict, strict=False)
# #
# #     if optimizer is not None and 'optimizer_state_dict' in checkpoint:
# #         try:
# #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# #         except Exception:
# #             print("  Warning: Failed to load optimizer state")
# #
# #     start_epoch = checkpoint.get('epoch', 0) + 1
# #     best_loss = checkpoint.get('loss', float('inf'))
# #     print(f"  Resuming from epoch {start_epoch}, previous loss: {best_loss:.4f}")
# #     return start_epoch, best_loss
# #
# #
# # def save_checkpoint(net, optimizer, epoch, loss, save_dir, is_best=False):
# #     try:
# #         os.makedirs(save_dir, exist_ok=True)
# #         model_state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
# #         checkpoint = {
# #             'epoch': epoch,
# #             'model_state_dict': model_state_dict,
# #             'optimizer_state_dict': optimizer.state_dict(),
# #             'loss': loss,
# #         }
# #         torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'), _use_new_zipfile_serialization=False)
# #         if is_best:
# #             torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pth'), _use_new_zipfile_serialization=False)
# #         if (epoch + 1) % 10 == 0:
# #             torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1:03d}.pth'),
# #                        _use_new_zipfile_serialization=False)
# #     except Exception as e:
# #         print(f"[ERROR] Failed to save checkpoint: {e}")
# #
# #
# # def train(args):
# #     if torch.cuda.is_available():
# #         num_gpus = torch.cuda.device_count()
# #         if args.use_multi_gpu and num_gpus > 1:
# #             device = torch.device('cuda:0')
# #             print(f"Detected {num_gpus} GPUs. Using DataParallel.")
# #         else:
# #             device = torch.device('cuda' if args.gpu_id is None else f'cuda:{args.gpu_id}')
# #     else:
# #         device = torch.device('cpu')
# #
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     exp_name = args.exp_name if args.exp_name else f"exp_{timestamp}"
# #     output_dir = os.path.join(args.output_dir, exp_name)
# #     os.makedirs(output_dir, exist_ok=True)
# #     os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
# #     os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
# #     os.makedirs(os.path.join(output_dir, 'debug_imgs'), exist_ok=True)
# #
# #     print(f"Experiment: {exp_name}")
# #     print(f"Output directory: {output_dir}")
# #
# #     net = network(input_size=64, cfe_feature_num=8, device=device).to(device)
# #     if args.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
# #         net = torch.nn.DataParallel(net)
# #
# #     optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# #         optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
# #     ) if args.use_scheduler else None
# #
# #     start_epoch = 0
# #     best_val_loss = float('inf')
# #     if args.resume:
# #         start_epoch, best_val_loss = load_checkpoint(net, args.resume, optimizer)
# #
# #     # ====================================================
# #     # [核心修改] 联合加载 LSMI 和 NUS 数据集
# #     # ====================================================
# #     print(f"[Data] Loading LSMI files from: {args.train_dir}")
# #     lsmi_files = dataset.Data.load_files(args.train_dir)
# #     print(f"[Data] Found {len(lsmi_files)} LSMI images.")
# #
# #     nus_files = []
# #     if args.nus_dir and os.path.exists(args.nus_dir):
# #         print(f"[Data] Loading NUS files from: {args.nus_dir}")
# #         nus_files = dataset.Data.load_files(args.nus_dir)
# #         print(f"[Data] Found {len(nus_files)} NUS images.")
# #     elif args.nus_dir:
# #         print(f"[Warning] NUS directory provided but not found: {args.nus_dir}")
# #
# #     # 合并两个列表
# #     all_files = lsmi_files + nus_files
# #     print(f"[Data] Total training images: {len(all_files)}")
# #
# #     # =================================================================
# #     # [新增] 严格“连坐”黑名单过滤逻辑 (Strict Filtering)
# #     # =================================================================
# #     # 定义配置文件路径
# #     blacklist_configs = {
# #         'galaxy': 'galaxy_blacklist.json',
# #         'nikon': 'nikon_blacklist.json'
# #     }
# #
# #     # 1. 预加载并解析黑名单结构
# #     # 结构: bad_sources[相机名][场景名] = {坏光源ID集合}
# #     bad_sources = {'galaxy': {}, 'nikon': {}}
# #
# #     has_blacklist = False
# #     for cam_key, json_path in blacklist_configs.items():
# #         if os.path.exists(json_path):
# #             try:
# #                 with open(json_path, 'r') as f:
# #                     raw_list = json.load(f)
# #
# #                 count = 0
# #                 for item in raw_list:
# #                     # item 格式如 "Place1012_3"
# #                     if '_' in item:
# #                         parts = item.split('_')
# #                         if len(parts) >= 2:
# #                             place_name = parts[0]
# #                             light_id = parts[1]
# #                             if place_name not in bad_sources[cam_key]:
# #                                 bad_sources[cam_key][place_name] = set()
# #                             bad_sources[cam_key][place_name].add(light_id)
# #                             count += 1
# #                 if count > 0:
# #                     print(f"[Filter] 已加载 {cam_key} 黑名单，涉及 {count} 个单光源异常样本")
# #                     has_blacklist = True
# #             except Exception as e:
# #                 print(f"[Filter] Error loading {json_path}: {e}")
# #         else:
# #             print(f"[Filter] Warning: Blacklist file {json_path} not found.")
# #
# #         # 2. 执行过滤
# #         if has_blacklist:
# #             clean_files = []
# #             filtered_count = 0
# #
# #             print("[Filter] 正在执行严格的‘连坐’过滤 (正则匹配模式)...")
# #
# #             # 定义正则：匹配 "Place" + "数字" + "_" + "1/2/3组合"
# #             # 例如匹配: "Place1012_3", "Place0_12", "Place0_1"
# #             pattern = re.compile(r'(Place\d+)_([123]+)')
# #
# #             for fpath in all_files:
# #                 fname = os.path.splitext(os.path.basename(fpath))[0]
# #                 should_drop = False
# #
# #                 # 确定当前图片属于哪个相机
# #                 current_cam = None
# #                 fpath_lower = fpath.lower()
# #                 if 'galaxy' in fpath_lower:
# #                     current_cam = 'galaxy'
# #                 elif 'nikon' in fpath_lower:
# #                     current_cam = 'nikon'
# #
# #                 # 如果是受监控的相机
# #                 if current_cam and current_cam in bad_sources:
# #                     # [Fix] 先移除 _sensorname_ 后缀干扰
# #                     clean_fname = fname.split('_sensorname_')[0] if '_sensorname_' in fname else fname
# #
# #                     # [Fix] 使用正则精准查找 "PlaceX_Y" 结构
# #                     # 即使文件名是 "galaxy_Place0_Place0_1_light1"，也能提取出 ("Place0", "1")
# #                     match = pattern.search(clean_fname)
# #
# #                     if match:
# #                         curr_place = match.group(1)  # 例如 "Place1012"
# #                         curr_lights = match.group(2)  # 例如 "3" 或 "13"
# #
# #                         # 检查该场景是否有坏记录
# #                         if curr_place in bad_sources[current_cam]:
# #                             bad_light_set = bad_sources[current_cam][curr_place]
# #
# #                             # 连坐检查：如果当前图片的光源组合里包含了坏光源
# #                             for bad_light in bad_light_set:
# #                                 if bad_light in curr_lights:
# #                                     should_drop = True
# #                                     # print(f"  [剔除] {fname} (命中坏光源 {bad_light})")
# #                                     break
# #
# #                 if should_drop:
# #                     filtered_count += 1
# #                 else:
# #                     clean_files.append(fpath)
# #
# #             print(f"[Filter] 过滤完成！共剔除 {filtered_count} 张受污染图片。")
# #             print(f"[Filter] 剩余可用图片: {len(clean_files)}")
# #
# #             # 更新列表
# #             all_files = clean_files
# #     # =================================================================
# #
# #     # 随机打乱 (这非常重要！否则验证集切分时可能只切到其中一个数据集)
# #     random.shuffle(all_files)
# #     # ====================================================
# #
# #     val_files = []
# #     train_files = all_files  # 使用合并后的列表
# #
# #     if args.val_ratio > 0:
# #         val_size = int(len(train_files) * args.val_ratio)
# #         val_files = train_files[:val_size]
# #         train_files = train_files[val_size:]
# #
# #         print(f"[Data] Split: Train={len(train_files)}, Val={len(val_files)}")
# #
# #         val_ds = dataset.Data(val_files, mode='testing', input_size=256)
# #         val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
# #     else:
# #         val_loader = None
# #
# #     train_ds = dataset.Data(train_files, input_size=256)
# #     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
# #                               pin_memory=True)
# #
# #     # Save config
# #     with open(os.path.join(output_dir, 'config.json'), 'w') as f:
# #         json.dump(vars(args), f, indent=4)
# #
# #     train_log = defaultdict(list)
# #
# #     print("Starting training with Trimmed Angular Error Loss (Train & Val Scheme)...")
# #     for epoch in range(start_epoch, args.epochs):
# #         net.train()
# #         total_loss = 0
# #         valid_batches = 0
# #
# #         last_batch_data = None
# #
# #         for i, batch in enumerate(train_loader):
# #             if i % 10 == 0:
# #                 print(f"[Epoch {epoch + 1}] Processing batch {i + 1}/{len(train_loader)}...", end='\r', flush=True)
# #
# #             img = batch['image'].to(device)
# #             gt_map = batch['gt_map'].to(device)
# #             cm1 = batch['cm1'].to(device)
# #             cm2 = batch['cm2'].to(device)
# #
# #             img_input = torch.clamp(img, 1e-8, 1.0)
# #             img_input = torch.pow(img_input, 1.0 / 2.2)
# #
# #             optimizer.zero_grad()
# #             decoder_output = net(img_input, cm1, cm2)
# #
# #             if isinstance(decoder_output, dict):
# #                 pred_map = decoder_output['final']
# #                 aux_map = decoder_output['aux']
# #             else:
# #                 pred_map = decoder_output
# #                 aux_map = None
# #
# #             # --- Loss Calculation ---
# #             err_map = angular_error_map(pred_map, gt_map)
# #             gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
# #             zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
# #             mask = (~zero_mask).float()
# #             err_map = err_map * mask
# #
# #             # Trimmed Loss (Keep 70%)
# #             batch_errors = torch.mean(err_map.view(err_map.size(0), -1), dim=1)
# #             keep_ratio = 0.8  # 保持训练时的截断，以防万一还有漏网之鱼
# #             current_batch_size = img.size(0)
# #             num_keep = int(current_batch_size * keep_ratio)
# #             num_keep = max(1, num_keep)
# #             good_errors, _ = torch.topk(batch_errors, k=num_keep, largest=False)
# #             loss = torch.mean(good_errors)
# #
# #             if epoch == 0 and i == 0:
# #                 print(f"\n  [Loss Strategy] Trimmed Angular Loss (Keeping {num_keep}/{current_batch_size})")
# #
# #             # Aux Loss
# #             aux_loss = 0.0
# #             if aux_map is not None:
# #                 gt_downsampled = nn.functional.interpolate(
# #                     gt_map, size=aux_map.shape[2:], mode='bilinear', align_corners=True
# #                 )
# #                 gt_ds_norm = torch.norm(gt_downsampled, dim=1, keepdim=False)
# #                 zero_mask_ds = torch.isclose(gt_ds_norm, torch.ones_like(gt_ds_norm) * 1.732, atol=1e-3)
# #                 mask_ds = (~zero_mask_ds).float()
# #                 aux_err = angular_error_map(aux_map, gt_downsampled)
# #                 aux_err = aux_err * mask_ds
# #                 aux_loss = torch.mean(aux_err)
# #
# #             tv_loss = total_variation_loss(pred_map)
# #             total_loss_value = loss + args.tv_weight * tv_loss + args.aux_weight * aux_loss
# #
# #             if torch.isnan(total_loss_value):
# #                 print(f"Warning: NaN loss detected. Skipping batch.")
# #                 continue
# #
# #             total_loss_value.backward()
# #             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
# #             optimizer.step()
# #
# #             total_loss += total_loss_value.item()
# #             valid_batches += 1
# #
# #             # --- Epoch End Logic ---
# #             if i == len(train_loader) - 1:
# #                 last_batch_data = (torch.clamp(img, 0, 1), pred_map, gt_map)
# #
# #                 if valid_batches > 0:
# #                     avg_loss = total_loss / valid_batches
# #                     train_log['train_loss'].append(avg_loss)
# #
# #                     # Validation
# #                     val_loss = 0
# #                     val_clean_mean = 0
# #                     is_best = False
# #                     metrics = {'mean_error': 0, 'median_error': 0}
# #
# #                     if val_loader:
# #                         print(f"\n[Epoch {epoch + 1}] Starting Validation...")
# #                         metrics = evaluate(net, val_loader, device)
# #                         val_clean_mean = metrics['clean_mean_error']
# #                         val_loss = val_clean_mean
# #
# #                         train_log['val_loss'].append(val_loss)
# #                         train_log['val_mean_error'].append(metrics['mean_error'])
# #                         train_log['val_median_error'].append(metrics['median_error'])
# #
# #                         if val_loss < best_val_loss:
# #                             is_best = True
# #                             best_val_loss = val_loss
# #
# #                         if scheduler: scheduler.step(val_loss)
# #
# #                     save_checkpoint(net, optimizer, epoch, avg_loss, os.path.join(output_dir, 'checkpoints'),
# #                                     is_best=is_best)
# #
# #                     # =========================================================
# #                     # [修复点] 将日志打印和保存全部缩进到 if 块内部
# #                     # 只有算出了 avg_loss 才能打印
# #                     # =========================================================
# #                     log_msg = f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f} | "
# #                     if val_loader:
# #                         log_msg += f"Val Clean Mean: {val_clean_mean:.4f} | Val Orig Mean: {metrics['mean_error']:.4f} | Val Median: {metrics['median_error']:.4f} | "
# #                     log_msg += f"LR: {optimizer.param_groups[0]['lr']:.6f}"
# #                     print(log_msg)
# #
# #                     # Vis
# #                     if last_batch_data:
# #                         save_debug_images(last_batch_data[0], last_batch_data[1], last_batch_data[2],
# #                                           epoch=epoch + 1, save_dir=os.path.join(output_dir, 'debug_imgs'))
# #
# #                     # Save Logs
# #                     with open(os.path.join(output_dir, 'logs', 'training_log.json'), 'w') as f:
# #                         json.dump(train_log, f, indent=4)
# #
# #     print("Training completed!")
# #
# #
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description="Train Dense-CCMNet (Trimmed Angular Loss)")
# #     parser.add_argument('--train_dir', type=str, default='/mnt/sda2/SMY/ccmnet/original_resized/LSMI')
# #     # [新增] NUS 数据集路径参数
# #     parser.add_argument('--nus_dir', type=str, default=None, help='Path to NUS dataset for joint training')
# #
# #     parser.add_argument('--val_ratio', type=float, default=0.1)
# #     parser.add_argument('--config', type=str, default=None)
# #     parser.add_argument('--gpu_id', type=int, default=None)
# #     parser.add_argument('--use_multi_gpu', action='store_true')
# #     parser.add_argument('--epochs', type=int, default=100)
# #     parser.add_argument('--batch_size', type=int, default=16)  # 推荐 16
# #     parser.add_argument('--lr', type=float, default=2e-4)
# #     parser.add_argument('--weight_decay', type=float, default=0.0001)
# #     parser.add_argument('--use_scheduler', action='store_true', default=True)
# #     parser.add_argument('--output_dir', type=str, default='./experiments')
# #     parser.add_argument('--exp_name', type=str, default=None)
# #     parser.add_argument('--log_interval', type=int, default=5)
# #     parser.add_argument('--use_combined_loss', action='store_true')
# #     parser.add_argument('--loss_alpha', type=float, default=0.7)
# #     parser.add_argument('--tv_weight', type=float, default=0.01)
# #     parser.add_argument('--aux_weight', type=float, default=0.4)
# #     parser.add_argument('--resume', type=str, default=None)
# #
# #     args = parser.parse_args()
# #
# #     if args.config and os.path.exists(args.config):
# #         with open(args.config, 'r') as f:
# #             config = json.load(f)
# #         for key, value in config.items():
# #             if hasattr(args, key) and key not in ['config', 'description']:
# #                 setattr(args, key, value)
# #
# #     try:
# #         train(args)
# #     except Exception as e:
# #         import traceback
# #
# #         traceback.print_exc()
#
#
# # import sys
# # import os
# # import re
# # import cv2
# # import numpy as np
# # import json
# # import time
# # import random
# # from datetime import datetime
# # from collections import defaultdict
# #
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # 获取项目根目录 (.../CCMNet-main)
# # project_root = os.path.dirname(current_dir)
# # # 将根目录加入 Python 搜索路径
# # if project_root not in sys.path:
# #     sys.path.append(project_root)
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from src.ccmnet.ccmnet import network
# # from src import dataset
# # import math
# # import argparse
# # import matplotlib
# #
# # matplotlib.use('Agg')  # 使用非交互式后端
# # import matplotlib.pyplot as plt
# #
# #
# # def angular_error_map(pred, gt):
# #     """ 计算像素级角误差 (Robust Version) - 强制归一化版本 """
# #     pred = nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
# #     gt = nn.functional.normalize(gt, p=2, dim=1, eps=1e-8)
# #     dot = torch.sum(pred * gt, dim=1)
# #     dot = torch.clamp(dot, -0.999999, 0.999999)
# #     angle = torch.acos(dot) * (180.0 / math.pi)
# #     angle = torch.clamp(angle, 0.0, 180.0)
# #     return angle
# #
# #
# # def total_variation_loss(pred_map):
# #     """ Total Variation (TV) Loss """
# #     batch_size = pred_map.size(0)
# #     diff_x = pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]
# #     tv_x = torch.abs(diff_x).sum()
# #     diff_y = pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]
# #     tv_y = torch.abs(diff_y).sum()
# #     H, W = pred_map.size(2), pred_map.size(3)
# #     tv_loss = (tv_x + tv_y) / (batch_size * H * W)
# #     return tv_loss
# #
# #
# # def save_debug_images(img, pred, gt, aux_map=None, epoch=None, img_count=None, save_dir='debug_imgs', prefix='epoch'):
# #     """ 保存可视化结果 """
# #     os.makedirs(save_dir, exist_ok=True)
# #
# #     if img.dim() == 4:
# #         img = img[0].detach().cpu().permute(1, 2, 0).numpy()
# #         pred = pred[0].detach().cpu().permute(1, 2, 0).numpy()
# #         gt = gt[0].detach().cpu().permute(1, 2, 0).numpy()
# #         if aux_map is not None:
# #             aux_map = aux_map[0].detach().cpu().permute(1, 2, 0).numpy()
# #     else:
# #         img = img.detach().cpu().permute(1, 2, 0).numpy()
# #         pred = pred.detach().cpu().permute(1, 2, 0).numpy()
# #         gt = gt.detach().cpu().permute(1, 2, 0).numpy()
# #         if aux_map is not None:
# #             aux_map = aux_map.detach().cpu().permute(1, 2, 0).numpy()
# #
# #     if img.size == 0 or img.max() == 0.0:
# #         img = np.ones((256, 256, 3), dtype=np.float32) * 0.5
# #
# #     if img.max() > 1.0: img = img / img.max()
# #     img = np.clip(img, 0.0, 1.0)
# #     img_vis = np.clip(np.power(img, 1.0 / 2.2), 0, 1) * 255
# #     img_vis = img_vis.astype(np.uint8)
# #
# #     def to_heatmap(illum_map, colormap='coolwarm', force_show_variation=False):
# #         intensity = np.linalg.norm(illum_map, axis=2)
# #         intensity_min, intensity_max = intensity.min(), intensity.max()
# #         intensity_range = intensity_max - intensity_min
# #
# #         if intensity_range < 1e-6 and not force_show_variation:
# #             rgb_vis = illum_map.copy()
# #             rgb_abs_max = np.abs(rgb_vis).max()
# #             if rgb_abs_max < 1e-6:
# #                 rgb_vis = np.ones_like(illum_map) * 0.5
# #             else:
# #                 rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
# #                 rgb_vis = np.clip(rgb_vis, 0, 1)
# #             heatmap = (rgb_vis * 255).astype(np.uint8)
# #         else:
# #             if intensity_range < 0.01:
# #                 rgb_vis = illum_map.copy()
# #                 rgb_abs_max = np.abs(rgb_vis).max()
# #                 if rgb_abs_max < 1e-6:
# #                     rgb_vis = np.ones_like(illum_map) * 0.5
# #                 else:
# #                     rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
# #                     rgb_vis = np.clip(rgb_vis, 0, 1)
# #                 heatmap = (rgb_vis * 255).astype(np.uint8)
# #             else:
# #                 intensity_vis = (intensity - intensity_min) / (intensity_range + 1e-8)
# #                 cmap = plt.get_cmap(colormap)
# #                 heatmap = cmap(intensity_vis)[:, :, :3]
# #                 heatmap = (heatmap * 255).astype(np.uint8)
# #         return heatmap
# #
# #     pred_heatmap = to_heatmap(pred, 'coolwarm', force_show_variation=True)
# #     gt_heatmap = to_heatmap(gt, 'coolwarm', force_show_variation=True)
# #
# #     target_h, target_w = pred_heatmap.shape[:2]
# #     img_vis_resized = cv2.resize(img_vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
# #
# #     images_to_concat = [img_vis_resized, pred_heatmap, gt_heatmap]
# #
# #     if aux_map is not None:
# #         aux_heatmap = to_heatmap(aux_map, 'coolwarm', force_show_variation=True)
# #         aux_heatmap_resized = cv2.resize(aux_heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
# #         images_to_concat.append(aux_heatmap_resized)
# #
# #     concat = np.hstack(images_to_concat)
# #     concat_bgr = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
# #
# #     if img_count is not None:
# #         filename = f'{prefix}_{img_count:06d}.png'
# #     elif epoch is not None:
# #         filename = f'{prefix}_{epoch:03d}.png'
# #     else:
# #         filename = f'{prefix}_unknown.png'
# #
# #     try:
# #         cv2.imwrite(os.path.join(save_dir, filename), concat_bgr)
# #     except Exception as e:
# #         print(f"[WARNING] Failed to save debug image: {e}")
# #
# #
# # def evaluate(net, val_loader, device):
# #     """在验证集上评估模型"""
# #     print(f"[Validation] Starting evaluation on {len(val_loader)} batches...", flush=True)
# #     net.eval()
# #     errors = []
# #
# #     with torch.no_grad():
# #         for batch_idx, batch in enumerate(val_loader):
# #             if batch_idx % 10 == 0 or batch_idx < 3:
# #                 print(f"[Validation] Processing batch {batch_idx + 1}/{len(val_loader)}...", flush=True)
# #             img = batch['image'].to(device)
# #             gt_map = batch['gt_map'].to(device)
# #             cm1 = batch['cm1'].to(device)
# #             cm2 = batch['cm2'].to(device)
# #
# #             if img.min().item() == 0.0 and img.max().item() == 0.0: continue
# #
# #             img_input = torch.clamp(img, 1e-8, 1.0)
# #             img_input = torch.pow(img_input, 1.0 / 2.2)
# #
# #             decoder_output = net(img_input, cm1, cm2)
# #
# #             if isinstance(decoder_output, dict):
# #                 pred_map = decoder_output['final']
# #             else:
# #                 pred_map = decoder_output
# #
# #             err_map = angular_error_map(pred_map, gt_map)
# #             gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
# #             zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
# #             err_map = err_map * (~zero_mask).float()
# #
# #             valid_pixels = torch.sum((~zero_mask).float().view(err_map.size(0), -1), dim=1) + 1e-8
# #             sum_errors = torch.sum(err_map.view(err_map.size(0), -1), dim=1)
# #             batch_errors = sum_errors / valid_pixels
# #
# #             errors.extend(batch_errors.cpu().numpy())
# #
# #     errors = np.array(errors)
# #
# #     if len(errors) == 0:
# #         return {'mean_error': 0, 'median_error': 0, 'clean_mean_error': 0, 'std_error': 0, 'min_error': 0,
# #                 'max_error': 0}
# #
# #     errors_sorted = np.sort(errors)
# #     keep_ratio = 1.0
# #     num_keep = int(len(errors) * keep_ratio)
# #     num_keep = max(1, num_keep)
# #
# #     clean_errors = errors_sorted[:num_keep]
# #     dirty_errors = errors_sorted[num_keep:]
# #
# #     clean_mean = np.mean(clean_errors) if len(clean_errors) > 0 else 0
# #     original_mean = np.mean(errors)
# #
# #     print(f"\n[Validation Trimmed Stat] Total: {len(errors)} | Keep Top {int(keep_ratio * 100)}%: {num_keep}")
# #     print(f"  Clean Mean: {clean_mean:.4f}° vs Original Mean: {original_mean:.4f}°")
# #
# #     metrics = {
# #         'mean_error': float(original_mean),
# #         'clean_mean_error': float(clean_mean),
# #         'median_error': float(np.median(errors)),
# #         'std_error': float(np.std(errors)),
# #         'min_error': float(np.min(errors)),
# #         'max_error': float(np.max(errors)),
# #         'percentile_25': float(np.percentile(errors, 25)),
# #         'percentile_75': float(np.percentile(errors, 75)),
# #     }
# #     return metrics
# #
# #
# # def load_checkpoint(net, checkpoint_path, optimizer=None, strict=False):
# #     if not os.path.exists(checkpoint_path):
# #         print(f"Checkpoint not found: {checkpoint_path}")
# #         return 0, float('inf')
# #
# #     print(f"Loading checkpoint from: {checkpoint_path}")
# #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
# #     model_state_dict = checkpoint.get('model_state_dict', checkpoint)
# #
# #     if isinstance(net, torch.nn.DataParallel):
# #         net.module.load_state_dict(model_state_dict, strict=False)
# #     else:
# #         net.load_state_dict(model_state_dict, strict=False)
# #
# #     if optimizer is not None and 'optimizer_state_dict' in checkpoint:
# #         try:
# #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# #         except Exception:
# #             print("  Warning: Failed to load optimizer state")
# #
# #     start_epoch = checkpoint.get('epoch', 0) + 1
# #     best_loss = checkpoint.get('loss', float('inf'))
# #     print(f"  Resuming from epoch {start_epoch}, previous loss: {best_loss:.4f}")
# #     return start_epoch, best_loss
# #
# #
# # def save_checkpoint(net, optimizer, epoch, loss, save_dir, is_best=False):
# #     try:
# #         os.makedirs(save_dir, exist_ok=True)
# #         model_state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
# #         checkpoint = {
# #             'epoch': epoch,
# #             'model_state_dict': model_state_dict,
# #             'optimizer_state_dict': optimizer.state_dict(),
# #             'loss': loss,
# #         }
# #         torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'), _use_new_zipfile_serialization=False)
# #         if is_best:
# #             torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pth'), _use_new_zipfile_serialization=False)
# #         if (epoch + 1) % 10 == 0:
# #             torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1:03d}.pth'),
# #                        _use_new_zipfile_serialization=False)
# #     except Exception as e:
# #         print(f"[ERROR] Failed to save checkpoint: {e}")
# #
# #
# # def train(args):
# #     if torch.cuda.is_available():
# #         num_gpus = torch.cuda.device_count()
# #         if args.use_multi_gpu and num_gpus > 1:
# #             device = torch.device('cuda:0')
# #             print(f"Detected {num_gpus} GPUs. Using DataParallel.")
# #         else:
# #             device = torch.device('cuda' if args.gpu_id is None else f'cuda:{args.gpu_id}')
# #     else:
# #         device = torch.device('cpu')
# #
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     exp_name = args.exp_name if args.exp_name else f"exp_{timestamp}"
# #     output_dir = os.path.join(args.output_dir, exp_name)
# #     os.makedirs(output_dir, exist_ok=True)
# #     os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
# #     os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
# #     os.makedirs(os.path.join(output_dir, 'debug_imgs'), exist_ok=True)
# #
# #     print(f"Experiment: {exp_name}")
# #     print(f"Output directory: {output_dir}")
# #
# #     net = network(input_size=64, cfe_feature_num=8, device=device).to(device)
# #     if args.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
# #         net = torch.nn.DataParallel(net)
# #
# #     optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# #         optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
# #     ) if args.use_scheduler else None
# #
# #     start_epoch = 0
# #     best_val_loss = float('inf')
# #     if args.resume:
# #         start_epoch, best_val_loss = load_checkpoint(net, args.resume, optimizer)
# #
# #     # ====================================================
# #     # [核心修改] 加载数据 & 过滤 Fujifilm
# #     # ====================================================
# #     print(f"[Data] Loading LSMI files from: {args.train_dir}")
# #     lsmi_files = dataset.Data.load_files(args.train_dir)
# #     print(f"[Data] Found {len(lsmi_files)} LSMI images.")
# #
# #     nus_files = []
# #     if args.nus_dir and os.path.exists(args.nus_dir):
# #         print(f"[Data] Loading NUS files from: {args.nus_dir}")
# #         raw_nus_files = dataset.Data.load_files(args.nus_dir)
# #
# #         # [新增] 过滤 Fujifilm 逻辑
# #         print("[Filter] Checking for Fujifilm images (unmasked color checker)...")
# #         fuji_dropped_count = 0
# #         for f in raw_nus_files:
# #             # NUS路径通常包含相机名，或者文件名中包含
# #             # 检查 "fujifilm" 是否在路径中 (不区分大小写)
# #             if 'fujifilm' in f.lower():
# #                 fuji_dropped_count += 1
# #             else:
# #                 nus_files.append(f)
# #
# #         print(f"[Filter] Dropped {fuji_dropped_count} Fujifilm images.")
# #         print(f"[Data] Remaining valid NUS images: {len(nus_files)}")
# #
# #     elif args.nus_dir:
# #         print(f"[Warning] NUS directory provided but not found: {args.nus_dir}")
# #
# #     # 合并两个列表
# #     all_files = lsmi_files + nus_files
# #     print(f"[Data] Total training images: {len(all_files)}")
# #
# #     # =================================================================
# #     # 黑名单过滤逻辑 (Galaxy/Nikon)
# #     # =================================================================
# #     blacklist_configs = {
# #         'galaxy': 'galaxy_blacklist.json',
# #         'nikon': 'nikon_blacklist.json'
# #     }
# #
# #     bad_sources = {'galaxy': {}, 'nikon': {}}
# #     has_blacklist = False
# #     for cam_key, json_path in blacklist_configs.items():
# #         if os.path.exists(json_path):
# #             try:
# #                 with open(json_path, 'r') as f:
# #                     raw_list = json.load(f)
# #                 count = 0
# #                 for item in raw_list:
# #                     if '_' in item:
# #                         parts = item.split('_')
# #                         if len(parts) >= 2:
# #                             place_name = parts[0]
# #                             light_id = parts[1]
# #                             if place_name not in bad_sources[cam_key]:
# #                                 bad_sources[cam_key][place_name] = set()
# #                             bad_sources[cam_key][place_name].add(light_id)
# #                             count += 1
# #                 if count > 0:
# #                     print(f"[Filter] 已加载 {cam_key} 黑名单，涉及 {count} 个单光源异常样本")
# #                     has_blacklist = True
# #             except Exception as e:
# #                 print(f"[Filter] Error loading {json_path}: {e}")
# #
# #     # 执行 Galaxy/Nikon 过滤
# #     if has_blacklist:
# #         clean_files = []
# #         filtered_count = 0
# #         pattern = re.compile(r'(Place\d+)_([123]+)')
# #
# #         for fpath in all_files:
# #             fname = os.path.splitext(os.path.basename(fpath))[0]
# #             should_drop = False
# #             current_cam = None
# #             fpath_lower = fpath.lower()
# #             if 'galaxy' in fpath_lower:
# #                 current_cam = 'galaxy'
# #             elif 'nikon' in fpath_lower:
# #                 current_cam = 'nikon'
# #
# #             if current_cam and current_cam in bad_sources:
# #                 clean_fname = fname.split('_sensorname_')[0] if '_sensorname_' in fname else fname
# #                 match = pattern.search(clean_fname)
# #                 if match:
# #                     curr_place = match.group(1)
# #                     curr_lights = match.group(2)
# #                     if curr_place in bad_sources[current_cam]:
# #                         bad_light_set = bad_sources[current_cam][curr_place]
# #                         for bad_light in bad_light_set:
# #                             if bad_light in curr_lights:
# #                                 should_drop = True
# #                                 break
# #             if should_drop:
# #                 filtered_count += 1
# #             else:
# #                 clean_files.append(fpath)
# #
# #         print(f"[Filter] 过滤完成！共剔除 {filtered_count} 张 Galaxy/Nikon 受污染图片。")
# #         all_files = clean_files
# #         print(f"[Data] Final training images count: {len(all_files)}")
# #     # =================================================================
# #
# #     random.shuffle(all_files)
# #
# #     val_files = []
# #     train_files = all_files
# #
# #     if args.val_ratio > 0:
# #         val_size = int(len(train_files) * args.val_ratio)
# #         val_files = train_files[:val_size]
# #         train_files = train_files[val_size:]
# #
# #         print(f"[Data] Split: Train={len(train_files)}, Val={len(val_files)}")
# #         val_ds = dataset.Data(val_files, mode='testing', input_size=256)
# #         val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
# #     else:
# #         val_loader = None
# #
# #     train_ds = dataset.Data(train_files, input_size=256)
# #     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
# #                               pin_memory=True)
# #
# #     with open(os.path.join(output_dir, 'config.json'), 'w') as f:
# #         json.dump(vars(args), f, indent=4)
# #
# #     train_log = defaultdict(list)
# #
# #     print("Starting training with Trimmed Angular Error Loss (Train & Val Scheme)...")
# #     for epoch in range(start_epoch, args.epochs):
# #         net.train()
# #         total_loss = 0
# #         valid_batches = 0
# #         last_batch_data = None
# #
# #         for i, batch in enumerate(train_loader):
# #             if i % 10 == 0:
# #                 print(f"[Epoch {epoch + 1}] Processing batch {i + 1}/{len(train_loader)}...", end='\r', flush=True)
# #
# #             img = batch['image'].to(device)
# #             gt_map = batch['gt_map'].to(device)
# #             cm1 = batch['cm1'].to(device)
# #             cm2 = batch['cm2'].to(device)
# #
# #             img_input = torch.clamp(img, 1e-8, 1.0)
# #             img_input = torch.pow(img_input, 1.0 / 2.2)
# #
# #             optimizer.zero_grad()
# #             decoder_output = net(img_input, cm1, cm2)
# #
# #             if isinstance(decoder_output, dict):
# #                 pred_map = decoder_output['final']
# #                 aux_map = decoder_output['aux']
# #             else:
# #                 pred_map = decoder_output
# #                 aux_map = None
# #
# #             err_map = angular_error_map(pred_map, gt_map)
# #             gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
# #             zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
# #             mask = (~zero_mask).float()
# #             err_map = err_map * mask
# #
# #             batch_errors = torch.mean(err_map.view(err_map.size(0), -1), dim=1)
# #             keep_ratio = 0.8
# #             current_batch_size = img.size(0)
# #             num_keep = int(current_batch_size * keep_ratio)
# #             num_keep = max(1, num_keep)
# #             good_errors, _ = torch.topk(batch_errors, k=num_keep, largest=False)
# #             loss = torch.mean(good_errors)
# #
# #             if epoch == 0 and i == 0:
# #                 print(f"\n  [Loss Strategy] Trimmed Angular Loss (Keeping {num_keep}/{current_batch_size})")
# #
# #             aux_loss = 0.0
# #             if aux_map is not None:
# #                 gt_downsampled = nn.functional.interpolate(
# #                     gt_map, size=aux_map.shape[2:], mode='bilinear', align_corners=True
# #                 )
# #                 gt_ds_norm = torch.norm(gt_downsampled, dim=1, keepdim=False)
# #                 zero_mask_ds = torch.isclose(gt_ds_norm, torch.ones_like(gt_ds_norm) * 1.732, atol=1e-3)
# #                 mask_ds = (~zero_mask_ds).float()
# #                 aux_err = angular_error_map(aux_map, gt_downsampled)
# #                 aux_err = aux_err * mask_ds
# #                 aux_loss = torch.mean(aux_err)
# #
# #             tv_loss = total_variation_loss(pred_map)
# #             total_loss_value = loss + args.tv_weight * tv_loss + args.aux_weight * aux_loss
# #
# #             if torch.isnan(total_loss_value):
# #                 print(f"Warning: NaN loss detected. Skipping batch.")
# #                 continue
# #
# #             total_loss_value.backward()
# #             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
# #             optimizer.step()
# #
# #             total_loss += total_loss_value.item()
# #             valid_batches += 1
# #
# #             if i == len(train_loader) - 1:
# #                 last_batch_data = (torch.clamp(img, 0, 1), pred_map, gt_map)
# #
# #                 if valid_batches > 0:
# #                     avg_loss = total_loss / valid_batches
# #                     train_log['train_loss'].append(avg_loss)
# #
# #                     val_loss = 0
# #                     val_clean_mean = 0
# #                     is_best = False
# #                     metrics = {'mean_error': 0, 'median_error': 0}
# #
# #                     if val_loader:
# #                         print(f"\n[Epoch {epoch + 1}] Starting Validation...")
# #                         metrics = evaluate(net, val_loader, device)
# #                         val_clean_mean = metrics['clean_mean_error']
# #                         val_loss = val_clean_mean
# #
# #                         train_log['val_loss'].append(val_loss)
# #                         train_log['val_mean_error'].append(metrics['mean_error'])
# #                         train_log['val_median_error'].append(metrics['median_error'])
# #
# #                         if val_loss < best_val_loss:
# #                             is_best = True
# #                             best_val_loss = val_loss
# #
# #                         if scheduler: scheduler.step(val_loss)
# #
# #                     save_checkpoint(net, optimizer, epoch, avg_loss, os.path.join(output_dir, 'checkpoints'),
# #                                     is_best=is_best)
# #
# #                     log_msg = f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f} | "
# #                     if val_loader:
# #                         log_msg += f"Val Clean Mean: {val_clean_mean:.4f} | Val Orig Mean: {metrics['mean_error']:.4f} | Val Median: {metrics['median_error']:.4f} | "
# #                     log_msg += f"LR: {optimizer.param_groups[0]['lr']:.6f}"
# #                     print(log_msg)
# #
# #                     if last_batch_data:
# #                         save_debug_images(last_batch_data[0], last_batch_data[1], last_batch_data[2],
# #                                           epoch=epoch + 1, save_dir=os.path.join(output_dir, 'debug_imgs'))
# #
# #                     with open(os.path.join(output_dir, 'logs', 'training_log.json'), 'w') as f:
# #                         json.dump(train_log, f, indent=4)
# #
# #     print("Training completed!")
# #
# #
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description="Train Dense-CCMNet (Trimmed Angular Loss)")
# #     parser.add_argument('--train_dir', type=str, default='/mnt/sda2/SMY/ccmnet/original_resized/LSMI')
# #     parser.add_argument('--nus_dir', type=str, default=None, help='Path to NUS dataset for joint training')
# #     parser.add_argument('--val_ratio', type=float, default=0.1)
# #     parser.add_argument('--config', type=str, default=None)
# #     parser.add_argument('--gpu_id', type=int, default=None)
# #     parser.add_argument('--use_multi_gpu', action='store_true')
# #     parser.add_argument('--epochs', type=int, default=100)
# #     parser.add_argument('--batch_size', type=int, default=16)
# #     parser.add_argument('--lr', type=float, default=2e-4)
# #     parser.add_argument('--weight_decay', type=float, default=0.0001)
# #     parser.add_argument('--use_scheduler', action='store_true', default=True)
# #     parser.add_argument('--output_dir', type=str, default='./experiments')
# #     parser.add_argument('--exp_name', type=str, default=None)
# #     parser.add_argument('--log_interval', type=int, default=5)
# #     parser.add_argument('--use_combined_loss', action='store_true')
# #     parser.add_argument('--loss_alpha', type=float, default=0.7)
# #     parser.add_argument('--tv_weight', type=float, default=0.01)
# #     parser.add_argument('--aux_weight', type=float, default=0.4)
# #     parser.add_argument('--resume', type=str, default=None)
# #
# #     args = parser.parse_args()
# #
# #     if args.config and os.path.exists(args.config):
# #         with open(args.config, 'r') as f:
# #             config = json.load(f)
# #         for key, value in config.items():
# #             if hasattr(args, key) and key not in ['config', 'description']:
# #                 setattr(args, key, value)
# #
# #     try:
# #         train(args)
# #     except Exception as e:
# #         import traceback
# #
# #         traceback.print_exc()
#
#
# import sys
# import os
# import re
# import cv2
# import numpy as np
# import json
# import time
# import random
# from datetime import datetime
# from collections import defaultdict
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录 (.../CCMNet-main)
# project_root = os.path.dirname(current_dir)
# # 将根目录加入 Python 搜索路径
# if project_root not in sys.path:
#     sys.path.append(project_root)
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from src.ccmnet.ccmnet import network
# from src import dataset
# import math
# import argparse
# import matplotlib
#
# matplotlib.use('Agg')  # 使用非交互式后端
# import matplotlib.pyplot as plt
#
#
# def angular_error_map(pred, gt):
#     """ 计算像素级角误差 (Robust Version) - 强制归一化版本 """
#     pred = nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
#     gt = nn.functional.normalize(gt, p=2, dim=1, eps=1e-8)
#     dot = torch.sum(pred * gt, dim=1)
#     dot = torch.clamp(dot, -0.999999, 0.999999)
#     angle = torch.acos(dot) * (180.0 / math.pi)
#     angle = torch.clamp(angle, 0.0, 180.0)
#     return angle
#
#
# def total_variation_loss(pred_map):
#     """ Total Variation (TV) Loss """
#     batch_size = pred_map.size(0)
#     diff_x = pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]
#     tv_x = torch.abs(diff_x).sum()
#     diff_y = pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]
#     tv_y = torch.abs(diff_y).sum()
#     H, W = pred_map.size(2), pred_map.size(3)
#     tv_loss = (tv_x + tv_y) / (batch_size * H * W)
#     return tv_loss
#
#
# def save_debug_images(img, pred, gt, aux_map=None, epoch=None, img_count=None, save_dir='debug_imgs', prefix='epoch'):
#     """ 保存可视化结果 """
#     os.makedirs(save_dir, exist_ok=True)
#
#     if img.dim() == 4:
#         img = img[0].detach().cpu().permute(1, 2, 0).numpy()
#         pred = pred[0].detach().cpu().permute(1, 2, 0).numpy()
#         gt = gt[0].detach().cpu().permute(1, 2, 0).numpy()
#         if aux_map is not None:
#             aux_map = aux_map[0].detach().cpu().permute(1, 2, 0).numpy()
#     else:
#         img = img.detach().cpu().permute(1, 2, 0).numpy()
#         pred = pred.detach().cpu().permute(1, 2, 0).numpy()
#         gt = gt.detach().cpu().permute(1, 2, 0).numpy()
#         if aux_map is not None:
#             aux_map = aux_map.detach().cpu().permute(1, 2, 0).numpy()
#
#     if img.size == 0 or img.max() == 0.0:
#         img = np.ones((256, 256, 3), dtype=np.float32) * 0.5
#
#     if img.max() > 1.0: img = img / img.max()
#     img = np.clip(img, 0.0, 1.0)
#     img_vis = np.clip(np.power(img, 1.0 / 2.2), 0, 1) * 255
#     img_vis = img_vis.astype(np.uint8)
#
#     def to_heatmap(illum_map, colormap='coolwarm', force_show_variation=False):
#         intensity = np.linalg.norm(illum_map, axis=2)
#         intensity_min, intensity_max = intensity.min(), intensity.max()
#         intensity_range = intensity_max - intensity_min
#
#         if intensity_range < 1e-6 and not force_show_variation:
#             rgb_vis = illum_map.copy()
#             rgb_abs_max = np.abs(rgb_vis).max()
#             if rgb_abs_max < 1e-6:
#                 rgb_vis = np.ones_like(illum_map) * 0.5
#             else:
#                 rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
#                 rgb_vis = np.clip(rgb_vis, 0, 1)
#             heatmap = (rgb_vis * 255).astype(np.uint8)
#         else:
#             if intensity_range < 0.01:
#                 rgb_vis = illum_map.copy()
#                 rgb_abs_max = np.abs(rgb_vis).max()
#                 if rgb_abs_max < 1e-6:
#                     rgb_vis = np.ones_like(illum_map) * 0.5
#                 else:
#                     rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
#                     rgb_vis = np.clip(rgb_vis, 0, 1)
#                 heatmap = (rgb_vis * 255).astype(np.uint8)
#             else:
#                 intensity_vis = (intensity - intensity_min) / (intensity_range + 1e-8)
#                 cmap = plt.get_cmap(colormap)
#                 heatmap = cmap(intensity_vis)[:, :, :3]
#                 heatmap = (heatmap * 255).astype(np.uint8)
#         return heatmap
#
#     pred_heatmap = to_heatmap(pred, 'coolwarm', force_show_variation=True)
#     gt_heatmap = to_heatmap(gt, 'coolwarm', force_show_variation=True)
#
#     target_h, target_w = pred_heatmap.shape[:2]
#     img_vis_resized = cv2.resize(img_vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#
#     images_to_concat = [img_vis_resized, pred_heatmap, gt_heatmap]
#
#     if aux_map is not None:
#         aux_heatmap = to_heatmap(aux_map, 'coolwarm', force_show_variation=True)
#         aux_heatmap_resized = cv2.resize(aux_heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#         images_to_concat.append(aux_heatmap_resized)
#
#     concat = np.hstack(images_to_concat)
#     concat_bgr = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
#
#     if img_count is not None:
#         filename = f'{prefix}_{img_count:06d}.png'
#     elif epoch is not None:
#         filename = f'{prefix}_{epoch:03d}.png'
#     else:
#         filename = f'{prefix}_unknown.png'
#
#     try:
#         cv2.imwrite(os.path.join(save_dir, filename), concat_bgr)
#     except Exception as e:
#         print(f"[WARNING] Failed to save debug image: {e}")
#
#
# def evaluate(net, val_loader, device):
#     """
#     在验证集上评估模型
#     [Update] 增加了截断逻辑：计算 Clean Mean (剔除最差20%) 以应对脏数据 GT
#     """
#     print(f"[Validation] Starting evaluation on {len(val_loader)} batches...", flush=True)
#     net.eval()
#     errors = []
#
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(val_loader):
#             if batch_idx % 10 == 0 or batch_idx < 3:
#                 print(f"[Validation] Processing batch {batch_idx + 1}/{len(val_loader)}...", flush=True)
#             img = batch['image'].to(device)
#             gt_map = batch['gt_map'].to(device)
#             cm1 = batch['cm1'].to(device)
#             cm2 = batch['cm2'].to(device)
#
#             if img.min().item() == 0.0 and img.max().item() == 0.0: continue
#
#             img_input = torch.clamp(img, 1e-8, 1.0)
#             img_input = torch.pow(img_input, 1.0 / 2.2)
#
#             decoder_output = net(img_input, cm1, cm2)
#
#             if isinstance(decoder_output, dict):
#                 pred_map = decoder_output['final']
#             else:
#                 pred_map = decoder_output
#
#             # 验证集始终使用 Angular Error
#             err_map = angular_error_map(pred_map, gt_map)
#
#             gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
#             zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
#             err_map = err_map * (~zero_mask).float()
#
#             # 计算每张图的平均误差，并收集到 errors 列表中
#             valid_pixels = torch.sum((~zero_mask).float().view(err_map.size(0), -1), dim=1) + 1e-8
#             sum_errors = torch.sum(err_map.view(err_map.size(0), -1), dim=1)
#             batch_errors = sum_errors / valid_pixels
#
#             errors.extend(batch_errors.cpu().numpy())
#
#     # 转为 numpy 数组
#     errors = np.array(errors)
#
#     if len(errors) == 0:
#         return {'mean_error': 0, 'median_error': 0, 'clean_mean_error': 0, 'std_error': 0, 'min_error': 0,
#                 'max_error': 0}
#
#     # ====================================================
#     # [核心修改] 计算截断后的指标 (Clean Metric)
#     # 截断 20% (即保留 80%)
#     # ====================================================
#     # 1. 排序
#     errors_sorted = np.sort(errors)
#
#     # 2. 截断：保留误差最小的 80%
#     keep_ratio = 0.8  # [User Request] 设置为 0.8
#     num_keep = int(len(errors) * keep_ratio)
#     num_keep = max(1, num_keep)  # 保护逻辑
#
#     clean_errors = errors_sorted[:num_keep]
#     dirty_errors = errors_sorted[num_keep:]
#
#     clean_mean = np.mean(clean_errors) if len(clean_errors) > 0 else 0
#     clean_median = np.median(clean_errors) if len(clean_errors) > 0 else 0
#     original_mean = np.mean(errors)
#
#     print(f"\n[Validation Trimmed Stat] Total: {len(errors)} | Keep Top {int(keep_ratio * 100)}%: {num_keep}")
#     print(f"  > Clean Mean (Top 80%):   {clean_mean:.4f}°")
#     print(f"  > Clean Median (Top 80%): {clean_median:.4f}°")
#     print(f"  > Original Mean (All):    {original_mean:.4f}°")
#
#     if len(dirty_errors) > 0:
#         print(f"  > Dirty Samples (Worst 20%) Mean: {np.mean(dirty_errors):.4f}° | Max: {np.max(dirty_errors):.4f}°")
#
#     metrics = {
#         'mean_error': float(original_mean),  # 原始均值 (用于参考)
#         'clean_mean_error': float(clean_mean),  # [新] 清洗后均值 (用于 Loss 和 Scheduler)
#         'median_error': float(np.median(errors)),  # 原始中位数
#         'clean_median_error': float(clean_median), # [新] 清洗后中位数
#         'std_error': float(np.std(errors)),
#         'min_error': float(np.min(errors)),
#         'max_error': float(np.max(errors)),
#         'percentile_25': float(np.percentile(errors, 25)),
#         'percentile_75': float(np.percentile(errors, 75)),
#     }
#     return metrics
#
#
# def load_checkpoint(net, checkpoint_path, optimizer=None, strict=False):
#     if not os.path.exists(checkpoint_path):
#         print(f"Checkpoint not found: {checkpoint_path}")
#         return 0, float('inf')
#
#     print(f"Loading checkpoint from: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     model_state_dict = checkpoint.get('model_state_dict', checkpoint)
#
#     if isinstance(net, torch.nn.DataParallel):
#         net.module.load_state_dict(model_state_dict, strict=False)
#     else:
#         net.load_state_dict(model_state_dict, strict=False)
#
#     if optimizer is not None and 'optimizer_state_dict' in checkpoint:
#         try:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         except Exception:
#             print("  Warning: Failed to load optimizer state")
#
#     start_epoch = checkpoint.get('epoch', 0) + 1
#     best_loss = checkpoint.get('loss', float('inf'))
#     print(f"  Resuming from epoch {start_epoch}, previous loss: {best_loss:.4f}")
#     return start_epoch, best_loss
#
#
# def save_checkpoint(net, optimizer, epoch, loss, save_dir, is_best=False):
#     try:
#         os.makedirs(save_dir, exist_ok=True)
#         model_state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model_state_dict,
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#         }
#         torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'), _use_new_zipfile_serialization=False)
#         if is_best:
#             torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pth'), _use_new_zipfile_serialization=False)
#         if (epoch + 1) % 10 == 0:
#             torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1:03d}.pth'),
#                        _use_new_zipfile_serialization=False)
#     except Exception as e:
#         print(f"[ERROR] Failed to save checkpoint: {e}")
#
#
# def train(args):
#     if torch.cuda.is_available():
#         num_gpus = torch.cuda.device_count()
#         if args.use_multi_gpu and num_gpus > 1:
#             device = torch.device('cuda:0')
#             print(f"Detected {num_gpus} GPUs. Using DataParallel.")
#         else:
#             device = torch.device('cuda' if args.gpu_id is None else f'cuda:{args.gpu_id}')
#     else:
#         device = torch.device('cpu')
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     exp_name = args.exp_name if args.exp_name else f"exp_{timestamp}"
#     output_dir = os.path.join(args.output_dir, exp_name)
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'debug_imgs'), exist_ok=True)
#
#     print(f"Experiment: {exp_name}")
#     print(f"Output directory: {output_dir}")
#
#     net = network(input_size=64, cfe_feature_num=8, device=device).to(device)
#     if args.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
#         net = torch.nn.DataParallel(net)
#
#     optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
#     ) if args.use_scheduler else None
#
#     start_epoch = 0
#     best_val_loss = float('inf')
#     if args.resume:
#         start_epoch, best_val_loss = load_checkpoint(net, args.resume, optimizer)
#
#     # ====================================================
#     # [核心修改] 加载数据 & 过滤 Fujifilm
#     # ====================================================
#     print(f"[Data] Loading LSMI files from: {args.train_dir}")
#     lsmi_files = dataset.Data.load_files(args.train_dir)
#     print(f"[Data] Found {len(lsmi_files)} LSMI images.")
#
#     nus_files = []
#     if args.nus_dir and os.path.exists(args.nus_dir):
#         print(f"[Data] Loading NUS files from: {args.nus_dir}")
#         raw_nus_files = dataset.Data.load_files(args.nus_dir)
#
#         # [新增] 过滤 Fujifilm 逻辑
#         print("[Filter] Checking for Fujifilm images (unmasked color checker)...")
#         fuji_dropped_count = 0
#         for f in raw_nus_files:
#             # NUS路径通常包含相机名，或者文件名中包含
#             # 检查 "fujifilm" 是否在路径中 (不区分大小写)
#             if 'fujifilm' in f.lower():
#                 fuji_dropped_count += 1
#             else:
#                 nus_files.append(f)
#
#         print(f"[Filter] Dropped {fuji_dropped_count} Fujifilm images.")
#         print(f"[Data] Remaining valid NUS images: {len(nus_files)}")
#
#     elif args.nus_dir:
#         print(f"[Warning] NUS directory provided but not found: {args.nus_dir}")
#
#     # 合并两个列表
#     all_files = lsmi_files + nus_files
#     print(f"[Data] Total training images: {len(all_files)}")
#
#     # =================================================================
#     # 黑名单过滤逻辑 (Galaxy/Nikon)
#     # =================================================================
#     blacklist_configs = {
#         'galaxy': 'galaxy_blacklist.json',
#         'nikon': 'nikon_blacklist.json'
#     }
#
#     bad_sources = {'galaxy': {}, 'nikon': {}}
#     has_blacklist = False
#     for cam_key, json_path in blacklist_configs.items():
#         if os.path.exists(json_path):
#             try:
#                 with open(json_path, 'r') as f:
#                     raw_list = json.load(f)
#                 count = 0
#                 for item in raw_list:
#                     if '_' in item:
#                         parts = item.split('_')
#                         if len(parts) >= 2:
#                             place_name = parts[0]
#                             light_id = parts[1]
#                             if place_name not in bad_sources[cam_key]:
#                                 bad_sources[cam_key][place_name] = set()
#                             bad_sources[cam_key][place_name].add(light_id)
#                             count += 1
#                 if count > 0:
#                     print(f"[Filter] 已加载 {cam_key} 黑名单，涉及 {count} 个单光源异常样本")
#                     has_blacklist = True
#             except Exception as e:
#                 print(f"[Filter] Error loading {json_path}: {e}")
#
#     # 执行 Galaxy/Nikon 过滤
#     if has_blacklist:
#         clean_files = []
#         filtered_count = 0
#         pattern = re.compile(r'(Place\d+)_([123]+)')
#
#         for fpath in all_files:
#             fname = os.path.splitext(os.path.basename(fpath))[0]
#             should_drop = False
#             current_cam = None
#             fpath_lower = fpath.lower()
#             if 'galaxy' in fpath_lower:
#                 current_cam = 'galaxy'
#             elif 'nikon' in fpath_lower:
#                 current_cam = 'nikon'
#
#             if current_cam and current_cam in bad_sources:
#                 clean_fname = fname.split('_sensorname_')[0] if '_sensorname_' in fname else fname
#                 match = pattern.search(clean_fname)
#                 if match:
#                     curr_place = match.group(1)
#                     curr_lights = match.group(2)
#                     if curr_place in bad_sources[current_cam]:
#                         bad_light_set = bad_sources[current_cam][curr_place]
#                         for bad_light in bad_light_set:
#                             if bad_light in curr_lights:
#                                 should_drop = True
#                                 break
#             if should_drop:
#                 filtered_count += 1
#             else:
#                 clean_files.append(fpath)
#
#         print(f"[Filter] 过滤完成！共剔除 {filtered_count} 张 Galaxy/Nikon 受污染图片。")
#         all_files = clean_files
#         print(f"[Data] Final training images count: {len(all_files)}")
#     # =================================================================
#
#     random.shuffle(all_files)
#
#     val_files = []
#     train_files = all_files
#
#     if args.val_ratio > 0:
#         val_size = int(len(train_files) * args.val_ratio)
#         val_files = train_files[:val_size]
#         train_files = train_files[val_size:]
#
#         print(f"[Data] Split: Train={len(train_files)}, Val={len(val_files)}")
#         val_ds = dataset.Data(val_files, mode='testing', input_size=256)
#         val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     else:
#         val_loader = None
#
#     train_ds = dataset.Data(train_files, input_size=256)
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
#                               pin_memory=True)
#
#     with open(os.path.join(output_dir, 'config.json'), 'w') as f:
#         json.dump(vars(args), f, indent=4)
#
#     train_log = defaultdict(list)
#
#     print("Starting training with Trimmed Angular Error Loss (Train & Val Scheme)...")
#     for epoch in range(start_epoch, args.epochs):
#         net.train()
#         total_loss = 0
#         valid_batches = 0
#         last_batch_data = None
#
#         for i, batch in enumerate(train_loader):
#             if i % 10 == 0:
#                 print(f"[Epoch {epoch + 1}] Processing batch {i + 1}/{len(train_loader)}...", end='\r', flush=True)
#
#             img = batch['image'].to(device)
#             gt_map = batch['gt_map'].to(device)
#             cm1 = batch['cm1'].to(device)
#             cm2 = batch['cm2'].to(device)
#
#             img_input = torch.clamp(img, 1e-8, 1.0)
#             img_input = torch.pow(img_input, 1.0 / 2.2)
#
#             optimizer.zero_grad()
#             decoder_output = net(img_input, cm1, cm2)
#
#             if isinstance(decoder_output, dict):
#                 pred_map = decoder_output['final']
#                 aux_map = decoder_output['aux']
#             else:
#                 pred_map = decoder_output
#                 aux_map = None
#
#             err_map = angular_error_map(pred_map, gt_map)
#             gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
#             zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
#             mask = (~zero_mask).float()
#             err_map = err_map * mask
#
#             batch_errors = torch.mean(err_map.view(err_map.size(0), -1), dim=1)
#             # [核心修改] 强制训练时也只取前 80% (Keep 80%)
#             # 找到这一行 (大约 420 行左右)
#             # keep_ratio = 0.8  <-- 删除或注释掉这行
#
#             # 替换为：
#             keep_ratio = args.keep_ratio
#             current_batch_size = img.size(0)
#             num_keep = int(current_batch_size * keep_ratio)
#             num_keep = max(1, num_keep)
#             good_errors, _ = torch.topk(batch_errors, k=num_keep, largest=False)
#             loss = torch.mean(good_errors)
#
#             if epoch == 0 and i == 0:
#                 print(f"\n  [Loss Strategy] Trimmed Angular Loss (Keeping {num_keep}/{current_batch_size}, Top {int(keep_ratio*100)}%)")
#
#             aux_loss = 0.0
#             if aux_map is not None:
#                 gt_downsampled = nn.functional.interpolate(
#                     gt_map, size=aux_map.shape[2:], mode='bilinear', align_corners=True
#                 )
#                 gt_ds_norm = torch.norm(gt_downsampled, dim=1, keepdim=False)
#                 zero_mask_ds = torch.isclose(gt_ds_norm, torch.ones_like(gt_ds_norm) * 1.732, atol=1e-3)
#                 mask_ds = (~zero_mask_ds).float()
#                 aux_err = angular_error_map(aux_map, gt_downsampled)
#                 aux_err = aux_err * mask_ds
#                 aux_loss = torch.mean(aux_err)
#
#             tv_loss = total_variation_loss(pred_map)
#             total_loss_value = loss + args.tv_weight * tv_loss + args.aux_weight * aux_loss
#
#             if torch.isnan(total_loss_value):
#                 print(f"Warning: NaN loss detected. Skipping batch.")
#                 continue
#
#             total_loss_value.backward()
#             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
#             optimizer.step()
#
#             total_loss += total_loss_value.item()
#             valid_batches += 1
#
#             if i == len(train_loader) - 1:
#                 last_batch_data = (torch.clamp(img, 0, 1), pred_map, gt_map)
#
#                 if valid_batches > 0:
#                     avg_loss = total_loss / valid_batches
#                     train_log['train_loss'].append(avg_loss)
#
#                     val_loss = 0
#                     val_clean_mean = 0
#                     is_best = False
#                     metrics = {'mean_error': 0, 'median_error': 0}
#
#                     if val_loader:
#                         print(f"\n[Epoch {epoch + 1}] Starting Validation...")
#                         metrics = evaluate(net, val_loader, device)
#                         val_clean_mean = metrics['clean_mean_error']
#                         val_loss = val_clean_mean
#
#                         train_log['val_loss'].append(val_loss)
#                         train_log['val_mean_error'].append(metrics['mean_error'])
#                         train_log['val_median_error'].append(metrics['median_error'])
#
#                         if val_loss < best_val_loss:
#                             is_best = True
#                             best_val_loss = val_loss
#
#                         if scheduler: scheduler.step(val_loss)
#
#                     save_checkpoint(net, optimizer, epoch, avg_loss, os.path.join(output_dir, 'checkpoints'),
#                                     is_best=is_best)
#
#                     log_msg = f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f} | "
#                     if val_loader:
#                         log_msg += f"Val Clean Mean (80%): {val_clean_mean:.4f} | Val Orig Mean: {metrics['mean_error']:.4f} | Val Median: {metrics['median_error']:.4f} | "
#                     log_msg += f"LR: {optimizer.param_groups[0]['lr']:.6f}"
#                     print(log_msg)
#
#                     if last_batch_data:
#                         save_debug_images(last_batch_data[0], last_batch_data[1], last_batch_data[2],
#                                           epoch=epoch + 1, save_dir=os.path.join(output_dir, 'debug_imgs'))
#
#                     with open(os.path.join(output_dir, 'logs', 'training_log.json'), 'w') as f:
#                         json.dump(train_log, f, indent=4)
#
#     print("Training completed!")
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train Dense-CCMNet (Trimmed Angular Loss)")
#     parser.add_argument('--train_dir', type=str, default='/mnt/sda2/SMY/ccmnet/original_resized/LSMI')
#     parser.add_argument('--nus_dir', type=str, default=None, help='Path to NUS dataset for joint training')
#     parser.add_argument('--val_ratio', type=float, default=0.1)
#     parser.add_argument('--config', type=str, default=None)
#     parser.add_argument('--gpu_id', type=int, default=None)
#     parser.add_argument('--use_multi_gpu', action='store_true')
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=2e-4)
#     parser.add_argument('--weight_decay', type=float, default=0.0001)
#     parser.add_argument('--use_scheduler', action='store_true', default=True)
#     parser.add_argument('--output_dir', type=str, default='./experiments')
#     parser.add_argument('--exp_name', type=str, default=None)
#     parser.add_argument('--log_interval', type=int, default=5)
#     parser.add_argument('--use_combined_loss', action='store_true')
#     parser.add_argument('--loss_alpha', type=float, default=0.7)
#     parser.add_argument('--tv_weight', type=float, default=0.01)
#     parser.add_argument('--aux_weight', type=float, default=0.4)
#     parser.add_argument('--resume', type=str, default=None)
#     # 找到 parser 定义的地方，添加这一行
#     parser.add_argument('--keep_ratio', type=float, default=0.8,
#                         help='Ratio of samples to keep for loss calculation (0.0-1.0)')
#     args = parser.parse_args()
#
#     if args.config and os.path.exists(args.config):
#         with open(args.config, 'r') as f:
#             config = json.load(f)
#         for key, value in config.items():
#             if hasattr(args, key) and key not in ['config', 'description']:
#                 setattr(args, key, value)
#
#     try:
#         train(args)
#     except Exception as e:
#         import traceback
#
#         traceback.print_exc()
#
#
import sys
import os
import re
import cv2
import numpy as np
import json
import time
import random
from datetime import datetime
from collections import defaultdict

# 1. 设置项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.ccmnet.ccmnet import network
from src import dataset
import math
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def angular_error_map(pred, gt):
    """ 计算像素级角误差 (Robust Version) """
    pred = nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
    gt = nn.functional.normalize(gt, p=2, dim=1, eps=1e-8)
    dot = torch.sum(pred * gt, dim=1)
    dot = torch.clamp(dot, -0.999999, 0.999999)
    angle = torch.acos(dot) * (180.0 / math.pi)
    angle = torch.clamp(angle, 0.0, 180.0)
    return angle


def total_variation_loss(pred_map):
    batch_size = pred_map.size(0)
    diff_x = pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]
    tv_x = torch.abs(diff_x).sum()
    diff_y = pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]
    tv_y = torch.abs(diff_y).sum()
    H, W = pred_map.size(2), pred_map.size(3)
    tv_loss = (tv_x + tv_y) / (batch_size * H * W)
    return tv_loss


def save_debug_images(img, pred, gt, aux_map=None, epoch=None, img_count=None, save_dir='debug_imgs', prefix='epoch'):
    """ 保存可视化结果 """
    os.makedirs(save_dir, exist_ok=True)

    if img.dim() == 4:
        img = img[0].detach().cpu().permute(1, 2, 0).numpy()
        pred = pred[0].detach().cpu().permute(1, 2, 0).numpy()
        gt = gt[0].detach().cpu().permute(1, 2, 0).numpy()
        if aux_map is not None:
            aux_map = aux_map[0].detach().cpu().permute(1, 2, 0).numpy()
    else:
        img = img.detach().cpu().permute(1, 2, 0).numpy()
        pred = pred.detach().cpu().permute(1, 2, 0).numpy()
        gt = gt.detach().cpu().permute(1, 2, 0).numpy()
        if aux_map is not None:
            aux_map = aux_map.detach().cpu().permute(1, 2, 0).numpy()

    if img.size == 0 or img.max() == 0.0:
        img = np.ones((256, 256, 3), dtype=np.float32) * 0.5

    if img.max() > 1.0: img = img / img.max()
    img = np.clip(img, 0.0, 1.0)
    img_vis = np.clip(np.power(img, 1.0 / 2.2), 0, 1) * 255
    img_vis = img_vis.astype(np.uint8)

    def to_heatmap(illum_map, colormap='coolwarm', force_show_variation=False):
        intensity = np.linalg.norm(illum_map, axis=2)
        intensity_min, intensity_max = intensity.min(), intensity.max()
        intensity_range = intensity_max - intensity_min

        if intensity_range < 1e-6 and not force_show_variation:
            rgb_vis = illum_map.copy()
            rgb_abs_max = np.abs(rgb_vis).max()
            if rgb_abs_max < 1e-6:
                rgb_vis = np.ones_like(illum_map) * 0.5
            else:
                rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
                rgb_vis = np.clip(rgb_vis, 0, 1)
            heatmap = (rgb_vis * 255).astype(np.uint8)
        else:
            if intensity_range < 0.01:
                rgb_vis = illum_map.copy()
                rgb_abs_max = np.abs(rgb_vis).max()
                if rgb_abs_max < 1e-6:
                    rgb_vis = np.ones_like(illum_map) * 0.5
                else:
                    rgb_vis = rgb_vis / (rgb_abs_max + 1e-8)
                    rgb_vis = np.clip(rgb_vis, 0, 1)
                heatmap = (rgb_vis * 255).astype(np.uint8)
            else:
                intensity_vis = (intensity - intensity_min) / (intensity_range + 1e-8)
                cmap = plt.get_cmap(colormap)
                heatmap = cmap(intensity_vis)[:, :, :3]
                heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    pred_heatmap = to_heatmap(pred, 'coolwarm', force_show_variation=True)
    gt_heatmap = to_heatmap(gt, 'coolwarm', force_show_variation=True)

    target_h, target_w = pred_heatmap.shape[:2]
    img_vis_resized = cv2.resize(img_vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    images_to_concat = [img_vis_resized, pred_heatmap, gt_heatmap]

    if aux_map is not None:
        aux_heatmap = to_heatmap(aux_map, 'coolwarm', force_show_variation=True)
        aux_heatmap_resized = cv2.resize(aux_heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        images_to_concat.append(aux_heatmap_resized)

    concat = np.hstack(images_to_concat)
    concat_bgr = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)

    if img_count is not None:
        filename = f'{prefix}_{img_count:06d}.png'
    elif epoch is not None:
        filename = f'{prefix}_{epoch:03d}.png'
    else:
        filename = f'{prefix}_unknown.png'

    try:
        cv2.imwrite(os.path.join(save_dir, filename), concat_bgr)
    except Exception as e:
        print(f"[WARNING] Failed to save debug image: {e}")


def evaluate(net, val_loader, device, use_tta=False):
    """
    在验证集上评估模型，支持 TTA 和 详细指标计算
    [Update] 增加了 clean_median_error 的计算，防止 KeyError
    """
    print(f"[Validation] Starting evaluation on {len(val_loader)} batches (TTA={'ON' if use_tta else 'OFF'})...",
          flush=True)
    net.eval()
    errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            img = batch['image'].to(device)
            gt_map = batch['gt_map'].to(device)
            cm1 = batch['cm1'].to(device)
            cm2 = batch['cm2'].to(device)

            if img.min().item() == 0.0 and img.max().item() == 0.0: continue

            img_input = torch.clamp(img, 1e-8, 1.0)
            img_input = torch.pow(img_input, 1.0 / 2.2)

            if use_tta:
                # 1. 原始预测
                out1 = net(img_input, cm1, cm2)
                pred1 = out1['final'] if isinstance(out1, dict) else out1

                # 2. 翻转预测 (水平翻转 dims=[3])
                img_flip = torch.flip(img_input, dims=[3])
                out2 = net(img_flip, cm1, cm2)
                pred2_flip = out2['final'] if isinstance(out2, dict) else out2

                # 3. 翻转回来并平均
                pred2 = torch.flip(pred2_flip, dims=[3])
                pred_map = (pred1 + pred2) / 2.0
            else:
                # 正常推理
                decoder_output = net(img_input, cm1, cm2)
                if isinstance(decoder_output, dict):
                    pred_map = decoder_output['final']
                else:
                    pred_map = decoder_output

            err_map = angular_error_map(pred_map, gt_map)
            gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
            zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
            err_map = err_map * (~zero_mask).float()

            valid_pixels = torch.sum((~zero_mask).float().view(err_map.size(0), -1), dim=1) + 1e-8
            sum_errors = torch.sum(err_map.view(err_map.size(0), -1), dim=1)
            batch_errors = sum_errors / valid_pixels

            errors.extend(batch_errors.cpu().numpy())

    errors = np.array(errors)

    if len(errors) == 0:
        # [Fix] 返回字典补全所有键，防止 KeyError
        return {
            'mean_error': 0, 'median_error': 0,
            'clean_mean_error': 0, 'clean_median_error': 0,
            'best_25': 0, 'worst_25': 0
        }

    # 统计指标计算
    errors_sorted = np.sort(errors)
    n = len(errors)

    mean_val = np.mean(errors)
    median_val = np.median(errors)

    # Clean Metrics (Top 80%) 用于 Scheduler 和 Log
    num_keep_80 = max(1, int(n * 0.8))
    clean_errors = errors_sorted[:num_keep_80]  # 切片取出前80%

    clean_mean_80 = np.mean(clean_errors)
    # [新增] 计算 Clean Median
    clean_median_80 = np.median(clean_errors)

    # Best/Worst 25%
    best_25 = np.mean(errors_sorted[:n // 4]) if n >= 4 else 0
    worst_25 = np.mean(errors_sorted[-n // 4:]) if n >= 4 else 0

    q1 = np.percentile(errors, 25)
    q3 = np.percentile(errors, 75)
    trimean = (q1 + 2 * median_val + q3) / 4.0

    print(f"\n[Validation Statistics (TTA={use_tta})]")
    print(f"  > Mean:      {mean_val:.4f}°")
    print(f"  > Median:    {median_val:.4f}°")
    print(f"  > Best 25%:  {best_25:.4f}°")
    print(f"  > Worst 25%: {worst_25:.4f}°")
    # [新增] 打印 Clean Median 方便观察
    print(f"  > Clean Mean:{clean_mean_80:.4f}° | Clean Med: {clean_median_80:.4f}°")

    metrics = {
        'mean_error': float(mean_val),
        'median_error': float(median_val),
        'clean_mean_error': float(clean_mean_80),
        'clean_median_error': float(clean_median_80),  # [新增] 必须包含此键
        'best_25': float(best_25),
        'worst_25': float(worst_25),
        'trimean': float(trimean),
        'min_error': float(errors.min()),
        'max_error': float(errors.max())
    }
    return metrics


def load_checkpoint(net, checkpoint_path, optimizer=None, strict=False):
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(model_state_dict, strict=False)
    else:
        net.load_state_dict(model_state_dict, strict=False)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('loss', float('inf'))
    return start_epoch, best_loss


def save_checkpoint(net, optimizer, epoch, loss, metrics, save_dir, is_best=False):
    try:
        os.makedirs(save_dir, exist_ok=True)
        model_state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics
        }
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'), _use_new_zipfile_serialization=False)
        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pth'), _use_new_zipfile_serialization=False)
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1:03d}.pth'),
                       _use_new_zipfile_serialization=False)
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")


def train(args):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.use_multi_gpu and num_gpus > 1:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda' if args.gpu_id is None else f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name if args.exp_name else f"exp_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'debug_imgs'), exist_ok=True)

    net = network(input_size=64, cfe_feature_num=8, device=device).to(device)
    if args.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    ) if args.use_scheduler else None

    start_epoch = 0
    best_val_metric = float('inf')
    best_epoch_stats = {}

    if args.resume:
        start_epoch, best_val_metric = load_checkpoint(net, args.resume, optimizer)

    # ====================================================
    # [Data Loading & Filtering Section]
    # ====================================================
    print(f"[Data] Loading LSMI files from: {args.train_dir}")
    lsmi_files = dataset.Data.load_files(args.train_dir)
    print(f"[Data] Found {len(lsmi_files)} LSMI images.")

    nus_files = []
    if args.nus_dir and os.path.exists(args.nus_dir):
        print(f"[Data] Loading NUS files from: {args.nus_dir}")
        raw_nus_files = dataset.Data.load_files(args.nus_dir)

        # Filter Fujifilm
        print("[Filter] Checking for Fujifilm images (unmasked color checker)...")
        fuji_dropped_count = 0
        for f in raw_nus_files:
            if 'fujifilm' in f.lower():
                fuji_dropped_count += 1
            else:
                nus_files.append(f)
        print(f"[Filter] Dropped {fuji_dropped_count} Fujifilm images.")
        print(f"[Data] Remaining valid NUS images: {len(nus_files)}")
    elif args.nus_dir:
        print(f"[Warning] NUS directory provided but not found: {args.nus_dir}")

    all_files = lsmi_files + nus_files
    print(f"[Data] Total training images: {len(all_files)}")

    # =================================================================
    # Blacklist Filtering Logic (Galaxy/Nikon)
    # [FIX] Use absolute paths based on project_root
    # =================================================================
    blacklist_configs = {
        'galaxy': os.path.join(project_root, 'galaxy_blacklist.json'),
        'nikon': os.path.join(project_root, 'nikon_blacklist.json')
    }

    bad_sources = {'galaxy': {}, 'nikon': {}}
    has_blacklist = False

    for cam_key, json_path in blacklist_configs.items():
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    raw_list = json.load(f)
                count = 0
                for item in raw_list:
                    if '_' in item:
                        parts = item.split('_')
                        if len(parts) >= 2:
                            place_name = parts[0]
                            light_id = parts[1]
                            if place_name not in bad_sources[cam_key]:
                                bad_sources[cam_key][place_name] = set()
                            bad_sources[cam_key][place_name].add(light_id)
                            count += 1
                if count > 0:
                    print(f"[Filter] Loaded {cam_key} blacklist with {count} entries.")
                    has_blacklist = True
            except Exception as e:
                print(f"[Filter] Error loading {json_path}: {e}")
        else:
            print(f"[Filter] Warning: Blacklist file {json_path} not found.")

    if has_blacklist:
        clean_files = []
        filtered_count = 0
        pattern = re.compile(r'(Place\d+)_([123]+)')

        for fpath in all_files:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            should_drop = False

            # Determine camera type
            current_cam = None
            fpath_lower = fpath.lower()
            if 'galaxy' in fpath_lower:
                current_cam = 'galaxy'
            elif 'nikon' in fpath_lower:
                current_cam = 'nikon'

            if current_cam and current_cam in bad_sources:
                clean_fname = fname.split('_sensorname_')[0] if '_sensorname_' in fname else fname
                match = pattern.search(clean_fname)
                if match:
                    curr_place = match.group(1)
                    curr_lights = match.group(2)

                    if curr_place in bad_sources[current_cam]:
                        bad_light_set = bad_sources[current_cam][curr_place]
                        for bad_light in bad_light_set:
                            if bad_light in curr_lights:
                                should_drop = True
                                break

            if should_drop:
                filtered_count += 1
            else:
                clean_files.append(fpath)

        print(f"[Filter] Filtering complete! Dropped {filtered_count} polluted images.")
        all_files = clean_files
        print(f"[Data] Final training images count: {len(all_files)}")

    random.shuffle(all_files)

    val_files = []
    train_files = all_files
    if args.val_ratio > 0:
        val_size = int(len(train_files) * args.val_ratio)
        val_files = train_files[:val_size]
        train_files = train_files[val_size:]
        val_ds = dataset.Data(val_files, mode='testing', input_size=256)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader = None

    train_ds = dataset.Data(train_files, input_size=256)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    train_log = defaultdict(list)

    print(f"Starting training... TTA Validation: {'ON' if args.val_tta else 'OFF'}")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        total_loss = 0
        valid_batches = 0
        last_batch_data = None

        for i, batch in enumerate(train_loader):
            if i % 10 == 0:
                print(f"[Epoch {epoch + 1}] Processing batch {i + 1}/{len(train_loader)}...", end='\r', flush=True)

            img = batch['image'].to(device)
            gt_map = batch['gt_map'].to(device)
            cm1 = batch['cm1'].to(device)
            cm2 = batch['cm2'].to(device)

            img_input = torch.clamp(img, 1e-8, 1.0)
            img_input = torch.pow(img_input, 1.0 / 2.2)

            optimizer.zero_grad()
            decoder_output = net(img_input, cm1, cm2)

            if isinstance(decoder_output, dict):
                pred_map = decoder_output['final']
                aux_map = decoder_output['aux']
            else:
                pred_map = decoder_output
                aux_map = None

            # Loss calculation (Trimmed Logic)
            err_map = angular_error_map(pred_map, gt_map)
            gt_norm = torch.norm(gt_map, dim=1, keepdim=False)
            zero_mask = torch.isclose(gt_norm, torch.ones_like(gt_norm) * 1.732, atol=1e-3)
            mask = (~zero_mask).float()
            err_map = err_map * mask

            batch_errors = torch.mean(err_map.view(err_map.size(0), -1), dim=1)
            keep_ratio = args.keep_ratio  # 使用参数
            current_batch_size = img.size(0)
            num_keep = int(current_batch_size * keep_ratio)
            num_keep = max(1, num_keep)
            good_errors, _ = torch.topk(batch_errors, k=num_keep, largest=False)
            loss = torch.mean(good_errors)

            aux_loss = 0.0
            if aux_map is not None:
                gt_downsampled = nn.functional.interpolate(gt_map, size=aux_map.shape[2:], mode='bilinear',
                                                           align_corners=True)
                gt_ds_norm = torch.norm(gt_downsampled, dim=1, keepdim=False)
                zero_mask_ds = torch.isclose(gt_ds_norm, torch.ones_like(gt_ds_norm) * 1.732, atol=1e-3)
                mask_ds = (~zero_mask_ds).float()
                aux_err = angular_error_map(aux_map, gt_downsampled)
                aux_err = aux_err * mask_ds
                aux_loss = torch.mean(aux_err)

            tv_loss = total_variation_loss(pred_map)
            total_loss_value = loss + args.tv_weight * tv_loss + args.aux_weight * aux_loss

            if torch.isnan(total_loss_value): continue

            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += total_loss_value.item()
            valid_batches += 1

            # ... (接上文 total_loss += total_loss_value.item() 等) ...

            if i == len(train_loader) - 1:
                last_batch_data = (torch.clamp(img, 0, 1), pred_map, gt_map)

                if valid_batches > 0:
                    avg_loss = total_loss / valid_batches
                    train_log['train_loss'].append(avg_loss)

                    val_loss = 0
                    is_best = False
                    # 初始化所有键，防止 val_loader 为空时报错
                    metrics = {'mean_error': 0, 'median_error': 0, 'clean_mean_error': 0, 'clean_median_error': 0,
                               'best_25': 0, 'worst_25': 0}

                    if val_loader:
                        print(f"\n[Epoch {epoch + 1}] Starting Validation...")
                        metrics = evaluate(net, val_loader, device, use_tta=args.val_tta)

                        # [Fix 1] 必须先从 metrics 里提取值赋给 val_loss！
                        # 我们使用 Clean Mean (剔除后均值) 作为验证集的 Loss 标准
                        val_loss = metrics['clean_mean_error']

                        # [Fix 2] 记录所有指标，包括“剔除后”的 Clean Median
                        train_log['val_loss'].append(val_loss)
                        train_log['val_mean'].append(metrics['mean_error'])
                        train_log['val_median'].append(metrics['median_error'])  # 原始 Median
                        train_log['val_clean_mean'].append(metrics['clean_mean_error'])  # 剔除后 Mean
                        train_log['val_clean_median'].append(metrics['clean_median_error'])  # [新增] 剔除后 Median

                        train_log['val_best25'].append(metrics['best_25'])
                        train_log['val_worst25'].append(metrics['worst_25'])

                        # 更新最佳模型
                        if val_loss < best_val_metric:
                            is_best = True
                            best_val_metric = val_loss
                            best_epoch_stats = metrics.copy()
                            best_epoch_stats['epoch'] = epoch + 1
                            print(f"  >>> New Best Model! (Clean Mean: {best_val_metric:.4f}°) <<<")

                        # 学习率调整
                        if scheduler:
                            scheduler.step(val_loss)

                    # 保存 Checkpoint
                    save_checkpoint(net, optimizer, epoch, avg_loss, metrics, os.path.join(output_dir, 'checkpoints'),
                                    is_best=is_best)

                    # 打印日志
                    log_msg = f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f} | "
                    if val_loader:
                        # 打印丰富的信息供观察
                        log_msg += f"CleanMean: {metrics['clean_mean_error']:.4f} | CleanMedian: {metrics['clean_median_error']:.4f} | OrigMedian: {metrics['median_error']:.4f} | "
                    log_msg += f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                    print(log_msg)

                    if last_batch_data:
                        save_debug_images(last_batch_data[0], last_batch_data[1], last_batch_data[2],
                                          epoch=epoch + 1, save_dir=os.path.join(output_dir, 'debug_imgs'))

                    with open(os.path.join(output_dir, 'logs', 'training_log.json'), 'w') as f:
                        json.dump(train_log, f, indent=4)

        print("\n" + "=" * 60)
        print("Training Completed.")
        # ... (后续打印 Best Model Stats 的代码) ...
    if best_epoch_stats:
        print(f"Best Model Stats (at Epoch {best_epoch_stats['epoch']}):")
        print(f"  Mean:      {best_epoch_stats['mean_error']:.4f}°")
        print(f"  Median:    {best_epoch_stats['median_error']:.4f}°")
        print(f"  Best 25%:  {best_epoch_stats['best_25']:.4f}°")
        print(f"  Worst 25%: {best_epoch_stats['worst_25']:.4f}°")

        with open(os.path.join(output_dir, 'final_best_metrics.json'), 'w') as f:
            json.dump(best_epoch_stats, f, indent=4)
        print(f"Best metrics saved to {os.path.join(output_dir, 'final_best_metrics.json')}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Dense-CCMNet")
    parser.add_argument('--train_dir', type=str, default='/mnt/sda2/SMY/ccmnet/original_resized/LSMI')
    parser.add_argument('--nus_dir', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--tv_weight', type=float, default=0.01)
    parser.add_argument('--aux_weight', type=float, default=0.4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--keep_ratio', type=float, default=0.8, help='Ratio of samples to keep (0.0-1.0)')

    # [New] 添加 Validation TTA 参数
    parser.add_argument('--val_tta', action='store_true', help='Enable TTA during validation')

    args = parser.parse_args()

    # ... (Config加载保持不变) ...

    try:
        train(args)
    except Exception as e:
        import traceback

        traceback.print_exc()