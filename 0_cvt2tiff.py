# Convert single channel bayer pattern RAW files to 3 channel RGB tiff files using DCRAW
import os
import shutil
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="将RAW文件转换为TIFF格式（用于生成mixture map）")
    parser.add_argument("--camera", type=str, required=True,
                       choices=["galaxy", "sony", "nikon"],
                       help="相机名称")
    parser.add_argument("--data_root", type=str,
                       default="/mnt/sda2/SMY/ccmnet/LSMI",
                       help="LSMI数据根目录 (default: /mnt/sda2/SMY/ccmnet/LSMI)")
    parser.add_argument("--source_dir", type=str, default=None,
                       help="RAW文件源目录（如果未指定，使用data_root/camera/PlaceXXX/）")
    
    args = parser.parse_args()
    
    # 确定RAW文件扩展名
    raw_extensions = {
        'sony': 'arw',
        'nikon': 'nef',
        'galaxy': 'dng'
    }
    EXT = raw_extensions.get(args.camera.lower(), 'dng')
    
    # 确定源目录（原始数据集路径，包含Place目录）
    if args.source_dir:
        SOURCE = args.source_dir
    else:
        # 根据相机自动确定原始数据集路径
        camera_source_paths = {
            'sony': '/mnt/sda2/SMY/dataset/sony',
            'galaxy': '/mnt/sda2/SMY/dataset/galaxy',
            'nikon': '/mnt/sda1/SMY/project/dataset/LSMI/nikon'
        }
        SOURCE = camera_source_paths.get(args.camera.lower())
        if SOURCE is None:
            # 回退到data_root下的路径
            SOURCE = os.path.join(args.data_root, args.camera)
    
    if not os.path.exists(SOURCE):
        print(f"错误: 源目录不存在: {SOURCE}")
        return
    
    # 确定输出目录（TIFF文件保存路径）
    camera_output_paths = {
        'sony': '/mnt/sda2/SMY/dataset/sony',
        'galaxy': '/mnt/sda2/SMY/dataset/galaxy',
        'nikon': '/mnt/sda2/SMY/dataset/nikon'  # Nikon输出到不同路径
    }
    OUTPUT_DIR = camera_output_paths.get(args.camera.lower(), SOURCE)
    
    # 如果输出目录不存在，创建它
    if not os.path.exists(OUTPUT_DIR):
        print(f"输出目录不存在，正在创建: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"转换 {args.camera} 相机的RAW文件为TIFF...")
    print(f"源目录: {SOURCE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"RAW扩展名: .{EXT}")
    
    # 查找所有Place目录
    places = [f for f in os.listdir(SOURCE) 
              if os.path.isdir(os.path.join(SOURCE, f)) and f.startswith("Place")]
    
    if not places:
        print(f"警告: 在 {SOURCE} 下没有找到Place目录")
        print(f"请确保数据目录结构正确")
        return

    for place in places:
        source_path = os.path.join(SOURCE, place)
        output_path = os.path.join(OUTPUT_DIR, place)
        
        # 如果输出Place目录不存在，创建它
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        files = [f for f in os.listdir(source_path) if f.lower().endswith(f".{EXT}")]
        
        if not files:
            print(f"跳过 {place}: 没有找到.{EXT}文件")
            continue
        
        print(f"\n处理 {place}: 找到 {len(files)} 个RAW文件")

        for file in files:
            source_file_path = os.path.join(source_path, file)
            # TIFF文件保存到输出目录
            tiff_filename = os.path.splitext(file)[0] + ".tiff"
            tiff_path = os.path.join(output_path, tiff_filename)
            
            # 检查是否已存在对应的TIFF文件
            if os.path.exists(tiff_path):
                print(f"  跳过 {file}: TIFF文件已存在")
                continue
            
            cmd = ["dcraw", "-h", "-D", "-4", "-T", "-o", "0", source_file_path]
            print(f"  转换: {file} -> {tiff_filename}")
            try:
                result = subprocess.call(cmd, timeout=300)  # 5分钟超时
            except subprocess.TimeoutExpired:
                print(f"  错误: 转换超时: {file}")
                continue
            except Exception as e:
                print(f"  错误: 转换失败: {file}, 错误: {e}")
                continue
            
            if result == 0:
                # dcraw会在源文件目录生成TIFF，需要移动到输出目录
                generated_tiff = os.path.splitext(source_file_path)[0] + ".tiff"
                if os.path.exists(generated_tiff) and generated_tiff != tiff_path:
                    shutil.move(generated_tiff, tiff_path)
                    print(f"    已移动到: {tiff_path}")
            else:
                print(f"  警告: 转换失败: {file} (退出码: {result})")
    
    print(f"\n完成！TIFF文件已生成在 {OUTPUT_DIR} 下的各个Place目录中")

if __name__ == "__main__":
    main()
