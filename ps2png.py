#!/usr/bin/env python3

import os
import glob
import subprocess
from PIL import Image
import sys

def convert_ps_to_png(input_path, rotated=True, output_dir=None):
    """Convert PS file to PNG, optionally rotate -90 degrees"""
    try:
        # Set output path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.png") if output_dir \
                    else f"{os.path.splitext(input_path)[0]}.png"
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Ghostscript conversion
        subprocess.run([
            "gs", "-dQUIET", "-dBATCH", "-dNOPAUSE",
            "-sDEVICE=png256",
            f"-sOutputFile={output_file}",
            "-r300", input_path
        ], check=True)

        # Rotate if needed
        if rotated:
            with Image.open(output_file) as img:
                img.rotate(-90, expand=True).save(output_file)
        
        print(f"✓ 转换成功: {output_file}")
        return True
    
    except Exception as e:
        print(f"❌ 错误: {input_path} 处理失败 - {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        input_pattern = "*.ps"
    else:
        input_pattern = sys.argv[1]

    # Check dependencies
    try:
        subprocess.run(["gs", "--version"], check=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Image.open  # Test PIL import
    except (subprocess.CalledProcessError, ImportError) as e:
        print("错误：需要安装依赖:")
        print("1. Ghostscript: sudo yum install ghostscript")
        print("2. Python包: pip install pillow")
        sys.exit(1)

    # Find matching files
    ps_files = glob.glob(input_pattern, recursive=False)
    if not ps_files:
        print(f"未找到匹配的.ps文件: {input_pattern}")
        return

    # Process files
    success_count = 0
    for ps_file in ps_files:
        if os.path.isfile(ps_file):
            if convert_ps_to_png(ps_file):
                success_count += 1

    print(f"\n处理完成! 成功转换 {success_count}/{len(ps_files)} 个文件")

if __name__ == "__main__":
    main()