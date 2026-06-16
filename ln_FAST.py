
import os
from pathlib import Path


base_path = Path('/home/pengl/pulsar/data/2.5/N2024_3/M72/20250331')
current_dir = Path('/home/pengl/pulsar/work/M72/20250331')


if base_path.name != current_dir.name:
    print(f"源文件夹: {base_path.name}, 目标文件夹: {current_dir.name}")
    print("警告：源目录与目标目录名称不一致！")
    user_input = input("是否继续操作？(y/n): ")
    if user_input.lower() != 'y':
        print("用户取消操作，程序退出。")
        exit()

suffixes = [f'{str(i).zfill(4)}.fits' for i in range(0, 2000)]

prefix_pattern = '*'

matched_files = list(base_path.glob(f'{prefix_pattern}[0-9][0-9][0-9][0-9].fits'))


filtered_files = []
for filepath in matched_files:
    if any(filepath.name.endswith(suffix) for suffix in suffixes):
        filtered_files.append(filepath)

for filepath in filtered_files:
    link_path = current_dir / filepath.name

    if link_path.exists():
        if link_path.is_symlink():
            print(f"符号链接已存在: {link_path} -> {os.readlink(link_path)}")
        else:
            print(f"文件已存在且不是符号链接: {link_path}")
        continue

    try:
        link_path.symlink_to(filepath)
        print(f"创建成功: {link_path} -> {filepath}")
    except Exception as e:
        print(f"创建失败 [{filepath.name}]: {str(e)}")

print(f"\n操作完成，共处理 {len(filtered_files)} 个文件。")
