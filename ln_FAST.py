import os
import sys
from pathlib import Path


def create_fits_symlink(mode: str = "range"):
    base_path = Path('/home/pl/work/pulsar/原始数据/FFT')
    current_dir = Path('/home/pl/work/python-workspace/power_stacking/19c13')
    current_dir.mkdir(exist_ok=True, parents=True)
    if base_path.name != current_dir.name:
        print(f"源文件夹: {base_path.name}, 目标文件夹: {current_dir.name}")
        print("警告：源目录与目标目录名称不一致！")
        user_input = input("是否继续操作？(y/n): ")
        if user_input.lower() != 'y':
            print("用户取消操作，程序退出。")
            return

    filtered_files = []
    if mode.lower() == "all":
        filtered_files = list(base_path.glob("*.fits"))
    elif mode.lower() == "range":
        suffixes = [f'{str(i).zfill(4)}.fits' for i in range(0, 2000)]
        all_fits = list(base_path.glob("*.fits"))
        for filepath in all_fits:
            if any(filepath.name.endswith(suffix) for suffix in suffixes):
                filtered_files.append(filepath)
    else:
        print("mode 参数仅支持 'all' / 'range'")
        return

    filtered_files.sort()

    for filepath in filtered_files:
        link_path = current_dir / filepath.name

        if link_path.exists():
            if link_path.is_symlink():
                print(f"已存在软链接: {link_path} -> {os.readlink(link_path)}")
            else:
                print(f"存在同名实体文件，跳过: {link_path}")
            continue

        try:
            link_path.symlink_to(filepath)
            print(f"创建成功: {link_path} -> {filepath}")
        except Exception as e:
            print(f"创建失败 [{filepath.name}]: {str(e)}")

    print(f"\n操作完成，共处理 {len(filtered_files)} 个 fits 文件。")


if __name__ == "__main__":

    # create_fits_symlink(mode="range")
    create_fits_symlink(mode="all")

