#!/usr/bin/env python3
import os,sys
import argparse
import glob
import re
import shutil
from datetime import datetime
from psr_fuc import *

def find_cfg_file():
    for folder in [".", "..", "../..", "../../..","../../../.."]:
        try:
            for fname in os.listdir(folder):
                if fname.endswith(".cfg"):
                    return os.path.join(folder, fname)
        except FileNotFoundError:
            continue
    return None

def parse_config_value(cfg_path, param_name):
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) >= 2 and parts[0] == param_name:
                return parts[1].strip()
    return None

def fold_task(cmd, ifok,logfile, work_dir,png_dir):
    whitelist = []
    filename = os.path.basename(ifok)
    png_name = f'{filename[:-4]}.png'
    print(png_name)
    ps_path = os.path.join(work_dir,f'{filename[:-4]}.ps')
    run_cmd(cmd, ifok = ifok, work_dir=work_dir,log_file=logfile,mode='both')
    ps2png(ps_path)
    handle_files(work_dir, png_dir, 'copy',png_name )

def pool_fold(num_processes, task_name, cmd_list, ifok_list,log_list, work_dir=os.getcwd(),png_dir = None):
    if len(cmd_list) != len(ifok_list):
        raise ValueError("cmd_list和ifok_list长度必须一致")

    progress_bar = tqdm(
        total=len(cmd_list),
        desc=f"{task_name}-{num_processes}核",
        unit="cmd",
        dynamic_ncols=True,
    )

    def update(*args):
        progress_bar.update()
    
    def handle_error(error):
        progress_bar.write(f"任务执行错误: {error}")

    process_pool = Pool(num_processes)
    try:
        results = [
            process_pool.apply_async(
                fold_task,
                args=(cmd, ifok, log_file,work_dir,png_dir),
                callback=update,
                error_callback=handle_error
            )
            for cmd, ifok,log_file in zip(cmd_list, ifok_list,log_list)
        ]
        process_pool.close()
        process_pool.join()
    except Exception as e:
        process_pool.terminate()
        raise e
    finally:
        progress_bar.close()

ncpus = 4
cmdfile_name = "fold_raw.sh"
if not os.path.isfile(cmdfile_name):
    print(f"当前文件夹中不存在文件：{cmdfile_name}")
    
if (len(sys.argv) == 2 or ("-h" in sys.argv) or ("-help" in sys.argv) or ("--help" in sys.argv)):
    print("Usage: %s -cmdfile \"commands.sh\" -ncpus N " % (os.path.basename(sys.argv[0])))
    print('将使用默认值')
    sys.exit(0)  
else:
    for j in range(1, len(sys.argv)):
        if (sys.argv[j] == "-cmdfile"):
            cmdfile_name = sys.argv[j+1]
            print(f"使用指定的：{cmdfile_name},注意不要使用绝对路径，这会导致折叠信息mask混淆")
        elif (sys.argv[j] == "-ncpus"):
            ncpus = int(sys.argv[j+1])

cfg_file = find_cfg_file()
current_path = os.getcwd()
maskstr = os.path.basename(current_path)

work_dir = parse_config_value(cfg_file, "ROOT_WORKDIR")
log_dir = os.path.join(work_dir,'LOG',f'ts2raw_{maskstr}')
os.makedirs(log_dir,exist_ok=True)
png_dir = os.path.join(work_dir,'06_PNG',maskstr)
os.makedirs(png_dir,exist_ok=True)

with open(cmdfile_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()
cmd_list = [line.strip() for line in lines]
cmd_list = sorted(cmd_list)

ifok_files = []
logfiles = []
for line in cmd_list:
    nsub = line.split("-nsub")[1].strip().split()[0] if "-nsub" in line else None
    n = line.split("-n")[1].strip().split()[0] if "-n " in line else None
    parfile = line.split("-par")[1].strip().split()[0] if "-par" in line else None
    maskfile = line.split("-mask")[1].strip().split()[0] if "-mask" in line else None
    outname = line.split("-o")[1].strip().split()[0] if "-o" in line else None
    datafile = line.strip().split()[-1]
    ps_files = glob.glob(f"{outname}*.ps") 

    if ps_files:
        png_file = os.path.join(png_dir,ps_files[0][:-3] + '.png')
        ifok_files.append(png_file)
    else:
        ifok_files.append(f"where/{outname}*.png")

    logfile = os.path.join(log_dir,f'LOG_{outname}.txt')
    logfiles.append(logfile)
    
print(ifok_files)
ncpus = min(ncpus,4*len(cmd_list))
pool_fold(ncpus,'fold',cmd_list,ifok_files,logfiles,work_dir = os.getcwd(),png_dir=png_dir)

# ====================== 简洁版追加功能（功能不变） ======================
def extract_path_segments(work_dir):
    path_parts = [p for p in work_dir.split(os.sep) if p.strip()]
    digital_segment = ""
    middle_segment = ""
    digital_pattern = re.compile(r'^\d{8}$')
    for idx, part in enumerate(path_parts):
        if digital_pattern.match(part):
            digital_segment = part
            if idx + 1 < len(path_parts):
                middle_segment = path_parts[idx + 1]
            break
    return digital_segment, middle_segment

def handle_fold_png_images(base_dir):
    if not cfg_file:
        print("未找到.cfg文件，跳过处理")
        return

    sourcename = parse_config_value(cfg_file, "SOURCE_NAME")
    search_label = parse_config_value(cfg_file, "SEARCH_LABEL")
    if not all([sourcename, search_label, work_dir, base_dir]):
        print("配置参数缺失，跳过处理")
        return

    digital_seg, middle_seg = extract_path_segments(work_dir)
    rename_suffix = ""
    target_dir = ""

    if digital_seg and middle_seg:
        print(f"提取到有效片段：{digital_seg} {middle_seg}")
        target_dir = os.path.join(base_dir, sourcename, digital_seg, middle_seg, search_label)
        rename_suffix = f"{digital_seg}-{middle_seg}-{search_label}"
    else:
        print(f"未提取有效数字片段，使用降级目录")
        target_dir = os.path.join(base_dir, sourcename, search_label)
        rename_suffix = f"{sourcename}-{search_label}"

    os.makedirs(target_dir, exist_ok=True)
    print(f"目标目录：{target_dir}")
    file_copy_info = []

    fold_raw_path = os.path.abspath(cmdfile_name)
    file_copy_info.append((fold_raw_path, "fold_raw.sh"))
    folding_dir = os.path.join(work_dir, "05_FOLDING")
    id_list_candidate_paths = [
        os.path.join(folding_dir, "raw_fold", "id_list.txt"),
        os.path.join(folding_dir, "raw", "id_list.txt"),
        os.path.join(folding_dir, "raw_prep", "id_list.txt")
    ]
    id_list_path = ""
    for candidate in id_list_candidate_paths:
        if os.path.exists(candidate):
            id_list_path = candidate
            break
    file_copy_info.append((id_list_path, "id_list.txt"))

    script_fold_ts_path = ""
    sourcename_mask = f"{sourcename}_{search_label}" if sourcename and search_label else ""
    if sourcename_mask:
        matching_folders = glob.glob(os.path.join(folding_dir, f"{sourcename_mask}*"))
        if matching_folders:
            script_fold_ts_path = os.path.join(matching_folders[0], "script_fold_ts.txt")
    file_copy_info.append((script_fold_ts_path, "script_fold_ts.txt"))

    for file_path, file_name in file_copy_info:
        if not file_path or not os.path.exists(file_path):
            print(f"{file_name} 不存在，跳过")
            continue
        target_file_path = os.path.join(target_dir, file_name)
        shutil.copy2(file_path, target_file_path)
        print(f"复制：{file_name} -> {target_dir}")

    png_source_dirs = [png_dir, os.getcwd(), os.path.join(work_dir, "05_FOLDING"), os.path.join(work_dir, "06_PNG")]
    png_files = []
    for src_dir in png_source_dirs:
        if not os.path.exists(src_dir):
            continue
        lower_png = glob.glob(os.path.join(src_dir, "*.png"))
        upper_png = glob.glob(os.path.join(src_dir, "*.PNG"))
        png_files.extend(lower_png)
        upper_png = [f for f in upper_png if f not in png_files]
        png_files.extend(upper_png)

    if not png_files:
        print("未找到PNG图片，跳过处理")
        return
    print(f"找到 {len(png_files)} 张PNG图片")

    for png_path in png_files:
        original_filename = os.path.basename(png_path)
        original_name, ext = os.path.splitext(original_filename)
        new_filename = f"{original_name}-{rename_suffix}{ext.lower()}"
        target_png_path = os.path.join(target_dir, new_filename)
        if os.path.exists(target_png_path):
            print(f"{new_filename} 已存在，跳过")
            continue
        shutil.copy2(png_path, target_png_path)
        print(f"复制：{original_filename} -> {new_filename}")

if __name__ == "__main__":
    base_dir = "/home/pengl/pulsar/png-fold"
    handle_fold_png_images(base_dir)