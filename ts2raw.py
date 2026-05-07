#!/usr/bin/env python3
import os
import re
import shutil
import glob
import shlex
import string
from datetime import datetime
from psr_fuc import *

def numbers_from_filenames(folder='.'):
    filenames = os.listdir(folder)
    numbers = []
    for filename in filenames:
        match = re.search(r'A(\d+)', filename)
        if match:
            numbers.append(int(match.group(1)))
    return numbers

def find_cfg_file():
    for folder in [".", "..", "../.."]:
        try:
            for fname in os.listdir(folder):
                if fname.endswith(".cfg"):
                    return os.path.join(folder, fname)
        except FileNotFoundError:
            continue
    return None

def extract_ra_dec(cfg_path):
    ra, dec = None, None
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("RA"):
                ra_match = re.search(r"RA\s+([\d:.\-+]+)", line)
                if ra_match:
                    ra = ra_match.group(1)
            elif line.strip().startswith("DEC"):
                dec_match = re.search(r"DEC\s+([\d:.\-+]+)", line)
                if dec_match:
                    dec = dec_match.group(1)
    return ra, dec

def parse_config_value(cfg_path, param_name):
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) >= 2 and parts[0] == param_name:
                return parts[1].strip().strip('"')
    return None  

def write_par_file(source,ra,dec,F0,DM,filename):
    with open(filename, "w") as f:
        f.write(f"PSR       {source}\nRAJ      {ra}\nDECJ     {dec}\nF0        {F0}\nF1         0.000000000000D+00 \nDM        {DM}\n")

def savefilenodb(file,obs):
    with open(file, 'a') as burst_file, open(file, 'r') as read_file:
        existing_obs = set(line.strip() for line in read_file)
        if obs not in existing_obs:
            burst_file.write(obs + '\n')

cfg_file = find_cfg_file()
if cfg_file:
    ra, dec = extract_ra_dec(cfg_file)
    print(f"RA: {ra}\nDEC: {dec}")
else:
    print("未找到.cfg文件")

ifbary = parse_config_value(cfg_file, 'IF_BARY')
label = parse_config_value(cfg_file, "SEARCH_LABEL")
work_dir = parse_config_value(cfg_file, "ROOT_WORKDIR")
part_ra = ra.split(":") if ra else []
part_dec = dec.split(":") if dec else []
sign = "" if (dec and dec.startswith("-")) else "+"
source = (part_ra[0]+part_ra[1]+sign+part_dec[0]+part_dec[1]) if (ra and dec) else ""

fold_add = parse_config_value(cfg_file, "PREPFOLD_FLAGS")
current_date = datetime.now().strftime("%Y-%m-%d")
cmd_dir = os.path.join(work_dir,'06_PNG','cmd',current_date)
os.makedirs(cmd_dir,exist_ok=True)

sourcename = parse_config_value(cfg_file, "SOURCE_NAME")
search_label = parse_config_value(cfg_file, "SEARCH_LABEL")
sourcename_mask = f"{sourcename}_{search_label}" if (sourcename and search_label) else ""

folding_dir = os.path.join(work_dir, '05_FOLDING') if work_dir else ""
if ifbary == '1':
    fold2dir1 = os.path.join(folding_dir, 'raw_prep')
    fold2dir2 = os.path.join(folding_dir, 'raw_fold')
    os.makedirs(fold2dir1,exist_ok=True)
else:
    fold2dir2 = os.path.join(folding_dir, 'raw') if folding_dir else ""
os.makedirs(fold2dir2,exist_ok=True)
output_file = os.path.join(fold2dir2, 'id_list.txt')

if not os.path.exists(output_file):
    numbers = numbers_from_filenames()
    sorted_numbers = sorted(numbers)
    with open(output_file, 'w') as f:
        for num in sorted_numbers:
            f.write(f"{num}\n")
    print(f"{sorted_numbers}\n已生成，查看并修改：\n{output_file}")
else:
    print(f"已修改{output_file}?\n读取其中的内容...")
    with open(output_file, 'r') as f:
        sorted_numbers = [int(line.strip()) for line in f if line.strip().isdigit()]
    print(sorted_numbers)

    if ifbary == '1':
        matching_folders = glob.glob(os.path.join(folding_dir, f'{sourcename_mask}*')) if (folding_dir and sourcename_mask) else []
        if matching_folders:
            target_folder = matching_folders[0]
            fold_file = os.path.join(target_folder, 'script_fold_ts.txt')
            for d in [fold2dir1, fold2dir2, cmd_dir]:
                shutil.copy(fold_file, d)
            print(f"文件已复制到 {fold2dir1}")

        SNR_file = os.path.join(work_dir, '04_SIFTING/cand_sifting.txt') if work_dir else ""
        if os.path.exists(SNR_file):
            for d in [fold2dir1, fold2dir2, cmd_dir]:
                shutil.copy(SNR_file, d)

        timebin = parse_config_value(cfg_file, 'RFIFIND_TIME')
        maskfile = os.path.join(work_dir, f'01_RFIFIND/rfi{timebin}s_rfifind.mask') if (work_dir and timebin) else ""
        inputfile = os.path.join(work_dir, 'RAW','*fits') if work_dir else ""

        fold_file_raw = os.path.join(fold2dir2, 'fold_raw.sh')
        is_first_write = True  

        idx = 0
        if folding_dir and sourcename_mask and maskfile and inputfile:
            with open(fold_file, "r") as f:
                for i, line in enumerate(f, start=1):  
                    if i in sorted_numbers:
                        if is_first_write:
                            if os.path.exists(fold_file_raw) and os.path.getsize(fold_file_raw) > 0:
                                backup_file = f"{fold_file_raw}_copy"
                                if os.path.exists(backup_file):
                                    os.remove(backup_file)
                                shutil.copy2(fold_file_raw, backup_file)  
                                os.remove(fold_file_raw)
                                print(f"检测到原有文件非空，已备份至：{backup_file}")
                            is_first_write = False

                        idx += 1
                        dm = line.split("-dm")[1].strip().split()[0] if "-dm" in line else ""
                        accelcand = line.split("-accelcand")[1].strip().split()[0] if "-accelcand" in line else ""
                        accelfile = line.split("-accelfile")[1].strip().split()[0] if "-accelfile" in line else ""
                        datafile = line.strip().split()[-1] if line.strip() else ""

                        txtcand = os.path.splitext(accelfile)[0] if accelfile else ""
                        print(txtcand, accelcand)
                        
                        if txtcand and os.path.exists(txtcand):
                            with open(txtcand, 'r') as file_cand:
                                infos = file_cand.readlines()
                                line_info = infos[int(accelcand)+2].strip() if (len(infos) > int(accelcand)+2) else ""
                                print(line_info)
                                fields = line_info.split() if line_info else []
                                if len(fields) >= 11:
                                    period = fields[5]
                                    frequency = fields[6]
                                    period_clean = period[:period.find("(")] if "(" in period else period
                                    frequency_clean = frequency[:frequency.find("(")] if "(" in frequency else frequency
                                    print(period_clean)

                                    type_par = string.ascii_uppercase[(idx - 1) % 26]
                                    outname = f'{type_par}{i}DM{dm}_{period_clean}ms' if (dm and period_clean) else ""
                                    parname = os.path.join(fold2dir1,f'{type_par}{i}.par') if fold2dir1 else ""
                                    if outname and parname and source and ra and dec and frequency_clean:
                                        write_par_file(source,ra,dec,frequency_clean,dm,parname)

                                        cmd = f'prepfold {fold_add} -noxwin  -par {parname} -mask {maskfile} -o {outname} {inputfile}'
                                        savefilenodb(fold_file_raw, cmd)
                                        shutil.copy(fold_file_raw, cmd_dir)

    else:
        matching_folders = glob.glob(os.path.join(work_dir, '06_PNG', f'{sourcename_mask}*dat')) if (work_dir and sourcename_mask) else []
        if matching_folders:
            target_folder = matching_folders[0]
            fold_file = os.path.join(target_folder, 'script_fold_raw.txt')
            shutil.copy(fold_file, fold2dir2)

            fold_file_raw = os.path.join(fold2dir2, 'fold_raw.sh')
            is_first_write = True  

            idx = 0
            with open(fold_file, "r") as f:
                for i, line in enumerate(f, start=1):  
                    if i in sorted_numbers:
                        if is_first_write:
                            if os.path.exists(fold_file_raw) and os.path.getsize(fold_file_raw) > 0:
                                backup_file = f"{fold_file_raw}_copy"
                                if os.path.exists(backup_file):
                                    os.remove(backup_file)
                                shutil.copy2(fold_file_raw, backup_file)
                                print(f"检测到原有文件非空，已备份至：{backup_file}")
                                os.remove(fold_file_raw)
                            is_first_write = False

                        savefilenodb(fold_file_raw, line.rstrip('\n'))
            shutil.copy(fold_file_raw, cmd_dir)
            print(f"请运行文件 {fold_file_raw}进行折叠")

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

def copy_aid_png(base_dir):
    if not (cfg_file and os.path.exists(output_file) and work_dir and sourcename and search_label):
        return

    with open(output_file, 'r') as f:
        id_list = [int(line.strip()) for line in f if line.strip().isdigit()]
    if not id_list:
        return

    IF_DDPLAN = parse_config_value(cfg_file, 'IF_DDPLAN') or '0'
    basename_dd_pl = 'dd' if IF_DDPLAN == '1' else 'pl'

    fits_or_dats = ''
    if parse_config_value(cfg_file, 'FLAG_FOLD_TIMESERIES') == '1':
        fits_or_dats += 'dat'
    if parse_config_value(cfg_file, 'FLAG_FOLD_RAWDATA') == '1':
        fits_or_dats = 'fits' if fits_or_dats == '' else f'{fits_or_dats}_fits'
    if parse_config_value(cfg_file, 'FLAG_JERK_SEARCH') == '1':
        fits_or_dats += '__jerk'
    if parse_config_value(cfg_file, 'IF_BARY') == '1':
        fits_or_dats += '__bary'

    basename_only = f'{sourcename_mask}_{basename_dd_pl}_{fits_or_dats}_merged'
    png_src_dir = os.path.join(work_dir, '06_PNG', basename_only)
    if not os.path.exists(png_src_dir):
        print(f"源目录不存在：{png_src_dir}")
        return

    
    digital_seg, middle_seg = extract_path_segments(work_dir)
    if digital_seg and middle_seg:
        target_dir = os.path.join(base_dir, sourcename, digital_seg, middle_seg, search_label)
    else:
        target_dir = os.path.join(base_dir, sourcename, search_label)
    os.makedirs(target_dir, exist_ok=True)

    for target_id in id_list:
        png_files = glob.glob(os.path.join(png_src_dir, f'A{target_id}_*.png')) + glob.glob(os.path.join(png_src_dir, f'A{target_id}_*.PNG'))
        for png in png_files:
            png_name = os.path.basename(png)
            target_png = os.path.join(target_dir, png_name)
            if not os.path.exists(target_png):
                shutil.copy2(png, target_png)
                print(f"复制：{png_name} → {target_dir}")


base_dir = "/home/pengl/pulsar/png-fold"
copy_aid_png(base_dir)