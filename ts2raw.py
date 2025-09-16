#!/usr/bin/env python3
import os
import re
import shutil
import shlex
import string
from datetime import datetime
from psr_fuc import *

def numbers_from_filenames(folder='.'):  #为了保证后续的添加，此处修改为没有sorted
    """
    从指定文件夹中提取所有符合 A<number> 模式的文件名，提取其中的数字并按升序排序。
    
    参数:
        folder (str): 要扫描的文件夹路径，默认为当前目录。
    
    返回:
        List[int]: 提取出的数字按升序排列。
    """
    filenames = os.listdir(folder)
    numbers = []

    for filename in filenames:
        match = re.search(r'A(\d+)', filename)
        if match:
            numbers.append(int(match.group(1)))

    # return sorted(numbers)
    return numbers

def find_cfg_file():
    # 检查当前目录、上级目录和上上级目录的 .cfg 文件
    for folder in [".", "..", "../.."]:
        try:
            for fname in os.listdir(folder):
                if fname.endswith(".cfg"):
                    return os.path.join(folder, fname)
        except FileNotFoundError:
            # 如果目录不存在，跳过
            continue
    return None

def extract_ra_dec(cfg_path):
    ra, dec = None, None
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            # 匹配形如 RA 19:08:26.97 和 DEC +00:10:07.72 的行
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
    """
    从形如 PARAM    VALUE    # 注释 的配置文件中提取指定参数的值。
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # 跳过空行和注释行
            # 去除 # 后注释内容
            line = line.split("#", 1)[0].strip()
            parts = line.split(None, 1)  # 按任意空白字符分割最多两段
            if len(parts) >= 2 and parts[0] == param_name:
                return parts[1].strip()
    return None  # 如果未找到该参数



def write_par_file(source,ra,dec,F0,DM,filename):
    # 转换为十进制度

    with open(filename, "w") as f:
        f.write(f"PSR       {source}\n") 
        f.write(f"RAJ      {ra}\n")
        f.write(f"DECJ     {dec}\n")
        
        #f.write(f"POSEPOCH        58871.0586 \n")
        f.write(f"F0        {F0}\n")    
        f.write(f"F1         0.000000000000D+00 \n")
        f.write(f"DM        {DM}\n") 
        
        # f.write(f" \n")
        # f.write(f" \n")   

def savefilenodb(file,obs):
    with open(file, 'a') as burst_file:

        with open(file, 'r') as read_file:
            existing_obs = set(line.strip() for line in read_file)
        # 检查obs是否已存在，如果不存在则写入
        if obs not in existing_obs:
            burst_file.write(obs + '\n')  


cfg_file = find_cfg_file()
if cfg_file:
    ra, dec = extract_ra_dec(cfg_file)
    print(f"RA: {ra}")
    print(f"DEC: {dec}")
else:
    print("未找到.cfg文件")
# source = parse_config_value(cfg_file, "SOURCE_NAME")
ifbary = parse_config_value(cfg_file, 'IF_BARY')
label = parse_config_value(cfg_file, "SEARCH_LABEL")
work_dir = parse_config_value(cfg_file, "ROOT_WORKDIR")
part_ra = ra.split(":")
part_dec = dec.split((":"))
sign = "" if dec.startswith("-") else "+"
source = part_ra[0] + part_ra[1] + sign + part_dec [0] + part_dec[1]

fold_add = parse_config_value(cfg_file, "PREPFOLD_FLAGS")
current_date = datetime.now().strftime("%Y-%m-%d")
cmd_dir = os.path.join(work_dir,'06_PNG','cmd',current_date)
os.makedirs(cmd_dir,exist_ok=True)

sourcename = parse_config_value(cfg_file, "SOURCE_NAME")
search_label = parse_config_value(cfg_file, "SEARCH_LABEL")
sourcename_mask = sourcename+'_'+search_label


# 判断文件是否存在
folding_dir = os.path.join(work_dir, '05_FOLDING')
if ifbary == '1':
    fold2dir1 = os.path.join(folding_dir, 'raw_prep')
    fold2dir2 = os.path.join(folding_dir, 'raw_fold')
    os.makedirs(fold2dir1,exist_ok=True)

else:
    fold2dir2 = os.path.join(folding_dir, 'raw')
os.makedirs(fold2dir2,exist_ok=True)
output_file = os.path.join(fold2dir2, 'id_list.txt')

if not os.path.exists(output_file):
    numbers = numbers_from_filenames()
    sorted_numbers = sorted((numbers))
    with open(output_file, 'w') as f:
        for number in sorted_numbers:
            f.write(f'{number}\n')

    print(sorted_numbers)
    print(f'已生成，查看并修改：\n{output_file}')
else:
    print(f'已修改{output_file}?\n读取其中的内容...')

    with open(output_file, 'r') as f:
        sorted_numbers = [int(line.strip()) for line in f if line.strip().isdigit()]
    print(sorted_numbers)

    if ifbary == '1':
        matching_folders = glob.glob(os.path.join(folding_dir, f'{sourcename_mask}*'))
        if matching_folders:
            target_folder = matching_folders[0]  #有时可能需要修改

            fold_file = os.path.join(target_folder, 'script_fold_ts.txt')

            shutil.copy(fold_file, fold2dir1)
            shutil.copy(fold_file, fold2dir2)
            shutil.copy(fold_file, cmd_dir)
            print(f"文件已复制到 {fold2dir1}")

        SNR_file = os.path.join(work_dir, '04_SIFTING/cand_sifting.txt')
        shutil.copy(SNR_file, fold2dir1)
        shutil.copy(SNR_file, fold2dir2)
        shutil.copy(SNR_file, cmd_dir)

        maskfile = os.path.join(work_dir, '01_RFIFIND/rfi0.1s_rfifind.mask')
        inputfile = os.path.join(work_dir, 'RAW','*fits')

        fold_file_raw = os.path.join(fold2dir2, 'fold_raw.sh')

        ##两种读取方式:推荐1，除非消色散文件被删除了
        idx = 0
        with open(fold_file, "r") as f:
            for i, line in enumerate(f, start=1):  
                if i in sorted_numbers:
                    idx += 1

                    dm = line.split("-dm")[1].split()[0].strip()
                    accelcand = line.split("-accelcand")[1].split()[0].strip()
                    accelfile = line.split("-accelfile")[1].split()[0].strip()
                    #outname = line.split("-o")[1].split()[0].strip()
                    datafile = line.strip().split()[-1]

                    txtcand = os.path.splitext(accelfile)[0] #+ ".txtcand"
                    print(txtcand,accelcand)
                    
                    with open(txtcand, 'r') as file_cand:
                        infos = file_cand.readlines()
                        line_info = infos[int(accelcand)+2].strip()

                        print(line_info)
                        fields = line_info.split() 
                        cand = fields[0]
                        sigma = fields[1]
                        power = fields[2]
                        num_power = fields[3]
                        harm = fields[4]
                        period = fields[5]
                        frequency = fields[6]
                        period_clean = period[:period.find("(")]
                        frequency_clean = frequency[:frequency.find("(")]
                        fft_r = fields[7]
                        freq_deriv = fields[8]
                        fft_z = fields[9]
                        accel = fields[10]
                        notes = ' '.join(fields[11:])
                        print(period_clean)

                        type_par = string.ascii_uppercase[(idx - 1) % 26]  # 自动循环回 A-Z

                        # type_par = f"{chr(64 + idx)}"
                        outname = f'{type_par}{i}DM{dm}_{period_clean}ms'
                        parname = os.path.join(fold2dir1,f'{type_par}{i}.par')
                        write_par_file(source,ra,dec,frequency_clean,dm, parname)

                        cmd = f'prepfold -topo {fold_add} -noxwin  -par {parname} -mask {maskfile} -o {outname} {work_dir}/RAW/*fits'
                        savefilenodb(fold_file_raw,cmd)
                        shutil.copy(fold_file_raw, cmd_dir)

    else:
        matching_folders = glob.glob(os.path.join(work_dir, '06_PNG', f'{sourcename_mask}*dat'))
        if matching_folders:
            target_folder = matching_folders[0]  

            fold_file = os.path.join(target_folder, 'script_fold_raw.txt')
        
            shutil.copy(fold_file, fold2dir2)
            # print(f"文件已复制到 {fold2dir2}")

            fold_file_raw = os.path.join(fold2dir2, 'fold_raw.sh')

            idx = 0
            with open(fold_file, "r") as f:
                for i, line in enumerate(f, start=1):  
                    if i in sorted_numbers:
                        savefilenodb(fold_file_raw,line.rstrip('\n'))
            shutil.copy(fold_file_raw, cmd_dir)
            print(f"请运行文件 {fold_file_raw}进行折叠")

                




# with open(SNR_file, "r") as f:
#     lines = f.readlines()
#     n = 0
#     for line in lines:
#         if line.startswith('#'):
#             pass
#         else:
#             parts = line.split()
#             candfile = parts[0]
#             cand_file = candfile.split(":")[0]
#             candnum = int(candfile.split(":")[-1])
#             dm = float(parts[1])
#             dm ="{:.2f}".format(dm)
#             snr = float(parts[2])
#             sigma = float(parts[3])
#             num_harm = int(parts[4])
#             ipow = float(parts[5])
#             cpow = float(parts[6])
#             p_ms = float(parts[7])
#             r = float(parts[8])
#             z = float(parts[9])
#             num_hits = int(parts[10][1:-1])
#             n += 1
#             if n in sorted_numbers:
#                 print(n)
#                 print(dm)