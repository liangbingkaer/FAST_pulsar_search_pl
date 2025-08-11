#!/usr/bin/env python3

import os,sys
import argparse
import glob
from psr_fuc import *

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

def fold_task(cmd, ifok,logfile, work_dir,png_dir):
    whitelist = []
    filename = os.path.basename(ifok)
    png_name = f'{filename[:-4]}.png'
    ps_path = os.path.join(work_dir,f'{filename[:-4]}.ps')
    """子任务执行函数"""
    run_cmd(cmd, ifok = ifok, work_dir=work_dir,log_file=logfile,mode='both')  #根据ifok判断是否运行cmd
    ps2png(ps_path)
    handle_files(work_dir, png_dir, 'copy',png_name )

def pool_fold(num_processes, task_name, cmd_list, ifok_list,log_list, work_dir=os.getcwd(),png_dir = None):
    """
    改进的多进程任务调度函数
    
    Args:
        num_processes (int): 并行进程数
        task_name (str): 任务名称（用于进度条显示）
        cmd_list (list): 要执行的命令列表
        ifok_list (list): 布尔值列表，控制是否执行对应命令
        work_dir (str): 工作目录路径
    """
    # 参数合法性校验
    if len(cmd_list) != len(ifok_list):
        raise ValueError("cmd_list和ifok_list长度必须一致")

    # 初始化进度条和线程锁
    progress_bar = tqdm(
        total=len(cmd_list),
        desc=f"{task_name}-{num_processes}核",
        unit="cmd",
        dynamic_ncols=True,
        # position=0
    )
    # lock = Lock()

    def update(*args):
        progress_bar.update()
    
    def handle_error(error):
        """统一错误处理函数"""
        progress_bar.write(f"任务执行错误: {error}")

    # 创建进程池并提交任务
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

cfg_file = find_cfg_file()
current_path = os.getcwd()
maskstr = os.path.basename(current_path)

work_dir = parse_config_value(cfg_file, "ROOT_WORKDIR")
log_dir = os.path.join(work_dir,'LOG',f'ts2raw_{maskstr}')
os.makedirs(log_dir,exist_ok=True)
png_dir = os.path.join(work_dir,'06_PNG',maskstr)
os.makedirs(png_dir,exist_ok=True)

ncpus = 4
cmdfile_name = "fold_raw.sh"
if not os.path.isfile(cmdfile_name):
    print(f"当前文件夹中不存在文件：{cmdfile_name}")
    
if (len(sys.argv) == 1 or ("-h" in sys.argv) or ("-help" in sys.argv) or ("--help" in sys.argv)):
    print("Usage: %s -cmdfile \"commands.sh\" -ncpus N " % (os.path.basename(sys.argv[0])))

    print('将使用默认值')
else:
    for j in range(1, len(sys.argv)):
        if (sys.argv[j] == "-cmdfile"):
            cmdfile_name = sys.argv[j+1]
            print(f"使用指定的：{cmdfile_name},注意不要使用绝对路径，这会导致折叠信息mask混淆")
        elif (sys.argv[j] == "-ncpus"):
            ncpus = int(sys.argv[j+1])


with open(cmdfile_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()
cmd_list = [line.strip() for line in lines]
cmd_list = sorted(cmd_list)

ifok_files = []
logfiles = []
for line in cmd_list:
    nsub = line.split("-nsub")[1].strip().split()[0] if "-nsub" in line else None
    n = line.split("-n")[1].strip().split()[0] if "-n " in line else None  # 注意避免匹配 "-noxwin"
    parfile = line.split("-par")[1].strip().split()[0] if "-par" in line else None
    maskfile = line.split("-mask")[1].strip().split()[0] if "-mask" in line else None
    outname = line.split("-o")[1].strip().split()[0] if "-o" in line else None
    datafile = line.strip().split()[-1]
    ps_files = glob.glob(f"{outname}*.ps") 

    if ps_files:
        # ps_file = os.path.join(log_dir,ps_files[0])
        png_file = os.path.join(png_dir,ps_files[0][:-3] + '.png')
        ifok_files.append(png_file)
    else:
        ifok_files.append('/test/whysohard')

    logfile = os.path.join(log_dir,f'LOG_{outname}.txt')
    logfiles.append(logfile)

ncpus = min(ncpus,len(cmd_list))
print(ifok_files)
pool_fold(ncpus,'fold',cmd_list,ifok_files,logfiles,work_dir = os.getcwd(),png_dir=png_dir)


ifok_files = []
logfiles = []
for line in cmd_list:
    nsub = line.split("-nsub")[1].strip().split()[0] if "-nsub" in line else None
    n = line.split("-n")[1].strip().split()[0] if "-n " in line else None  # 注意避免匹配 "-noxwin"
    parfile = line.split("-par")[1].strip().split()[0] if "-par" in line else None
    maskfile = line.split("-mask")[1].strip().split()[0] if "-mask" in line else None
    outname = line.split("-o")[1].strip().split()[0] if "-o" in line else None
    datafile = line.strip().split()[-1]
    ps_files = glob.glob(f"{outname}*.ps") 

    if ps_files:
        # ps_file = os.path.join(log_dir,ps_files[0])
        png_file = os.path.join(png_dir,ps_files[0][:-3] + '.png')
        ifok_files.append(png_file)
    else:
        ifok_files.append('/test/whysohard')

    logfile = os.path.join(log_dir,f'LOG_{outname}.txt')
    logfiles.append(logfile)

ncpus = min(ncpus,len(cmd_list))
print(ifok_files)
pool_fold(ncpus,'fold',cmd_list,ifok_files,logfiles,work_dir = os.getcwd(),png_dir=png_dir)