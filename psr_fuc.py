
import sys
import os,re
import os.path
import glob
import subprocess
from tqdm import tqdm
from multiprocessing import Pool,Lock, cpu_count
import shlex
import shutil
import copy
import random
import time
from datetime import datetime,timedelta
import numpy as np
import urllib
from presto import filterbank, infodata, parfile, psr_utils, psrfits, rfifind, sifting
from multiprocessing.pool import ThreadPool
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import base64

cwd = os.getcwd()

class colors:
    HEADER = '\033[95m'      # 亮紫色（Magenta），通常用于标题或重要提示
    OKBLUE = '\033[94m'      # 亮蓝色，用于正常信息或状态提示
    OKCYAN = '\033[96m'      # 亮青色（Cyan），用于正常信息或状态提示
    OKGREEN = '\033[92m'     # 亮绿色，通常用于表示成功或正常状态
    WARNING = '\033[93m'     # 亮黄色，用于警告信息
    ERROR = '\033[91m'       # 亮红色，用于错误信息
    BOLD = '\033[1m'         # 加粗文本（不改变颜色），使文本更突出
    ENDC = '\033[0m'     # 重置文本格式（包括颜色和加粗等），恢复默认显示

def makedir(*dirs): 
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def write2file(content, file_path, add_newline=True):
    """将消息写入指定文件"""
    with open(file_path, 'a') as f:
        if add_newline:
            f.write(content + "\n")
        else:
            f.write(content)

def print_log(*args, sep=' ', end='\n', file=None, flush=False, log_files=None, masks=None, color=None, mode='both'):
    """
    打印并记录日志，支持高亮显示特定内容或整个消息。
    masks: 需要高亮显示的内容列表（如果为 None 且 color 有值，则整个消息高亮）
    color: 高亮显示的颜色代码（如 colors.ERROR）
    - mode: 决定操作模式：
        - 'w'：仅写入文件
        - 'p'：仅打印到控制台
        - 'both'：同时写入文件并打印到控制台（默认）
    """
    default_dir = os.path.join(cwd, 'logall.txt')
    if log_files is None:
        log_files = [default_dir]
    elif isinstance(log_files, str):
        log_files = [log_files, default_dir]
    else:
        log_files = list(log_files) + [default_dir]

    message = sep.join(str(arg) for arg in args) + end

    if mode in ['w', 'both']:
        for file_path in log_files:
            write2file(message, file_path, add_newline=False)

    highlighted_message = message
    if color:
        if masks:
            if isinstance(masks, (str, bytes)):
                masks = [masks]  # 如果传入单个字符串，转为列表
            for mask in masks:
                highlighted_message = highlighted_message.replace(str(mask), f"{color}{mask}{colors.ENDC}")
        else:
            highlighted_message = f"{color}{message}{colors.ENDC}"

    if mode in ['p', 'both']:
        print(highlighted_message, end='', file=file or sys.stdout, flush=flush)

def format_execution_time(execution_time):
    """将执行时间格式化为易读的字符串"""
    delta = timedelta(seconds=execution_time)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}天{hours}小时{minutes}分钟"
    elif hours > 0:
        return f"{hours}小时{minutes}分钟"
    elif minutes > 0:
        return f"{minutes}分钟{seconds:.1f}秒"
    else:
        return f"{seconds:.1f}秒"

def time_consum(start_time,cmd,mode='both'):
    """计算并记录执行时间"""
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = format_execution_time(execution_time)
    
    log_message = f"运行时间： {execution_time_str}"
    print_log(f'运行命令：{cmd}',masks=cmd,color=colors.OKCYAN,log_files='logruntime.txt',mode=mode)
    print_log(log_message,masks=log_message,color=colors.OKBLUE,log_files='logruntime.txt',mode=mode)
    time.sleep(2)

# 记录程序开始和结束
def get_current_time_to_minute():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def print_now():
      time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
      print_log(f'\n时间戳：{time_str}')

def print_program_message(phase):
    cwd = os.getcwd()
    current_time = get_current_time_to_minute()
    if phase == 'start':
        print_log('\n\n\n**************************************程序开始***********************************\n',masks='程序开始',color=colors.HEADER)
        print_log('本次程序运行开始时间为：' + current_time)
        print_header()
        print_log(f'程序开始：当前路径{cwd}\n')
        time.sleep(1)
    elif phase == 'end':
        print_log('本次程序运行结束时间为：' + current_time)
        print_log('\n**************************************退出程序***********************************\n\n\n',masks='退出程序',color=colors.HEADER)
        sys.exit(0)

def print_header():
    print_log(f' FAST DATA REDUCTION PIPELINE '.center(80, '-'))
    print_log(' Author: Long Peng '.center(80, ' '))
    print_log(f' See web page: https://www.plxray.cn/ '.center(80, ' '), masks='https://www.plxray.cn/',color=colors.OKGREEN)
    print_log(' Script created on Feb 2025'.center(80, ' '))
    print_log('This program is adapted from: https://github.com/alex88ridolfi/PULSAR_MINER.'.center(80, ' '))
    print_log('--'.center(80, '-'))
    time.sleep(1)

def append_to_script_if_not_exists(file_path, content):
    """
    如果内容不存在，则追加内容。如果文件不存在，则创建文件。
    """
    # 检查文件是否存在，如果不存在则创建
    if not os.path.exists(file_path):
        open(file_path, 'w').close()  # 创建一个空文件

    with open(file_path, 'r+') as file:
        existing_content = file.read()
        if content not in existing_content:
            file.write(content+'\n')

def run_cmd(cmd, ifok=None, work_dir=None, log_file=None, dict_envs={}, flag_append=True,mode='both'):
    """
    执行命令并记录日志，支持条件执行、目录切换、环境变量设置和日志记录。

    Args:
        cmd (str): 要执行的命令。
        ifok (str, optional): 文件路径。如果文件存在，则跳过命令。默认为None。
        work_dir (str, optional): 命令执行的目录。默认为None。
        log_file (str, optional): 日志文件路径。默认为None。
        dict_envs (dict, optional): 自定义环境变量。默认为空字典。
        flag_append (bool, optional): 是否追加日志。默认为False。
    """
    if ifok and os.path.isfile(ifok):
        print_log(f'File {ifok} exists. Skipping command: {cmd}', log_file,mode=mode)
        return

    start_time = time.time()
    datetime_start = datetime.now().strftime("%Y/%m/%d %H:%M")
    global cwd  # 声明使用全局变量 cwd

    if work_dir:
        os.chdir(work_dir)
    else:
        work_dir = cwd

    log_mode = "a" if flag_append else "w"
    log_handle = open(log_file, log_mode) if log_file else None

    print_log(f'程序运行路径为: {work_dir}',mode=mode)
    print_log(f'日志文件为：{log_file}',mode=mode)
    #print_log(f'运行命令：{cmd}\n', log_file,masks=cmd,color=colors.OKCYAN)

    if log_handle:
        log_handle.write(f"****************************************************************\n")
        log_handle.write(f"开始日期和时间：{datetime_start}\n")
        log_handle.write(f"命令：{cmd}\n")
        log_handle.write(f"工作目录：{work_dir}\n")
        log_handle.write(f"****************************************************************\n")
        log_handle.flush()

    env = os.environ.copy()
    env.update(dict_envs)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=work_dir)
    stdout, stderr = proc.communicate()

    if log_handle:
        log_handle.write(stdout.decode())
        log_handle.write(stderr.decode())
    else:
        print(stdout.decode())
        if stderr:
            print_log(f"Error: {stderr.decode()}", log_file,color=colors.ERROR)

    datetime_end = datetime.now().strftime("%Y/%m/%d %H:%M")
    execution_time = time.time() - start_time

    if log_handle:
        log_handle.write(f"\n结束日期和时间：{datetime_end}\n")
        log_handle.write(f"总耗时：{execution_time:.2f} 秒\n")
        log_handle.close()

    append_to_script_if_not_exists(os.path.join(work_dir, 'cmd.sh'),f'#程序运行路径为: {work_dir}\n{cmd}\n')
    time_consum(start_time,cmd=cmd,mode=mode)

    if ifok and ifok.endswith(('.txt', '.ifok')):
        with open(ifok, 'a') as f:
            f.write(f"#Command executed:\n {cmd}\n")
    os.chdir(cwd)

    if ifok and os.path.isfile(ifok):
        cmd_name = cmd.split()[0]
        ifok_dir = os.path.join(cwd, '00_IFOK', 'rm_script')
        script_name = f"rm_{cmd_name}.sh"
        os.makedirs(ifok_dir, exist_ok=True)
        rm_script_path = os.path.join(ifok_dir, script_name)

        append_to_script_if_not_exists(rm_script_path, "#!/bin/bash\n")
        append_to_script_if_not_exists(rm_script_path, f"rm -f {ifok}\n")
        append_to_script_if_not_exists(rm_script_path, f"echo 'Deleted {ifok}'\n")
        os.chmod(rm_script_path, 0o755)
        #print_log(f"Created deletion script: {rm_script_path}\n", log_file)



def check_presto_path(presto_path, key):
    # 检查 PRESTO 路径是否存在
    if os.path.exists(presto_path):
        # 检查 accelsearch 是否存在于 bin 目录下
        if os.path.exists(presto_path + "/bin/accelsearch"):
            return True
        else:
            # 如果 accelsearch 不存在，打印错误信息并退出
            print("%s错误%s：%s 目录 '%s' 存在，但我在 %s/bin 中找不到 'accelsearch'！" % (colors.ERROR + colors.BOLD, colors.ENDCOLOR, key, presto_path, presto_path))
            print("请确保您的 %s 安装路径正确且功能正常。" % (key))
            exit()
    else:
        # 如果 PRESTO 路径不存在，打印错误信息并退出
        print("%s错误：%s：%s 目录 '%s' 不存在！" % (colors.ERROR + colors.BOLD, colors.ENDCOLOR, key, presto_path))
        print("请确保配置文件中 %s 的路径设置正确。" % (key))
        exit()

def check_if_enough_disk_space(root_workdir, num_DMs, T_obs_s, t_samp_s, flag_remove_fftfiles):
    # 获取根工作目录的磁盘使用情况
    disk_space = shutil.disk_usage(root_workdir)
    disk_space_free_bytes = disk_space.free  # 可用磁盘空间（字节）

    # 计算全长度数据的采样点数和数据文件大小
    N_samples_per_datfile_full = int(T_obs_s / t_samp_s)  # 全长度数据的采样点数
    datfile_full_size_bytes = N_samples_per_datfile_full * 4  # 全长度数据文件大小（字节）

    # 如果不删除 .fft 文件，则每个 DM 试验占用的空间将翻倍
    if flag_remove_fftfiles == 0:
        datfile_full_size_bytes = datfile_full_size_bytes * 2

    # 计算全长度搜索所需的磁盘空间
    full_length_search_size_bytes = num_DMs * datfile_full_size_bytes

    if flag_remove_fftfiles == 0:
        print_log("是否删除 .fft 文件？否  --> 每个 DM 试验将占用双倍空间")
    else:
        print_log("是否删除 .fft 文件？是")

    size_G = f"{full_length_search_size_bytes / 1.0e9:4.2f}" 
    print_log(f"全长度搜索：~{size_G} GB       ({num_DMs} DM 试验 * 每次试验 {datfile_full_size_bytes / 1.0e6:5.0f} MB)",masks=size_G,color=colors.OKGREEN)

    # 初始化总搜索所需空间
    total_search_size_bytes = full_length_search_size_bytes

    # 输出预估的磁盘空间使用情况和可用磁盘空间
    size_G = f"{1.1 * total_search_size_bytes / 1.0e9:5.2f}" 
    size_G_total = f"{disk_space_free_bytes / 1.0e9:5.2f}" 
    print_log(f"预期磁盘空间使用量：~{size_G} GB",masks=size_G,color=colors.OKGREEN)

    if disk_space_free_bytes > 1.1 * total_search_size_bytes:
        print_log(f"可用磁盘空间：~{size_G_total} GB   --> 太好了！磁盘空间足够。",masks=size_G_total,color=colors.OKGREEN)
        return True
    else:
        print_log(f"可用磁盘空间：~{size_G_total} GB   --> 哎呀！磁盘空间不足！ ",masks=size_G_total,color=colors.ERROR)
        return False

def return_all_par_files(pulsar_list_file):
    """
    从脉冲星列表文件中读取脉冲星名称，并下载对应的 .par 文件。
    如果文件已存在，则跳过下载。
    """
    if not os.path.exists(pulsar_list_file):
        print(f"未找到脉冲星列表文件: {pulsar_list_file}")
        exit(1)

    output_dir = os.path.dirname(pulsar_list_file)
    with open(pulsar_list_file, "r") as f:
        lines = f.readlines()

    # 跳过表头和分隔线
    pulsars = []
    for line in lines:
        line = line.strip()
        if line.startswith("PSRJ") or line.startswith("-") or not line:
            continue
        # 提取脉冲星名称，假设名称位于每行的开头
        try:
            pulsar_name = line.split()[1]  # 假设名称是第二列
            if pulsar_name.startswith("J"):  # 确保是有效的脉冲星名称
                pulsars.append(pulsar_name)
        except IndexError:
            continue  # 跳过格式不正确的行

    cmd_list = []
    par_path_list = []
    for pulsar in pulsars:
        par_file = f"{pulsar}.par"
        par_path = os.path.join(output_dir,par_file)
        cmd = f"psrcat -e {pulsar} > {par_path}"
        # psrcat -e 0953+0755 > J0953+0755.par
        cmd_list.append(cmd)
        par_path_list.append(par_path)
    return cmd_list,par_path_list
        

###多线程函数最终优化版
#需要参数：进程池数，总进程名，cmd列表，判断是否需要运行的文件列表
def child_task(cmd, ifok,logfile, work_dir):
    """子任务执行函数"""
    run_cmd(cmd, ifok = ifok, work_dir=work_dir,log_file=logfile,mode='both')  #根据ifok判断是否运行cmd

def pool(num_processes, task_name, cmd_list, ifok_list, log_list=None, work_dir=os.getcwd()):
    """
    改进的多进程任务调度函数
    
    Args:
        num_processes (int): 并行进程数
        task_name (str): 任务名称（用于进度条显示）
        cmd_list (list): 要执行的命令列表
        ifok_list (list): 布尔值列表，控制是否执行对应命令
        log_list (list, optional): 日志文件名列表或布尔值列表，默认为与 cmd_list 长度相同的 False 列表
        work_dir (str): 工作目录路径
    """
    # 参数合法性校验
    if len(cmd_list) != len(ifok_list):
        raise ValueError("cmd_list 和 ifok_list 长度必须一致")
    
    # 如果未提供 log_list，则生成默认的 False 列表
    if log_list is None:
        log_list = [False] * len(cmd_list)
    elif len(cmd_list) != len(log_list):
        raise ValueError("cmd_list 和 log_list 长度必须一致")

    # 初始化进度条和线程锁
    progress_bar = tqdm(
        total=len(cmd_list),
        desc=f"{task_name}-{num_processes}核",
        unit="cmd",
        dynamic_ncols=True,
    )

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
                child_task,
                args=(cmd, ifok, log_file, work_dir),
                callback=update,
                error_callback=handle_error
            )
            for cmd, ifok, log_file in zip(cmd_list, ifok_list, log_list)
        ]
        process_pool.close()
        process_pool.join()
    except Exception as e:
        process_pool.terminate()
        raise e
    finally:
        progress_bar.close()

def handle_files(directory, to_dir, action, pattern, whitelist=None):
    """
    根据指定操作（复制、移动或删除）处理匹配的文件。
    
    参数：
    - directory：源目录。
    - to_dir：目标目录（适用于复制或移动操作）。
    - action：操作类型："copy"、"move" 或 "delete"。
    - pattern：文件名或通配符（如 "*.png"）。
    - whitelist：白名单文件列表（可选），其中的文件将被排除。
    """
    if whitelist is None:
        whitelist = []

    # 获取匹配的文件路径列表
    full_pattern = os.path.join(directory, pattern)
    matching_files = glob.glob(full_pattern)

    for file_path in matching_files:
        file_name = os.path.basename(file_path)
        if file_name in whitelist:
            continue

        if action == "delete":
            os.remove(file_path)
        elif action == "move":
            destination_path = os.path.join(to_dir, file_name)
            shutil.move(file_path, destination_path)
        elif action == "copy":
            destination_path = os.path.join(to_dir, file_name)
            if not os.path.exists(destination_path):
                shutil.copy(file_path, destination_path)

    print(f"文件 {action} 操作成功")

def extract_dm_part(file_path):
    pattern = r"DM(.*?)(?=\.\w+$)"
    filename = os.path.basename(file_path)
    # 使用正则表达式提取DM之后的部分
    DM = re.search(pattern, filename)
    if DM:
        DM_value = DM.group(1)
        # 将匹配到的部分转换为浮点数，并格式化为 "05.2f"
        DM_formatted = f"{float(DM_value):05.2f}"
        return DM_formatted
    else:
        return None  # 如果没有匹配到，返回None

# PREPDATA：对原始 FITS 文件进行去除 RFI、去除通道等预处理，生成 .dat 和 .inf 文件
def prepdata(infile,sourcename, out_dir,ifok_dir, log_path, DM, Nsamples=0, ignorechan_list="",
             mask="", downsample_factor=1, reference="barycentric", other_flags="",
             presto_env=os.environ['PRESTO']):

    outfile_basename = f"{sourcename}_DM{DM:05.2f}"
    datfile_abspath = os.path.join(out_dir, f"{outfile_basename}.dat")
    inffile_abspath = os.path.join(out_dir, f"{outfile_basename}.inf")

    # 构造 prepdata 参数
    if reference =="topocentric":
            flag_nobary = "-nobary "
    elif reference =="barycentric":
            flag_nobary = ""
    else:
            print_log("ERROR: 无效的参考系选项: \"%s\"" % (reference),color=colors.ERROR)
            sys.exit()

    flag_numout = ""
    if Nsamples > 0:
        numout = make_even_number(int(Nsamples / float(downsample_factor)))
        flag_numout = f"-numout {numout} "

    flag_mask = f"-mask {mask} " if mask else ""
    flag_ignorechan = f"-ignorechan {ignorechan_list} " if ignorechan_list else ""

    cmd_prepdata = f"prepdata -o {outfile_basename} {flag_ignorechan}{flag_mask}{flag_nobary}{other_flags}{flag_numout} -dm {DM} -downsamp {downsample_factor} {infile}"

    # 如果输出文件已存在，跳过执行，直接提示
    if os.path.exists(datfile_abspath) and os.path.exists(inffile_abspath):
        print_log(f"\n警告：文件 '{outfile_basename}.dat' 和 '.inf' 已存在，跳过处理并检查结果。", color=colors.WARNING,mode='p')
    else:
        # 设置环境变量并执行命令
        dict_env = {
            'PRESTO': presto_env,
            'PATH': f"{presto_env}/bin:{os.environ['PATH']}",
            'LD_LIBRARY_PATH': f"{presto_env}/lib:{os.environ['LD_LIBRARY_PATH']}"
        }
        ifokfile = os.path.join(ifok_dir,f'prepdata-{DM}.ifok')
        run_cmd(cmd_prepdata, ifok=ifokfile, work_dir=out_dir,
                log_file=log_path, dict_envs=dict_env, flag_append=None)

        # 检查文件是否成功生成
        if os.path.exists(datfile_abspath) and os.path.exists(inffile_abspath):
            print(f"{datetime.now().strftime('%Y/%m/%d  %H:%M')} | prepdata 处理 '{sourcename}' 成功！")
        else:
            print(f"警告 ({datetime.now().strftime('%Y/%m/%d  %H:%M')}) | prepdata 未成功生成所有输出文件：'{sourcename}'")

def prepdata2bary(infile,sourcename, out_dir,ifok_dir, log_dir, Nsamples=0, ignorechan_list="",
             mask="", downsample_factor=1, other_flags="",presto_env=os.environ['PRESTO']):

    cmd_list =[]
    ifok_list = []
    log_list = []
    for dat in infile:
        DM = extract_dm_part(dat)
        outfile_basename = f"{sourcename}_DM{DM}"
        datfile_abspath = os.path.join(out_dir, f"{outfile_basename}.dat")  #ifokfile
        inffile_abspath = os.path.join(out_dir, f"{outfile_basename}.inf")
        log_path = os.path.join(log_dir,f'BARY-{DM}.txt')
        ifok_path = os.path.join(ifok_dir,f'BARY-{DM}.ifok')

        # 构造 prepdata 参数
        flag_nobary = " " #-nobary

        flag_numout = ""
        if Nsamples > 0:
            numout = make_even_number(int(Nsamples / float(downsample_factor)))
            flag_numout = f"-numout {numout} "

        flag_mask = f"-mask {mask} " if mask else ""
        flag_ignorechan = f"-ignorechan {ignorechan_list} " if ignorechan_list else ""

        cmd_prepdata = f"prepdata -o {outfile_basename} {flag_ignorechan}{flag_mask}{flag_nobary}{other_flags}{flag_numout} -dm {DM} -downsamp {downsample_factor} {dat}"
        cmd_list.append(cmd_prepdata)
        # ifok_list.append(datfile_abspath)
        ifok_list.append(ifok_path)

        log_list.append(log_path)
        file_script_prepdata_abspath = "%s/%s" % (out_dir, 'prepdata_script.txt')
        write2file(cmd_prepdata,file_script_prepdata_abspath)

    return cmd_list,ifok_list,log_list

         

def dedisperse(infile,open_mask,sourcename, out_dir, log_dir, ignorechan_list, mask_file, list_DD_schemes, nchan, nsubbands=0, other_flags="", presto_env=os.environ['PRESTO']):

        global cwd
        prepsubband_outfilename = sourcename
        # 设置环境变量，确保 PRESTO 工具链的路径正确
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
        # 定义 prepsubband 脚本文件的名称和绝对路径
        file_script_prepsubband_name = f"script_prepsubband_{open_mask}.txt"
        file_script_prepsubband_abspath = "%s/%s" % (out_dir, file_script_prepsubband_name)
        
        N_schemes = len(list_DD_schemes)

        string_mask = ""
        if mask_file != "":
                string_mask = "-mask %s" % (mask_file)
        string_ignorechan = ""
        if ignorechan_list != "":
                string_ignorechan = "-ignorechan %s" % (ignorechan_list)

        # 打印分隔符和去色散方案信息
        print_log("----------------------------------------------------------------------")
        print_log("prepsubband 将运行 %d 次，使用以下 DM 范围：" % (N_schemes))
        print_log()
        print_log("%10s %10s %10s %10s %10s " % ("低 DM", "高 DM", "dDM",  "下采样",   "DM 数量"))
        for i in range(N_schemes):
                offset = 0
                if i == N_schemes-1 : offset = 1
                print_log("%10.3f %10.3f %10s %10s %10d " % (list_DD_schemes[i]['loDM'], np.float64(list_DD_schemes[i]['loDM']) + int(list_DD_schemes[i]['num_DMs'])*np.float64(list_DD_schemes[i]['dDM']), list_DD_schemes[i]['dDM'],  list_DD_schemes[i]['downsamp'],  list_DD_schemes[i]['num_DMs'] + offset))
        print_log("----------------------------------------------------------------------")
        sys.stdout.flush()
        
        # 检查子带数量是否合理
        if nsubbands == 0:
                nsubbands = nchan
        elif (nchan % nsubbands != 0):
                print_log("错误：请求的子带数量为 %d，这不是通道数量 %d 的整数倍！" % (nsubbands, nchan),color=colors.ERROR)
                exit()

        # 打开 prepsubband 脚本文件进行写入
        list_prepsubband_commands = []
        file_script_prepsubband = open(file_script_prepsubband_abspath, "w")
        print_log(f"使用 {nsubbands} 个子带进行去色散（原始通道数量：{nchan}）",color=colors.WARNING)


        for i in range(N_schemes):
                info_str = f'{i+1}/{N_schemes}'
                print_log(f'第{info_str}次去色散',masks=info_str,color=colors.BOLD)
                # 获取当前方案的低 DM、DM 步长和高 DM
                loDM = np.float64(list_DD_schemes[i]['loDM'])
                dDM  = np.float64(list_DD_schemes[i]['dDM'])
                hiDM = loDM + int(list_DD_schemes[i]['num_DMs'])*dDM

                LOG_basename="03_prepsubband_%s%s" % (open_mask,i)
                log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
                ifok_path = cwd+f'/ok-prepsubband-{open_mask}{i}.ifok'

                if N_schemes == 1:
                        print_log(f"提示：使用 'tail -f {log_abspath}' 查看 prepsubband 的进度",masks=f'tail -f {log_abspath}',color=colors.OKCYAN)
                elif N_schemes > 1:
                        print_log(f"提示：使用 'for f in {log_dir}/LOG_prepsubband_*.txt; do tail -1 ${{f}}; echo; done' 查看 prepsubband 的进度",masks=f'for f in {log_dir}/LOG_prepsubband_*.txt; do tail -1 ${{f}}; echo; done',color=colors.OKCYAN)

                flag_numout = ""
                if i < N_schemes-1:
                        # 构造 prepsubband 命令（非最后一个方案）
                        cmd_prepsubband = "prepsubband -nobary %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'], list_DD_schemes[i]['downsamp'], nsubbands, infile)
                elif i == N_schemes-1:
                        # 构造 prepsubband 命令（最后一个方案，DM 数量加 1）
                        cmd_prepsubband = "prepsubband -nobary %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'] + 1, list_DD_schemes[i]['downsamp'], nsubbands, infile)
                # 将命令写入脚本文件
                run_cmd(cmd_prepsubband,ifok=ifok_path,work_dir=out_dir,log_file=log_abspath,dict_envs=dict_env,flag_append=None)
                file_script_prepsubband.write("%s\n" % cmd_prepsubband)

        # 关闭脚本文件
        file_script_prepsubband.close()

def dedisperse2cmd(infile,open_mask,sourcename, out_dir, log_dir, ignorechan_list, mask_file, list_DD_schemes, nchan, nsubbands=0, other_flags="", presto_env=os.environ['PRESTO']):

        global cwd
        
        prepsubband_outfilename = sourcename
        # 设置环境变量，确保 PRESTO 工具链的路径正确
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
        # 定义 prepsubband 脚本文件的名称和绝对路径
        file_script_prepsubband_name = f"script_prepsubband_{open_mask}.txt"
        file_script_prepsubband_abspath = "%s/%s" % (out_dir, file_script_prepsubband_name)
        
        N_schemes = len(list_DD_schemes)

        string_mask = ""
        if mask_file != "":
                string_mask = "-mask %s" % (mask_file)
        string_ignorechan = ""
        if ignorechan_list != "":
                string_ignorechan = "-ignorechan %s" % (ignorechan_list)

        # 打印分隔符和去色散方案信息
        print_log("----------------------------------------------------------------------")
        print_log("prepsubband 将运行 %d 次，使用以下 DM 范围：" % (N_schemes))
        print_log()
        print_log("%10s %10s %10s %10s %10s " % ("低 DM", "高 DM", "dDM",  "下采样",   "DM 数量"))
        for i in range(N_schemes):
                offset = 0
                if i == N_schemes-1 : offset = 1
                print_log("%10.3f %10.3f %10s %10s %10d " % (list_DD_schemes[i]['loDM'], np.float64(list_DD_schemes[i]['loDM']) + int(list_DD_schemes[i]['num_DMs'])*np.float64(list_DD_schemes[i]['dDM']), list_DD_schemes[i]['dDM'],  list_DD_schemes[i]['downsamp'],  list_DD_schemes[i]['num_DMs'] + offset))
        print_log("----------------------------------------------------------------------")
        sys.stdout.flush()
        
        # 检查子带数量是否合理
        if nsubbands == 0:
                nsubbands = nchan
        elif (nchan % nsubbands != 0):
                print_log("错误：请求的子带数量为 %d，这不是通道数量 %d 的整数倍！" % (nsubbands, nchan),color=colors.ERROR)
                exit()

        # 打开 prepsubband 脚本文件进行写入
        list_prepsubband_commands = []
        file_script_prepsubband = open(file_script_prepsubband_abspath, "w")
        print_log(f"使用 {nsubbands} 个子带进行去色散（原始通道数量：{nchan}）",color=colors.WARNING)

        print_log(f"提示：使用 'for f in {log_dir}/LOG_prepsubband_*.txt; do tail -1 ${{f}}; echo; done' 查看 prepsubband 的进度",masks=f'for f in {log_dir}/LOG_prepsubband_*.txt; do tail -1 ${{f}}; echo; done',color=colors.OKCYAN)

        cmd_prepsubband_list = []
        ifok_list = []
        log_list = []

        for i in range(N_schemes):
                info_str = f'{i+1}/{N_schemes}'
                print_log(f'第{info_str}次去色散',masks=info_str,color=colors.BOLD)
                # 获取当前方案的低 DM、DM 步长和高 DM
                loDM = np.float64(list_DD_schemes[i]['loDM'])
                dDM  = np.float64(list_DD_schemes[i]['dDM'])
                hiDM = loDM + int(list_DD_schemes[i]['num_DMs'])*dDM

                LOG_basename="03_prepsubband_%s%s" % (open_mask,i)
                log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
                ifok_path = cwd+f'/00_IFOK/ok-prepsubband-{open_mask}{i}.ifok'

                flag_numout = ""
                if i < N_schemes-1:
                        # 构造 prepsubband 命令（非最后一个方案）
                        cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'], list_DD_schemes[i]['downsamp'], nsubbands, infile)
                elif i == N_schemes-1:
                        # 构造 prepsubband 命令（最后一个方案，DM 数量加 1）
                        cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'] + 1, list_DD_schemes[i]['downsamp'], nsubbands, infile)
                cmd_prepsubband_list.append(cmd_prepsubband)
                ifok_list.append(ifok_path)
                log_list.append(log_abspath)

                file_script_prepsubband.write("%s\n" % cmd_prepsubband)

        file_script_prepsubband.close()
        return cmd_prepsubband_list,ifok_list,log_list

def realfft(infile,sourcename, out_dir,ifok_dir, log_path, other_flags="", presto_env=os.environ['PRESTO']):

    DM = extract_dm_part(infile)
    outfile_basename = f"{sourcename}_DM{DM}"

    fftfile_abspath = os.path.join(out_dir, "%s.fft" % (outfile_basename))  
    cmd_realfft = "realfft %s %s" % (other_flags, infile)  

    if os.path.exists(fftfile_abspath) and (os.path.getsize(fftfile_abspath) > 0):  
        print_log("警告：文件 %s 已存在。跳过 realfft..." % (fftfile_abspath),color=colors.WARNING,mode='p')
    else:  
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}  # 设置环境变量
        #execute_and_log(cmd_realfft, out_dir, log_abspath, dict_env, 0)  
        ifokfile = os.path.join(ifok_dir,f'real-{DM}.ifok')
        run_cmd(cmd_realfft,ifok=ifokfile,work_dir=out_dir,log_file=log_path,dict_envs=dict_env,flag_append=None)

def realfft2cmd(infile_list,sourcename, out_dir,ifok_dir, log_dir, other_flags="", presto_env=os.environ['PRESTO']):
    cmd_rfft_list=[]
    ifok_list = []
    log_list = []

    for dat in infile_list:
        DM = extract_dm_part(dat)
        outfile_basename = f"{sourcename}_DM{DM}"
        
        cmd_realfft = "realfft %s %s" % (other_flags, dat)  
        ifokfile = os.path.join(ifok_dir,f'real-{DM}.ifok')
        log_file = os.path.join(log_dir,f'LOG_04-FFT-{DM}.txt')

        cmd_rfft_list.append(cmd_realfft)
        ifok_list.append(ifokfile)
        log_list.append(log_file)
        file_script_prepdata_abspath = "%s/%s" % (out_dir, 'realfft_script.txt')
        write2file(cmd_realfft,file_script_prepdata_abspath)

    return cmd_rfft_list,ifok_list,log_list   

def rednoise(fftfile,sourcename, out_dir, ifok_dir,log_dir, other_flags="", presto_env=os.environ['PRESTO']):
        # 获取文件名和基本路径
        fftfile_nameonly = os.path.basename(fftfile)
        DM = extract_dm_part(fftfile)
        fftfile_basename = os.path.splitext(fftfile_nameonly)[0]

        dereddened_ffts_filename = "%s/dereddened_ffts.txt" % (out_dir)  # 已去红噪声文件列表
        fftfile_rednoise_abspath = os.path.join(out_dir, "%s_red.fft" % (fftfile_basename))  # 去红噪声后的.fft文件路径
        inffile_rednoise_abspath = os.path.join(out_dir, "%s_red.inf" % (fftfile_basename))  # 去红噪声后的.inf文件路径
        inffile_original_abspath = os.path.join(out_dir, "%s.inf" % (fftfile_basename))  # 原始.inf文件路径

        cmd_rednoise = "rednoise %s %s" % (other_flags, fftfile)  # 构造rednoise命令

        try:  # 尝试打开已去红噪声文件列表
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')
        except: 
                os.mknod(dereddened_ffts_filename)
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')

        # 检查当前fft文件是否已在去红噪声列表中
        if "%s\n" % (fftfile) in file_dereddened_ffts.readlines():
                if os.path.getsize(fftfile) > 0:  # 如果文件大小大于0，则跳过
                        operation = "skip"
                else:  # 如果文件大小为0，则重新处理
                        operation = "make_from_scratch"
        else:  # 如果文件不在去红噪声列表中，则重新处理
                operation = "make_from_scratch"
                #print("rednoise:: 文件 '%s' 不在去红噪声文件列表中 (%s)，将从头开始处理..." % (fftfile_basename, dereddened_ffts_filename))

        file_dereddened_ffts.close()
        if operation == "make_from_scratch":  # 如果需要从头处理
                print_log(f'检查{dereddened_ffts_filename}不完整，将从头开始处理',color=colors.WARNING)
                ifokfile = os.path.join(ifok_dir,f'red-{DM}.ifok')
                dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
                run_cmd(cmd_rednoise,ifok=ifokfile,work_dir=out_dir,log_file=log_dir,dict_envs=dict_env,flag_append=None)

                file_dereddened_ffts = open(dereddened_ffts_filename, 'a')  # 将文件添加到去红噪声列表
                file_dereddened_ffts.write("%s\n" % (fftfile))
                file_dereddened_ffts.close()
                os.rename(fftfile_rednoise_abspath, fftfile_rednoise_abspath.replace("_red.", "."))  # 重命名文件
                os.rename(inffile_rednoise_abspath, inffile_rednoise_abspath.replace("_red.", "."))

def rednoise2cmd(fftfile_list,sourcename, out_dir, ifok_dir,log_dir, other_flags="", presto_env=os.environ['PRESTO']):
        cmd_red_list=[]
        ifok_list = []
        log_list = []
        # 获取文件名和基本路径
        for fft in fftfile_list:
            fftfile_nameonly = os.path.basename(fft)
            DM = extract_dm_part(fft)
            fftfile_basename = os.path.splitext(fftfile_nameonly)[0]

            fftfile_rednoise_abspath = os.path.join(out_dir, "%s_red.fft" % (fftfile_basename))  # 去红噪声后的.fft文件路径
            inffile_rednoise_abspath = os.path.join(out_dir, "%s_red.inf" % (fftfile_basename))  # 去红噪声后的.inf文件路径
            inffile_original_abspath = os.path.join(out_dir, "%s.inf" % (fftfile_basename))  # 原始.inf文件路径

            cmd_rednoise = "rednoise %s %s" % (other_flags, fft)  # 构造rednoise命令
            ifokfile = os.path.join(ifok_dir,f'red-{DM}.ifok')
            logfile = os.path.join(log_dir,f'LOG_04-RED-{DM}.txt')

            cmd_red_list.append(cmd_rednoise)
            ifok_list.append(ifokfile)
            log_list.append(logfile)
        return cmd_red_list,ifok_list,log_list
               
def check_accelsearch_result(fft_infile, zmax, verbosity_level=0):
    fft_infile_nameonly = os.path.basename(fft_infile)
    fft_infile_basename = os.path.splitext(fft_infile_nameonly)[0]


    ACCEL_filename = fft_infile.replace(".fft", "_ACCEL_%d" % (zmax))
    ACCEL_cand_filename = fft_infile.replace(".fft", "_ACCEL_%d.cand" % (zmax))
    ACCEL_txtcand_filename = fft_infile.replace(".fft", "_ACCEL_%d.txtcand" % (zmax))

    if verbosity_level >= 2:
        print("check_accelsearch_result:: 输入文件基本名称: ", fft_infile_basename)
        print("check_accelsearch_result:: ACCEL文件名 = ", ACCEL_filename)
        print("check_accelsearch_result:: ACCEL候选文件名 = ", ACCEL_cand_filename)
        print("check_accelsearch_result:: ACCEL文本候选文件名 = ", ACCEL_txtcand_filename)

    try:
        if (os.path.getsize(ACCEL_filename) > 0) and (os.path.getsize(ACCEL_cand_filename) > 0) and (os.path.getsize(ACCEL_txtcand_filename) > 0):
            result_message = "check_accelsearch_result:: 文件存在且大小 > 0! 跳过..."
            check_result = True
        else:
            result_message = "check_accelsearch_result:: 文件存在但至少有一个文件大小 = 0!"
            check_result = False
    except OSError:
        result_message = "check_accelsearch_result:: OSError: 看起来accelsearch尚未执行!"
        check_result = False

    if verbosity_level >= 1:
        print(result_message)

    return check_result
        
def accelsearch(infile,sourcename, work_dir,ifok_dir, log_abspath, numharm=8, zmax=0, other_flags="", dict_env={}):

        DM = extract_dm_part(infile)
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        # 构造空结果文件的路径（用于标记未产生候选结果的情况）
        inffile_empty = infile.replace(".fft", "_ACCEL_%d_empty" % (zmax))

        # 构造 accelsearch 命令
        cmd_accelsearch = f"accelsearch {other_flags} -zmax {zmax} -numharm {numharm} {infile}"

        # 检查是否已运行过 accelsearch
        if not check_accelsearch_result(infile, int(zmax)) and not check_accelsearch_result(inffile_empty, int(zmax)):
            ifokfile = os.path.join(ifok_dir,f'search{zmax}-{DM}.ifok')
            run_cmd(cmd_accelsearch,ifok=ifokfile,work_dir=work_dir,log_file=log_abspath,dict_envs=dict_env)
        else:
            print_log("accelsearch:: 警告：accelsearch（zmax=%d）似乎已经对文件 %s 执行过。跳过..." % (int(zmax), infile_nameonly),color=colors.WARNING,mode='p')

        # 检查 accelsearch 是否产生候选结果
        if not check_accelsearch_result(infile, int(zmax)):
            with open(inffile_empty, "w") as file_empty:
                print_log("警告：accelsearch 没有产生任何候选结果！写入文件 %s 以标记此情况..." % (inffile_empty),color=colors.WARNING,mode='p')
                file_empty.write("ACCELSEARCH DID NOT PRODUCE ANY CANDIDATES!")

def accelsearch2cmd(infile_list,ifok_dir, log_dir, numharm=8, zmax=0, other_flags=""):

        cmd_search_list = []
        ifok_list = []
        log_list = []

        for fft_file in infile_list:
            DM = extract_dm_part(fft_file)

            cmd_accelsearch = f"accelsearch {other_flags} -zmax {zmax} -numharm {numharm} {fft_file}"
            ifokfile = os.path.join(ifok_dir,f'search{zmax}-{DM}.ifok')
            logfile = os.path.join(log_dir,f'LOG_05-SEARCH-{DM}.txt')

            cmd_search_list.append(cmd_accelsearch)
            ifok_list.append(ifokfile)
            log_list.append(logfile)

        return cmd_search_list,ifok_list,log_list


def jeaksearch2cmd(infile_list,ifok_dir, log_dir,jerksearch_flags='',jerksearch_zmax=10,jerksearch_wmax=30,jerksearch_numharm=8):

        cmd_search_list = []
        ifok_list = []
        log_list = []

        for fft_file in infile_list:
            DM = extract_dm_part(fft_file)

            cmd_accelsearch = "accelsearch %s -zmax %d -wmax %d -numharm %d %s" % (jerksearch_flags, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm, fft_file)
            ifokfile = os.path.join(ifok_dir,f'search-{DM}.ifok')
            logfile = os.path.join(log_dir,f'LOG_05-SEARCH-{DM}.txt')

            cmd_search_list.append(cmd_accelsearch)
            ifok_list.append(ifokfile)
            log_list.append(logfile)

        return cmd_search_list,ifok_list,log_list

def check_zaplist_outfiles(fft_infile):
        birds_filename   = fft_infile.replace(".fft", ".birds")
        zaplist_filename = fft_infile.replace(".fft", ".zaplist")
        try:
                if (os.path.getsize(birds_filename) > 0) and (os.path.getsize(zaplist_filename) >0): #checks if it exists and its
                        return True
                else:
                        return False
        except OSError:
                return False

def make_zaplist(fft_infile,sourcename, out_dir, ifok_dir,log_dir, common_birdies_filename, birds_numharm=4, other_flags_accelsearch="", presto_env=os.environ['PRESTO']):
    fft_infile_nameonly = os.path.basename(fft_infile)  # 获取输入文件的文件名
    fft_infile_basename = os.path.splitext(fft_infile_nameonly)[0]  # 获取输入文件的基本名称（无扩展名）
    dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}  # 设置环境变量

    # 检查是否已存在zaplist文件
    if check_zaplist_outfiles(fft_infile) == False:
        accelsearch(fft_infile, sourcename,out_dir,ifok_dir, log_dir, birds_numharm, 0, other_flags_accelsearch, dict_env)  # 执行accelsearch
        ACCEL_0_filename = fft_infile.replace(".fft", "_ACCEL_0")  # 生成的ACCEL_0文件名
        fourier_bin_width_Hz = get_Fourier_bin_width(fft_infile)  # 获取傅里叶频宽
        print_log("傅里叶频宽：", fourier_bin_width_Hz)
        print_log("正在生成鸟频文件...")

        try:
            birds_filename = make_birds_file(ACCEL_0_filename=ACCEL_0_filename, out_dir=out_dir, log_dir=log_dir, log_filename=log_abspath, width_Hz=fourier_bin_width_Hz, flag_grow=1, flag_barycentre=0, sigma_birdies_threshold=4, verbosity_level=0)  # 生成鸟频文件
        except:
            print()
            print_log("警告：在0-DM时序中未发现更多鸟频：频带非常干净或掩模效果很好？",color=colors.WARNING)
            infile_nameonly = os.path.basename(ACCEL_0_filename)
            infile_basename = infile_nameonly.replace("_ACCEL_0", "")
            birds_filename = ACCEL_0_filename.replace("_ACCEL_0", ".birds")

        file_common_birdies = open(common_birdies_filename, 'r')  # 打通鸟频文件
        file_birds = open(birds_filename, 'a')  # 打开当前鸟频文件

        for line in file_common_birdies:  # 将通鸟频文件内容追加到当前鸟频文件
            file_birds.write(line)
        file_birds.close()

        zaplist_filename = fft_infile.replace(".fft", ".zaplist")
        cmd_makezaplist = "makezaplist.py %s" % (birds_filename)  # 构造makezaplist命令
        run_cmd(cmd_makezaplist,ifok=zaplist_filename,work_dir=out_dir,log_file=log_dir,dict_envs=dict_env,flag_append=None)
        #execute_and_log(cmd_makezaplist, out_dir, log_abspath, dict_env, 0)  # 执行makezaplist命令并记录日志
    else:
        print_log("文件 %s 的zaplist已存在！" % (fft_infile_basename),color=colors.OKBLUE)

    zaplist_filename = fft_infile.replace(".fft", ".zaplist")  # 生成的zaplist文件名
    return zaplist_filename

def zapbirds2cmd(fft_infile_list, zapfile_name,ifok_dir,log_dir):
        cmd_zapbirds_list = []
        ifok_list = []
        log_list = []

        for fft_file in fft_infile_list:
            DM = extract_dm_part(fft_file)

            cmd_zapbirds = "zapbirds -zap -zapfile %s %s" % (zapfile_name, fft_file)
            ifokfile = os.path.join(ifok_dir,f'zap-{DM}.ifok')
            logfile = os.path.join(log_dir,f'LOG_04-ZAP-{DM}.txt')

            cmd_zapbirds_list.append(cmd_zapbirds)
            ifok_list.append(ifokfile)
            log_list.append(logfile)

        return cmd_zapbirds_list,ifok_list,log_list

def check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax,v = 0):
    dat_file_nameonly = os.path.basename(dat_file)
    fft_file = dat_file.replace(".dat", ".fft")
    fft_file_nameonly = os.path.basename(fft_file)

    # 遍历所有 zmax 值，逐个检查
    for z in list_zmax:
        ACCEL_filename = dat_file.replace(".dat", f"_ACCEL_{int(z)}")
        ACCEL_filename_empty = dat_file.replace(".dat", f"_ACCEL_{int(z)}_empty")
        ACCEL_cand_filename = ACCEL_filename + ".cand"
        ACCEL_txtcand_filename = ACCEL_filename + ".txtcand"

        if (not os.path.exists(ACCEL_filename) or os.path.getsize(ACCEL_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if v >0:
                print(f"check_if_DM_trial_was_searched:: 返回 False - 情况 1: z={z}, 缺少或空文件: {ACCEL_filename} / {ACCEL_filename_empty}")
            return False

        if (not os.path.exists(ACCEL_cand_filename) or os.path.getsize(ACCEL_cand_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if v >0:
                print(f"check_if_DM_trial_was_searched:: 返回 False - 情况 2: z={z}, 缺少或空候选文件: {ACCEL_cand_filename} / {ACCEL_filename_empty}")
            return False

        if not os.path.exists(ACCEL_txtcand_filename):
            if v >0:
                print(f"check_if_DM_trial_was_searched:: 返回 False - 情况 3: z={z}, 缺少 txtcand 文件: {ACCEL_txtcand_filename}")
            return False

    # 检查 jerk 搜索结果（如果启用）
    if flag_jerk_search == 1 and jerksearch_wmax > 0:
        ACCEL_filename = dat_file.replace(".dat", f"_ACCEL_{jerksearch_zmax}_JERK_{jerksearch_wmax}")
        ACCEL_filename_empty = dat_file.replace(".dat", f"_ACCEL_{jerksearch_zmax}_JERK_{jerksearch_wmax}_empty")
        ACCEL_cand_filename = ACCEL_filename + ".cand"
        ACCEL_txtcand_filename = ACCEL_filename + ".txtcand"

        if (not os.path.exists(ACCEL_filename) or os.path.getsize(ACCEL_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if v >0:
                print(f"check_if_DM_trial_was_searched:: 返回 False - 情况 4: jerk搜索结果缺失或为空: {ACCEL_filename} / {ACCEL_filename_empty}")
            return False

        if (not os.path.exists(ACCEL_cand_filename) or os.path.getsize(ACCEL_cand_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if v >0:
                print(f"check_if_DM_trial_was_searched:: 返回 False - 情况 5: jerk搜索候选文件缺失或为空: {ACCEL_cand_filename} / {ACCEL_filename_empty}")
            return False

        if not os.path.exists(ACCEL_txtcand_filename):
            if v >0:
                print(f"check_if_DM_trial_was_searched:: 返回 False - 情况 6: jerk搜索 txtcand 缺失: {ACCEL_txtcand_filename}")
            return False

    return True

def sift_candidates(work_dir,sourcename, log_dir,  dedispersion_dir, list_zmax, jerksearch_zmax, jerksearch_wmax, flag_remove_duplicates, flag_DM_problems, flag_remove_harmonics, minimum_numDMs_where_detected, minimum_acceptable_DM=2.0, period_to_search_min_s=0.001, period_to_search_max_s=15.0):

        best_cands_filename = "%s/best_candidates_%s.siftedcands" % (work_dir, sourcename)

        list_ACCEL_files = []
        for z in list_zmax:
                string_glob = "%s/*ACCEL_%d" % (dedispersion_dir, z)
                print("Reading files '%s'..." % (string_glob), end=' ')
                list_ACCEL_files = list_ACCEL_files + glob.glob(string_glob)
                print("done!")

        string_glob_jerk_files = "%s/*ACCEL_%d_JERK_%d" % (dedispersion_dir, jerksearch_zmax, jerksearch_wmax)

        list_ACCEL_files = list_ACCEL_files + glob.glob(string_glob_jerk_files)

        log_abspath = "%s/LOG_%s.txt" % (log_dir, 'SIFTING')
        print("\033[1m >> TIP:\033[0m Check sifting output with '\033[1mcat %s\033[0m'" % (log_abspath))

        list_DMs = [x.split("_ACCEL")[0].split("DM")[-1] for x in list_ACCEL_files]
        candidates = sifting.read_candidates(list_ACCEL_files, track=True)

        print("sift_candidates:: z = %d" % (z))
        print("sift_candidates:: %s/*ACCEL_%d" % (dedispersion_dir, z))
        print("sift_candidates:: Original N_cands = ", len(candidates.cands))
        print("sift_candidates:: sifting.sigma_threshold = ", sifting.sigma_threshold)

        sifting.short_period = period_to_search_min_s
        sifting.long_period = period_to_search_max_s
        print()
        print("Selecting candidates with periods %.4f < P < %.4f seconds..." % (period_to_search_min_s, period_to_search_max_s), end=' ')
        sys.stdout.flush()
        candidates.reject_shortperiod()
        candidates.reject_longperiod()
        print("done!")
        #sifting.write_candlist(cands,cwd+"/candidate_list_from_script")


        if flag_remove_duplicates == 1:  # 如果设置了去除重复候选者的标志
                candidates = sifting.remove_duplicate_candidates(candidates)  # 去除重复候选者
                print("sift_candidates:: 已去除重复项。候选者数量 = ", len(candidates.cands))  # 打印去除重复后的候选者数量

        if flag_DM_problems == 1:  # 如果设置了去除 DM 问题的标志
                candidates = sifting.remove_DM_problems(candidates, minimum_numDMs_where_detected, list_DMs, minimum_acceptable_DM)  # 去除 DM 问题的候选者
                print("sift_candidates:: 已去除 DM 问题。候选者数量 = ", len(candidates.cands))  # 打印去除 DM 问题后的候选者数量

        if flag_remove_harmonics == 1:  # 如果设置了去除谐波的标志
                try:
                        candidates = sifting.remove_harmonics(candidates)  # 尝试去除谐波
                except:
                        pass  # 如果发生异常则忽略
                print("sift_candidates:: 已去除谐波。候选者数量 = ", len(candidates.cands))  # 打印去除谐波后的候选者数量

        print("sift_candidates:: 正在按 sigma 排序候选者...", end=' '); sys.stdout.flush()  # 提示正在按 sigma 排序候选者
        try:
                candidates.sort(sifting.cmp_sigma)              # If using PRESTO 2.1's sifting.py
        except AttributeError:
                candidates.sort(key=sifting.attrgetter('sigma'), reverse=True)  # If using PRESTO 3's sifting.py

        print("完成！")  # 提示上一步操作完成
        print("sift_candidates:: 正在将最佳候选者写入文件 '%s'..." % (best_cands_filename), end=' '); sys.stdout.flush()  # 提示正在写入最佳候选者文件
        sifting.write_candlist(candidates, best_cands_filename)  # 调用函数写入最佳候选者
        print("完成！")  # 提示写入最佳候选者文件完成
        print("sift_candidates:: 正在将报告写入文件 '%s'..." % (log_abspath), end=' '); sys.stdout.flush()  # 提示正在写入报告文件
        candidates.write_cand_report(log_abspath)  # 调用函数写入报告
        print("完成！")  # 提示写入报告文件完成
        return candidates

def get_DDplan_scheme(infile, out_dir, log_dir, LOG_basename, loDM, highDM, DM_coherent_dedispersion, N_DMs_per_prepsubband, freq_central_MHz, bw_MHz, nchan, nsubbands, t_samp_s):
    # 获取输入文件的名称和基础名
    infile_nameonly = os.path.basename(infile)
    infile_basename = os.path.splitext(infile_nameonly)[0]
    infile_basename = infile_basename+'.ps'
    # 构造日志文件的绝对路径
    log_abspath = f"{log_dir}/LOG_{LOG_basename}.txt"
    
    # 根据去色散策略构造 DDplan 命令
    if np.float64(DM_coherent_dedispersion) == 0:
        # 如果未启用相干去色散
        if nsubbands == 0:                        
            # 未启用子带
            cmd_DDplan = f"DDplan.py -o ddplan_{infile_basename} -l {loDM} -d {highDM} -f {freq_central_MHz} -b {np.fabs(bw_MHz)} -n {nchan} -t {t_samp_s}"
        else:
            # 启用子带
            cmd_DDplan = f"DDplan.py -o ddplan_{infile_basename} -l {loDM} -d {highDM} -f {freq_central_MHz} -b {np.fabs(bw_MHz)} -n {nchan} -t {t_samp_s} -s {nsubbands}"
            print(f"已启用子带，共 {nsubbands} 个子带（数据中的通道数：{nchan}")

    elif np.float64(DM_coherent_dedispersion) > 0:
        # 启用相干去色散
        print(f"已启用相干去色散，DM = {np.float64(DM_coherent_dedispersion):.3f} pc cm^-3")
        if nsubbands == 0: 
            # 未启用子带
            cmd_DDplan = f"DDplan.py -o ddplan_{infile_basename} -l {loDM} -d {highDM} -c {DM_coherent_dedispersion} -f {freq_central_MHz} -b {np.fabs(bw_MHz)} -n {nchan} -t {t_samp_s}"
        else:
            # 启用子带
            cmd_DDplan = f"DDplan.py -o ddplan_{infile_basename} -l {loDM} -d {highDM} -c {DM_coherent_dedispersion} -f {freq_central_MHz} -b {np.fabs(bw_MHz)} -n {nchan} -t {t_samp_s} -s {nsubbands}"
            print(f"已启用子带，共 {nsubbands} 个子带（数据中的通道数：{nchan}）")
            
    elif np.float64(DM_coherent_dedispersion) < 0:
        # DM 值小于 0，报错退出
        print_log("错误：相干去色散的 DM 值小于 0！程序退出...",color=colors.ERROR)
        exit()

    # 执行 DDplan 命令并获取输出结果
    output_DDplan = get_command_output(cmd_DDplan, shell_state=False, work_dir=out_dir)

    # 从 DDplan 输出中解析去色散方案
    list_DD_schemes = get_DD_scheme_from_DDplan_output(output_DDplan, N_DMs_per_prepsubband, nsubbands)
    
    return list_DD_schemes

def get_command_output(command, shell_state=False, work_dir=os.getcwd()):
        print_log(f'运行命令：{command}', masks=command ,color=colors.OKCYAN)
        time.sleep(0.8) 
        list_for_Popen = command.split()
        if shell_state ==False:
                proc = subprocess.Popen(list_for_Popen, stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        else:
                proc = subprocess.Popen([command], stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        out, err = proc.communicate()  
        append_to_script_if_not_exists(os.path.join(work_dir, 'ddplan.sh'),f'#程序运行路径为:{work_dir}')
        append_to_script_if_not_exists(os.path.join(work_dir, 'ddplan.sh'),command)   
        return out.decode('ascii')

def get_DD_scheme_from_DDplan_output(output_DDplan, N_DMs_per_prepsubband, nsubbands):
        list_dict_schemes = []
        downsamp = 1
        output_DDplan_list_lines = output_DDplan.split("\n")
        if nsubbands == 0:
                index = output_DDplan_list_lines.index("  Low DM    High DM     dDM  DownSamp   #DMs  WorkFract")   + 1
        else:
                index = output_DDplan_list_lines.index("  Low DM    High DM     dDM  DownSamp  dsubDM   #DMs  DMs/call  calls  WorkFract")   + 1
                
        print_log("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print_log(output_DDplan)
        print_log("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        flag_add_plans = 1
        while flag_add_plans == 1:
                if output_DDplan_list_lines[index] == "":
                        return list_dict_schemes
                else:
                        if nsubbands == 0:
                                param = output_DDplan_list_lines[index].split()
                                low_DM_by_DDplan   = np.float64(param[0])
                                high_DM_by_DDplan = np.float64(param[1])
                                dDM = np.float64(param[2])
                                # downsamp = int(param[3])
                                num_DMs = int(param[4])

                                if num_DMs > N_DMs_per_prepsubband:
                                        N_schemes = int(num_DMs / N_DMs_per_prepsubband) + 1

                                        for n in range(N_schemes-1):
                                                lowDM   = low_DM_by_DDplan + (n    * N_DMs_per_prepsubband * dDM)
                                                highDM = lowDM + N_DMs_per_prepsubband * dDM                                
                                                dict_scheme = {'loDM': lowDM, 'highDM': highDM, 'dDM': dDM, 'downsamp': downsamp, 'num_DMs': N_DMs_per_prepsubband}
                                                print("dict_scheme for loop = ",  dict_scheme)
                                                list_dict_schemes.append(dict_scheme)

                                        lowDM =  round(low_DM_by_DDplan + (N_schemes-1)   * N_DMs_per_prepsubband * dDM,             3 ) #round to third decimal digit
                                        highDM = round(high_DM_by_DDplan, 3)
                                        numDMs = int(round((highDM - lowDM) / dDM, 3))
				
                                        dict_scheme = {'loDM': lowDM, 'highDM': highDM, 'dDM': dDM, 'downsamp': downsamp, 'num_DMs': numDMs}
                                        list_dict_schemes.append(dict_scheme)

                                else:
                                        dict_scheme = {'loDM': low_DM_by_DDplan, 'highDM': high_DM_by_DDplan, 'dDM': dDM, 'downsamp': downsamp, 'num_DMs': num_DMs}
                                        #print("num_DMs else =", num_DMs)
                                        #print("dict_scheme else =", dict_scheme)
                                        list_dict_schemes.append(dict_scheme)

                        elif nsubbands > 0:
                                param = output_DDplan_list_lines[index].split()
                                low_DM_by_DDplan   = np.float64(param[0])
                                high_DM_by_DDplan = np.float64(param[1])
                                dDM = np.float64(param[2])
                                # downsamp = int(param[3])
                                dsubDM = np.float64(param[4])
                                num_DMs = int(param[5])
                                num_DMs_percall = int(param[6])
                                N_calls = int(param[7])

                                
                                for k in range(N_calls):
                                        dict_scheme = {'loDM': round(low_DM_by_DDplan + k*dsubDM, 3), 'highDM': round(low_DM_by_DDplan+(k+1)*dsubDM, 3), 'dDM': dDM, 'downsamp': downsamp, 'num_DMs': num_DMs_percall}
                                        list_dict_schemes.append(dict_scheme)


                                        
                index = index + 1


def ps2png(input_pattern, rotated=True, recursive=False, output_dir=None):
    # 匹配文件
    ps_files = glob.glob(input_pattern, recursive=recursive)
    if not ps_files:
        print(f"未找到匹配文件: {input_pattern}")
        return

    for input_file in ps_files:
        if not os.path.isfile(input_file):
            continue

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, base_name + ".png") if output_dir else os.path.splitext(input_file)[0] + ".png"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            subprocess.run([
                "gs", "-dQUIET", "-dBATCH", "-dNOPAUSE",
                "-sDEVICE=png256",
                "-sOutputFile=" + output_file,
                "-r300", input_file
            ], check=True)

            if rotated:
                image = Image.open(output_file)
                rotated_image = image.rotate(-90, expand=True)
                rotated_image.save(output_file)

        except Exception as e:
            print(f"错误: {input_file} 转换失败 - {e}")

def resize_and_pad(img, target_width):
    """调整图像大小并填充以匹配目标宽度"""
    w, h = img.size
    if w != target_width:
        new_height = int(h * (target_width / w))
        img = img.resize((target_width, new_height), Image.LANCZOS)
    return img

def merge_images(file_a, file_b, output_path):
    """合并两个图像并保存到指定路径"""
    img_a = Image.open(file_a)
    img_b = Image.open(file_b)

    target_width = max(img_a.width, img_b.width)
    img_a = resize_and_pad(img_a, target_width)
    img_b = resize_and_pad(img_b, target_width)

    spacing = 20
    total_height = img_a.height + img_b.height + spacing
    merged_img = Image.new('RGB', (target_width, total_height), color=(255, 255, 255))

    merged_img.paste(img_a, (0, 0))
    merged_img.paste(img_b, (0, img_a.height + spacing))

    merged_img.save(output_path)

def send_email(content, file_paths=None):
    # Email configuration
    mail_host = "smtp.qq.com"  
    mail_pass = 'bfahykurlmsoiaab'  # 这是你的 QQ 邮箱授权码
    sender = '1964865346@qq.com'  
    receivers = ['2107053791@qq.com']  
    subject = '运行成功'  

    # Create a multipart message
    message = MIMEMultipart()
    message['From'] = Header(f'=?utf-8?B?{base64.b64encode("python自动发送".encode()).decode()}=?= <1964865346@qq.com>')      
    message['To'] = Header("python自动发送", 'utf-8')    
    message['Subject'] = Header(subject, 'utf-8')

    # Add message body
    message.attach(MIMEText(content, 'plain', 'utf-8'))

    # Add attachments if file paths are provided
    if file_paths:
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)  # 获取文件名
                with open(file_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={Header(filename, "utf-8")}')
                message.attach(part)
            except Exception as e:
                print(f"附件 {file_path} 添加失败: {e}")

    try:
        # Connect to SMTP server
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)  # 使用SSL连接
        smtpObj.login(sender, mail_pass)  
        smtpObj.sendmail(sender, receivers, message.as_string())  
        smtpObj.quit()
        print('邮件发送成功！！')
    except smtplib.SMTPException as e:
        print(f"错误：{e}")
        print('邮件发送失败！！')

# def execute_and_log(command, work_dir, log_abspath, dict_envs={}, flag_append=0):
#         datetime_start = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
#         time_start = time.time()
#         if flag_append == 1:
#                 flag_open_mode = "a"
#         else:
#                 flag_open_mode = "w+"
#         log_file = open("%s" % (log_abspath), flag_open_mode)
#         executable = command.split()[0]

#         # 写入日志文件的开头部分，包括启动时间和命令等信息
#         log_file.write("****************************************************************\n")
#         log_file.write("开始日期和时间：%s\n" % (datetime_start))  # 将 "START DATE AND TIME" 改为中文
#         log_file.write("\n命令：\n")  # 将 "COMMAND" 改为中文
#         log_file.write("%s\n\n" % (command))  # 输出执行的命令
#         log_file.write("工作目录：%s\n" % (work_dir))  # 将 "WORKING DIRECTORY" 改为中文
#         log_file.write("****************************************************************\n")
#         log_file.flush()  # 刷新日志文件缓冲区，确保内容写入文件

#         list_for_Popen = command.split()
#         env_subprocess = os.environ.copy()
#         if dict_envs:  # If the dictionary is not empty                                                                                                                                                            
#                 for k in list(dict_envs.keys()):
#                         env_subprocess[k] = dict_envs[k]

#         proc = subprocess.Popen(list_for_Popen, stdout=log_file, stderr=log_file, cwd=work_dir, env=env_subprocess)
#         proc.communicate()  # Wait for the process to complete                                                                                                                                                    

#         datetime_end = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
#         time_end = time.time()

#         #print("execute_and_log:: 命令：%s" % (command))  # 打印执行的命令
#         #print("execute_and_log:: 找到的可执行文件路径：%s: " % (executable), get_command_output("which %s" % (executable)))  # 打印可执行文件路径
#         #print("execute_and_log:: 工作目录 = ", work_dir)  # 打印工作目录
#         #print("execute_and_log:: 查看日志的方式：\"tail -f %s\"" % (log_abspath))  # 提示如何查看日志
#         #sys.stdout.flush()  # 刷新标准输出缓冲区
#         #print("execute_and_log: 用于 Popen 的命令列表 = ", list_for_Popen)  # 打印用于子进程的命令列表
#         #print("execute_and_log: 日志文件 = ", log_file)  # 打印日志文件对象
#         #print("execute_and_log: 子进程环境变量 = ", env_subprocess)  # 打印子进程的环境变量

#         log_file.write("\n结束日期和时间：%s\n" % (datetime_end))  # 写入结束时间和日期
#         log_file.write("\n总耗时：%d 秒\n" % (time_end - time_start))  # 写入总耗时
#         log_file.close()  # 关闭日志文件

#         with open(os.path.join(os.getcwd(), 'cmd.sh'), 'a') as file:
#                 file.write(f'#程序运行路径为:{work_dir}  \n')
#                 file.write(command + '\n')

# def execute_and_log_in_thread_pool(command, log_dir, work_dir, id_num, N_ids, flag_log=1):
#     # 获取当前时间，用于日志记录
#     datetime_start = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
#     datetime_start_single_string = (datetime.now()).strftime("%Y%m%d_%H%M")
#     time_start = time.time()

#     # 提取命令的标签（用于日志文件命名）
#     if "/" in command.split()[0]:
#         command_label = command.split("/")[-1].split()[0]
#     else:
#         command_label = command.split()[0]

#     # 将命令字符串分割为列表，用于传递给 subprocess.Popen
#     list_for_Popen = command.split()

#     # 遍历命令列表，处理包含通配符（? 或 *）的参数
#     for i in range(len(list_for_Popen)):
#         current_piece = list_for_Popen[i]
#         if "?" in current_piece or "*" in current_piece:
#             # 使用 glob 模块展开通配符，并替换原列表中的对应部分
#             new_list_for_Popen = list_for_Popen[:i] + sorted(glob.glob(current_piece)) + list_for_Popen[i+1:]
#             list_for_Popen = new_list_for_Popen

#     # 根据 flag_log 决定是否记录日志
#     if flag_log == 1:
#         # 构造日志文件名并打开日志文件
#         log_filename = "LOG_%s_%s_%03d.txt" % (command_label, datetime_start_single_string, int(id_num))
#         log_abspath = os.path.join(log_dir, log_filename)
#         log_file = open(log_abspath, "w+")

#         # 写入日志文件的头部信息
#         log_file.write("****************************************************************\n")
#         log_file.write("开始日期和时间：%s\n" % (datetime_start))
#         log_file.write("\n命令：\n")
#         log_file.write("%s\n\n" % (command))
#         log_file.write("工作目录：%s\n" % (work_dir))
#         log_file.write("****************************************************************\n")
#         log_file.flush()

#         # 在指定工作目录下执行命令，并将输出重定向到日志文件
#         proc = subprocess.Popen(list_for_Popen, cwd=work_dir, stdout=log_file, stderr=log_file)
#     elif flag_log == 0:
#         # 如果不记录日志，直接打印提示信息
#         print("未记录日志...")
#         proc = subprocess.Popen(list_for_Popen, cwd=work_dir, stdout=subprocess.PIPE)

#     # 等待命令执行完成
#     proc.communicate()

#     # 获取结束时间和总耗时
#     datetime_end = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
#     time_end = time.time()

#     # 如果记录日志，写入结束时间和总耗时
#     if flag_log == 1:
#         log_file.write("\n结束日期和时间：%s\n" % (datetime_end))
#         log_file.write("\n总耗时：%d 秒\n" % (time_end - time_start))
#         log_file.close()

#     # 打印命令执行完成的提示信息
#     print("命令 %4d/%d ('%s') 执行完成。" % (id_num + 1, N_ids, command_label)); sys.stdout.flush()



def get_command_output_with_pipe(command1, command2):
        list_for_Popen_cmd1 = command1.split()
        list_for_Popen_cmd2 = command2.split()

        p1 = subprocess.Popen(list_for_Popen_cmd1, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(list_for_Popen_cmd2, stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()

        out, err = p2.communicate()
        return out.decode('ascii')

def readfile_with_str(command1, command2):
    cmd = f"{command1} | {command2}"
    #print(cmd)  

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()  # 获取命令的输出和错误信息
    output_str = output.decode('utf-8').strip()
    return output_str


def fold_candidate(work_dir, log_dir, LOG_basename, observation, dir_dedispersion, obs, seg, ck, candidate, ignorechan_list, other_flags_prepfold="", presto_env=os.environ['PRESTO'], flag_LOG_append=1, what_fold="rawdata", num_simultaneous_prepfolds=1):
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
        cand = candidate
        dir_accelfile = "%s/%s/%s/%s" % (dir_dedispersion, obs, seg, ck)
        cand_zmax = cand.filename.split("ACCEL_")[-1].split("_JERK")[0]

        if "JERK_" in os.path.basename(cand.filename):
                cand_wmax = cand.filename.split("JERK_")[-1]
                str_zmax_wmax = "z%s_w%s" % (cand_zmax, cand_wmax)
        else:
                str_zmax_wmax = "z%s" % (cand_zmax)

        file_script_fold_name = "script_fold.txt"
        file_script_fold_abspath = "%s/%s" % (work_dir, file_script_fold_name)
        file_script_fold = open(file_script_fold_abspath, "a")

        if ignorechan_list !="":
                flag_ignorechan = "-ignorechan %s " % (ignorechan_list)
        else:
                flag_ignorechan = ""

        if '-nsub' not in other_flags_prepfold:
                other_flags_prepfold = other_flags_prepfold + " -nsub %d" % (observation.nchan)

        if what_fold =="timeseries":
                file_to_fold = os.path.join(dir_dedispersion, cand.filename.split("_ACCEL")[0] + ".dat")
                cmd_prepfold = "prepfold %s -noxwin -accelcand %d -accelfile %s/%s.cand -o ts_fold_%s_%s_%s_DM%.2f_%s   %s" % (other_flags_prepfold, cand.candnum, dir_accelfile, cand.filename, obs, seg, ck, cand.DM, str_zmax_wmax, file_to_fold)
        elif what_fold =="rawdata":
                file_to_fold = observation.file_abspath
                if seg == "full":
                        cmd_prepfold = "prepfold %s -noxwin -accelcand %d -accelfile %s/%s.cand -dm %.2f %s -mask %s -o raw_fold_%s_%s_%s_DM%.2f_%s    %s" % (other_flags_prepfold, cand.candnum, dir_accelfile, cand.filename, cand.DM, flag_ignorechan, observation.mask, obs, seg, ck, cand.DM, str_zmax_wmax, file_to_fold)

                file_script_fold.write("%s\n" % cmd_prepfold)
                print(cmd_prepfold)
        file_script_fold.close()


def make_even_number(number_int):  # 定义一个函数，将输入的数字转换为偶数
        if int(number_int) % 2 == 1:  # 如果数字是奇数
                return int(number_int) - 1  # 返回该数字减1后的偶数
        elif int(number_int) % 2 == 0:  # 如果数字已经是偶数
                return int(number_int)  # 直接返回该数字
        else:  # 如果输入的数字既不是奇数也不是偶数（理论上不可能发生）
                print("ERROR: make_even_number:: 输入的数字既不是偶数也不是奇数！")
                exit()  


def get_rfifind_result(file_mask, LOG_file):
        rfifind_mask = rfifind.rfifind(file_mask)  # 加载 rfifind 对象

        N_int = rfifind_mask.nint  # 获取时间积分的数量
        N_chan = rfifind_mask.nchan  # 获取频率通道的数量
        N_int_masked = len(rfifind_mask.mask_zap_ints)  # 获取被屏蔽的时间积分数量
        N_chan_masked = len(rfifind_mask.mask_zap_chans)  # 获取被屏蔽的频率通道数量
        fraction_int_masked = np.float64(N_int_masked / N_int)  # 计算被屏蔽的时间积分比例
        fraction_chan_masked = np.float64(N_chan_masked / N_chan)  # 计算被屏蔽的频率通道比例

        # print("get_rfifind_result:: 文件掩膜：%s" % file_mask)  # 打印文件掩膜
        # print("get_rfifind_result:: 日志文件：%s" % LOG_file)  # 打印日志文件

        if (fraction_int_masked > 0.5) or (fraction_chan_masked > 0.5):  # 如果屏蔽比例超过 50%
                return "!Mask>50%"  # 返回错误信息

        # 检查日志文件中是否存在第一个块的裁剪问题，并获取有问题的文件名。否则返回 True。
        cmd_grep_problem_clipping = "grep -l 'problem with clipping' %s" % (LOG_file)  # -l 选项返回包含该表达式的文件名
        cmd_grep_inf_results = "grep -l ' inf ' %s" % (LOG_file)
        output = get_command_output(cmd_grep_problem_clipping, True).strip()  # 执行命令并获取输出
        if output != "":
                print()
                print("警告：文件 '%s' 中第一个块存在裁剪问题！" % (LOG_file))  # 提示裁剪问题
                return "!ProbFirstBlock"  # 返回错误信息

        output = get_command_output(cmd_grep_inf_results, True).strip()  # 检查是否存在无穷大结果
        if output != "":
                print()
                print("警告：文件 '%s' 中存在无穷大结果！" % (LOG_file))  # 提示无穷大结果
                return "!ProbInfResult"  # 返回错误信息

        return "done"  # 如果没有问题，返回 "done"


def check_prepdata_result(LOG_file, verbosity_level=0):
        # Check if there was a problem with the clipping in first block and get the filename with that problem. Otherwise return True.
        cmd_grep_problem_clipping = "grep -l 'problem with clipping' %s" % (LOG_file)  # -l option returns the name of the file that contains the expression
        cmd_grep_inf_results = "grep -l ' inf ' %s" % (LOG_file)
        output = get_command_output(cmd_grep_problem_clipping, True).strip()
        print("check_prepdata_result::output: -%s-" % (output))
        if output != "":
                if verbosity_level >= 1:
                        print("WARNING: File '%s' contains a problem with clipping in first block!" % (LOG_file))
                return False

        return True


# 检查 RFIFIND 输出文件是否存在
def check_rfifind_outfiles(out_dir, basename):
    for suffix in ["bytemask", "inf", "mask", "ps", "rfi", "stats"]:
        file_path = f"{out_dir}/{basename}_rfifind.{suffix}"
        if not os.path.exists(file_path):
            print(f"ERROR: 文件 {file_path} 未找到！")
            return False
        if os.stat(file_path).st_size == 0:
            print(f"ERROR: 文件 {file_path} 大小为 0！")
            return False
    return True


def check_rednoise_outfiles(fftfile_rednoise_abspath, verbosity_level=0):
        inffile_rednoise_abspath = fftfile_rednoise_abspath.replace("_red.fft", "_red.inf")

        if os.path.exists(fftfile_rednoise_abspath ) and (os.path.getsize(fftfile_rednoise_abspath) > 0) and os.path.exists(inffile_rednoise_abspath) and (os.path.getsize(inffile_rednoise_abspath) > 0):
                return True
        else:
                return False


def check_jerksearch_result(fft_infile, zmax, wmax, verbosity_level=0):
        fft_infile_nameonly = os.path.basename(fft_infile)
        fft_infile_basename = os.path.splitext(fft_infile_nameonly)[0]

        if verbosity_level >= 1:
                print("check_jerksearch_result:: infile_basename: ", fft_infile_basename)
                print("check_jerksearch_result:: ACCEL_filename = ", ACCEL_filename)
                print("check_jerksearch_result:: ACCEL_cand_filename", ACCEL_cand_filename)
                print("check_jerksearch_result:: ACCEL_txtcand_filename = ", ACCEL_txtcand_filename)

        ACCEL_filename =  fft_infile.replace(".fft", "_ACCEL_%d_JERK_%d"          % (zmax, wmax))
        ACCEL_cand_filename =  fft_infile.replace(".fft", "_ACCEL_%d_JERK_%d.cand"     % (zmax, wmax))
        ACCEL_txtcand_filename =  fft_infile.replace(".fft", "_ACCEL_%d_JERK_%d.txtcand"  % (zmax, wmax))

        try:
                if (os.path.getsize(ACCEL_filename) > 0) and (os.path.getsize(ACCEL_cand_filename) > 0) and (os.path.getsize(ACCEL_txtcand_filename) > 0):
                        result_message = "check_jerksearch_result:: Files exist and their size is > 0! Skipping..."
                        check_result = True
                else:
                        result_message = "check_jerksearch_result:: Files exists but at least one of them has size = 0!"
                        check_result = False
        except OSError:
                result_message = "check_jerksearch_result:: OSError: It seems jerksearch has not been executed!"
                check_result = False

        if verbosity_level >= 1:
                print(result_message)

        return check_result



def jerksearch(infile, work_dir, log_abspath, numharm=4, zmax=50, wmax=150, other_flags="", dict_env={}, verbosity_level=0, flag_LOG_append=1):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        inffile_empty = infile.replace(".fft", "_ACCEL_%d_JERK_%d_empty" % (zmax, wmax))
        sys.stdout.flush()
        cmd_jerksearch = "accelsearch %s -zmax %d -wmax %d -numharm %d %s" % (other_flags, zmax, wmax, numharm, infile)

        if verbosity_level >= 2:
                print()
                print("BEGIN JERKSEARCH ----------------------------------------------------------------------")

                print("jerksearch:: cmd_jerksearch: ", cmd_jerksearch)
                print("jerksearch:: AND THIS IS THE ENV: ", dict_env)
                print("jerksearch:: check_accelsearch_result(infile, int(zmax)) :: %s" % (check_accelsearch_result(infile, int(zmax))))
                print("jerksearch:: work_dir = %s" % (work_dir))
                print("jerksearch:: infile = %s" % (infile))

        if check_jerksearch_result(infile, zmax, wmax) == False and check_jerksearch_result(inffile_empty, zmax, wmax) == False:
                if verbosity_level >= 2:
                        print("jerksearch:: executing: %s" % (cmd_jerksearch))
                execute_and_log(cmd_jerksearch, work_dir, log_abspath, dict_env, flag_LOG_append)
        else:
                if verbosity_level >= 2:
                        print("jerksearch:: WARNING: jerk search with zmax=%d and wmax=%s seems to have been already executed on file %s. Skipping..." % (int(zmax), int(wmax), infile_nameonly))

        if verbosity_level >= 2:
                print("jerksearch:: NOW I CHECK THE RESULT OF THE EXECUTION!")

        if check_jerksearch_result(infile, zmax, wmax) == False:
                if verbosity_level >= 2:
                        print("False! Then I create a _empty file!")
                file_empty = open(inffile_empty, "w")
                if verbosity_level >=1:
                        print("%sWARNING%s: jerk search did not produce any candidates! Writing file %s to signal this..." % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, inffile_empty))
                file_empty.write("JERK SEARCH DID NOT PRODUCE ANY CANDIDATES!")
        else:
                if verbosity_level >= 2:
                        print("jerksearch:: GOOD! CANDIDATES HAVE BEEN PRODUCED for %s!" % (infile))

        if verbosity_level >= 2:
                print("END JERKSEARCH ---------------------------------------------------------------------- ")


def split_into_chunks(infile, list_datfiles_to_split, log_dir, LOG_basename,  work_dir, segment_min, i_chunk, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, presto_env=os.environ['PRESTO'], flag_LOG_append=1, flag_remove_datfiles_of_segments=0, verbosity_level=1):
        segment_length_s = segment_min * 60
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}

        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)

        observation_filename = os.path.basename(infile)
        observation_nameonly = os.path.splitext(observation_filename)[0]
        
        str_seg = int(segment_min)
        for datfile_name in sorted(list_datfiles_to_split):
                datfile_nameonly = os.path.basename(datfile_name)               # myfile_DM24.40.dat
                datfile_dirname  = os.path.dirname(datfile_name)
                inffile_name = datfile_name.replace(".dat", ".inf")             # myfile_DM24.40.inf
                str_DM = datfile_nameonly.split(".dat")[0].split("DM")[-1]      # 24.40
                
                info_datfile = infodata.infodata(inffile_name)

                
                datfile_chunk_name = "%s/%s_%sm_ck%02d_DM%s.dat" % (work_dir, observation_nameonly, str_seg, i_chunk, str_DM)

                DM_trial_was_searched = check_if_DM_trial_was_searched(datfile_chunk_name, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax)

                
                string_min = "%dm" % int(segment_min)
                string_chunk = "ck%02d" % i_chunk
                #path_old = os.path.splitext(datfile_name)[0]
                #path_new = path_old.replace("full", string_min).replace("ck00", string_chunk)


                old_suffixes =  datfile_nameonly.split(observation_nameonly)[-1]
                new_suffixes =  old_suffixes.replace("full", string_min).replace("ck00", string_chunk)
                new_datfile_name = observation_nameonly + new_suffixes    #with extension
                
                new_outfile_name = os.path.splitext(new_datfile_name)[0]

                if verbosity_level >= 2:
                        print("split_into_chunks:: datfile_nameonly: ", datfile_nameonly)
                        print("split_into_chunks:: datfile_dirname: ", datfile_dirname)
                        print("split_into_chunks:: inffile_name: ", inffile_name)
                        print("split_into_chunks:: str_DM: ", str_DM)
                        print("split_into_chunks:: info_datfile: ", info_datfile)
                        print("split_into_chunks:: datfile_chunk_name: ", datfile_chunk_name)
                        print("split_into_chunks:: DM_trial_was_searched: ", DM_trial_was_searched)
                        print("split_into_chunks:: old_suffixes: ", old_suffixes)
                        print("split_into_chunks:: new_suffixes: ", new_suffixes)
                        print("split_into_chunks:: new_datfile_name: ", new_datfile_name)
                        print("split_into_chunks:: new_outfile_name: ", new_outfile_name)
                        
                
                if DM_trial_was_searched == False:
                        t_samp_s = info_datfile.dt
                        N_samp = info_datfile.N
                        T_obs_s = t_samp_s * N_samp

                        start_fraction = (i_chunk * segment_length_s)/T_obs_s
                        numout = make_even_number(int(segment_length_s / t_samp_s))

                        
                        cmd_prepdata_split = "prepdata -nobary -o %s/%s -start %.3f -numout %s %s" % (work_dir, new_outfile_name, start_fraction, numout,  datfile_name)

                        output_datfile = "%s/%s.dat" % (work_dir, new_outfile_name)
                        output_inffile = "%s/%s.inf" % (work_dir, new_outfile_name)
                        output_scriptfile = "%s/%s.dat.makecmd" % (work_dir, new_outfile_name)

                        if flag_remove_datfiles_of_segments == 1 and (not os.path.exists(output_scriptfile)):
                                with open(output_scriptfile, 'w') as f:
                                        f.write("%s\n" % (cmd_prepdata_split))
                                os.chmod(output_scriptfile, 0o775)

                        if check_prepdata_outfiles(output_datfile.replace(".dat", "")) == False:
                                if verbosity_level >= 1:        print("Making chunk '%s' of segment '%sm' from '%s'..." % (string_chunk, segment_min, datfile_name), end=''); sys.stdout.flush()
                                execute_and_log(cmd_prepdata_split, work_dir, log_abspath, dict_env, flag_LOG_append)
                                if verbosity_level >= 1:        print("done!")
                        else:
                                if verbosity_level >= 1:
                                        print("NOTE: '%s.dat' already exists. No need to create it again." % (new_outfile_name))
                else:
                        if verbosity_level >= 1:
                                print("NOTE: '%s.dat' was already successfully searched. Skipping..." % (new_outfile_name))





def singlepulse_search(work_dir, log_dir, LOG_basename, list_files_to_search, singlepulse_search_flags, num_simultaneous_singlepulse_searches, presto_env=os.environ['PRESTO'], verbosity_level=1, flag_singlepulse_search=1):
        if verbosity_level >= 2:
                print()
                print("BEGIN SINGLE PULSE SEARCH ----------------------------------------------------------------------")

        N_files_to_search = len(list_files_to_search)

        log_abspath = os.path.join(log_dir, LOG_basename+".txt")
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
        if num_simultaneous_singlepulse_searches == 1:
                for i in range(N_files_to_search):
                        dat_file = list_files_to_search[i]
                        cmd_singlepulse_search = "single_pulse_search.py --noplot %s %s" % (singlepulse_search_flags, dat_file)
                        print("Running '%s'... " % cmd_singlepulse_search, end=""); sys.stdout.flush()
                        execute_and_log(cmd_singlepulse_search, work_dir, log_abspath, dict_env, 0)
                        print("done!"); sys.stdout.flush()
        elif num_simultaneous_singlepulse_searches >= 2 and flag_singlepulse_search == 1:
                list_singlepulse_search_commands = []
                list_singlepulse_search_workdirs = []
                
                print("\nSingle-pulse search with multiple CPUs active")

                for i in range(N_files_to_search):
                        print()
                        if verbosity_level >= 2:
                                print("singlepulse_search: inside loop with i = %d / %d" % (i, N_files_to_search-1))
                        dat_file = list_files_to_search[i]
                        dat_file_nameonly = os.path.basename(dat_file)
                        
                        #DM_trial_was_searched = check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level)


                        
                        cmd_singlepulse_search = "single_pulse_search.py --noplot %s %s" % (singlepulse_search_flags, dat_file)
                        list_singlepulse_search_commands.append(cmd_singlepulse_search)
                        list_singlepulse_search_workdirs.append(work_dir)
                print()


                TP = ThreadPool(num_simultaneous_singlepulse_searches)
                N_commands = len(list_singlepulse_search_commands)
                print()
                print("Now doing parallelized single-pulse search using %d CPUs..." % num_simultaneous_singlepulse_searches);  sys.stdout.flush()
                print()
                for k in range(len(list_singlepulse_search_commands)):
                        print("Queing single-pulse search command #%d: '%s'" % (k+1, list_singlepulse_search_commands[k]))
                        time.sleep(0.1)
                        TP.apply_async(execute_and_log_in_thread_pool, (list_singlepulse_search_commands[k], log_dir, list_singlepulse_search_workdirs[k], k, N_commands, 1) )
                print("\n")
                print("Running %d single-pulse search commands at once..." % (num_simultaneous_singlepulse_searches)); sys.stdout.flush()
                TP.close()
                TP.join()
                print()
                print("%d commands completed!" % (len(list_singlepulse_search_commands)))


        print("Making final single-pulse plot with 'single_pulse_search.py *.singlepulse'...", end=''); sys.stdout.flush()
        os.system('single_pulse_search.py *.singlepulse')
        print("done!"); sys.stdout.flush()


        

def periodicity_search_FFT(work_dir, log_dir, LOG_basename, zapfile, segment_label, chunk_label, list_seg_ck_indices, list_DD_scheme, flag_use_cuda=0, list_cuda_ids=[0], flag_acceleration_search=1, numharm=8, list_zmax=[20], flag_jerk_search=1, jerksearch_zmax=0, jerksearch_wmax=0, jerksearch_numharm=4, num_simultaneous_jerksearches=1, period_to_search_min_s=0.001, period_to_search_max_s=20.0, other_flags_accelsearch="", flag_remove_fftfiles=0, flag_remove_datfiles_of_segments=0, presto_env_zmax_0=os.environ['PRESTO'], presto_env_zmax_any=os.environ['PRESTO'], flag_LOG_append=1, dict_flag_steps={'flag_step_dedisperse': 1, 'flag_step_realfft': 1, 'flag_step_periodicity_search': 1}):

        i_seg, N_seg, i_ck, N_ck = list_seg_ck_indices  # 分段和块的索引
        print("periodicity_search_FFT:: 需要搜索的文件：", "%s/*DM*.*.dat" % (work_dir))
        list_files_to_search = sorted([x for x in glob.glob("%s/*DM*.*.dat" % (work_dir))])  # 获取所有需要搜索的文件
        N_DMs_to_search = 0
        for k in range(len(list_DD_scheme)):  # 计算需要搜索的DM数量
                N_DMs_to_search = N_DMs_to_search + list_DD_scheme[k]['num_DMs']
        
        N_files_to_search = len(list_files_to_search)  # 需要搜索的文件总数
        N_files_searched = N_DMs_to_search - N_files_to_search  # 已搜索的文件数量
        
        frequency_to_search_max = 1. / period_to_search_min_s  # 最大频率
        frequency_to_search_min = 1. / period_to_search_max_s  # 最小频率

        print("最小搜索频率, ", frequency_to_search_min)
        print("最大搜索频率, ", frequency_to_search_max)
        print("periodicity_search_FFT:: 警告：当前版本中 -flo 和 -fhi 参数已被禁用")

        dict_env_zmax_0 = {'PRESTO': presto_env_zmax_0, 'PATH': "%s/bin:%s" % (presto_env_zmax_0, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env_zmax_0, os.environ['LD_LIBRARY_PATH'])}
        dict_env_zmax_any = {'PRESTO': presto_env_zmax_any, 'PATH': "%s/bin:%s" % (presto_env_zmax_any, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env_zmax_any, os.environ['LD_LIBRARY_PATH'])}

        # print("periodicity_search_FFT:: dict_env_zmax_0 = ", dict_env_zmax_0)
        # print("periodicity_search_FFT:: dict_env_zmax_any = ", dict_env_zmax_any)
        # print("periodicity_search_FFT:: LOG文件名 = ", LOG_basename)
        # print("periodicity_search_FFT:: 需要搜索的文件列表 = ", list_files_to_search)

        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)  # 日志文件的绝对路径
        print("\033[1m >> 提示：\033[0m 可以通过以下命令实时查看周期性搜索的日志：\033[1mtail -f %s\033[0m" % (log_abspath))
        zapfile_nameonly = os.path.basename(zapfile)  # 获取zap文件的文件名

        # #########################################################################################################
        # #                                     非并行化的 Jerk 搜索
        # #########################################################################################################
        # if num_simultaneous_jerksearches == 1 or jerksearch_wmax == 0 or flag_jerk_search == 0:
        #         for i in range(N_files_to_search):
        #                 print("\n周期性搜索 FFT：在循环中，i = %d / %d" % (i, N_files_to_search-1))
        #                 dat_file = list_files_to_search[i]
        #                 dat_file_nameonly = os.path.basename(dat_file)
        #                 fft_file = dat_file.replace(".dat", ".fft")
        #                 fft_file_nameonly = os.path.basename(fft_file)

        #                 DM_trial_was_searched = check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level)
        #                 # print("周期性搜索 FFT：DM_trial_was_searched =", DM_trial_was_searched)
                        
        #                 if dict_flag_steps['flag_step_realfft'] == 1:

        #                         if DM_trial_was_searched == False:
        #                                 print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 realfft..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
        #                                 sys.stdout.flush()
        #                                 realfft(dat_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, 0, flag_LOG_append)
        #                                 print("完成！")
        #                                 sys.stdout.flush()

        #                                 if flag_remove_datfiles_of_segments == 1 and (segment_label != "full") and os.path.exists(dat_file):
        #                                         if verbosity_level >= 1:
        #                                                 print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 删除 %s 以节省磁盘空间（使用 \"%s\" 可重新创建它）..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly, dat_file_nameonly+".makecmd"), end=' ')
        #                                                 sys.stdout.flush()
        #                                         os.remove(dat_file)
        #                                         if verbosity_level >= 1:
        #                                                 print("完成！")
        #                                                 sys.stdout.flush()

        #                                 print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 rednoise..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
        #                                 sys.stdout.flush()
        #                                 rednoise(fft_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, verbosity_level)
        #                                 print("完成！")
        #                                 sys.stdout.flush()

        #                                 print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在将消噪文件 '%s' 应用到 '%s'..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapfile_nameonly, fft_file_nameonly), end=' ')
        #                                 sys.stdout.flush()
        #                                 zapped_fft_filename, zapped_inf_filename = zapbirds(fft_file, zapfile, work_dir, log_dir, LOG_basename, presto_env_zmax_0, verbosity_level)
        #                                 zapped_fft_nameonly = os.path.basename(zapped_fft_filename)
        #                                 print("完成！")
        #                                 sys.stdout.flush()
        #                         else:
        #                                 print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 文件 '%s' 已成功搜索过，跳过..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly)); sys.stdout.flush()
                                        
        #                 else:
        #                         print("STEP_REALFFT = 0，跳过 realfft、rednoise、zapbirds...")

        #                 # print "\033[1m >> 提示：\033[0m 使用 '\033[1mtail -f %s\033[0m' 跟踪 accelsearch 日志" % (log_abspath)

        #                 if dict_flag_steps['flag_step_periodicity_search'] == 1:
        #                         if DM_trial_was_searched == False:
        #                                 if flag_acceleration_search == 1:
        #                                         for z in list_zmax:
        #                                                 tstart_accelsearch = time.time()
        #                                                 print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行加速度搜索，zmax = %4d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, z), end=' ')
        #                                                 sys.stdout.flush()
        #                                                 if int(z) == 0:
        #                                                         dict_env = copy.deepcopy(dict_env_zmax_0)
        #                                                         flag_cuda = ""
        #                                                 else:
        #                                                         if flag_use_cuda == 1:
        #                                                                 dict_env = copy.deepcopy(dict_env_zmax_any)
        #                                                                 gpu_id = random.choice(list_cuda_ids)
        #                                                                 flag_cuda = " -cuda %d " % (gpu_id)
        #                                                         else:
        #                                                                 dict_env = copy.deepcopy(dict_env_zmax_0)
        #                                                                 flag_cuda = ""

        #                                                 accelsearch_flags = other_flags_accelsearch + flag_cuda  # + " -flo %s -fhi %s" % (frequency_to_search_min, frequency_to_search_max)

        #                                                 accelsearch(fft_file, work_dir, log_abspath, numharm=numharm, zmax=z, other_flags=accelsearch_flags, dict_env=dict_env, verbosity_level=verbosity_level, flag_LOG_append=flag_LOG_append)
        #                                                 tend_accelsearch = time.time()
        #                                                 time_taken_accelsearch_s = tend_accelsearch - tstart_accelsearch
        #                                                 print("完成，耗时 %.2f 秒！" % (time_taken_accelsearch_s))
        #                                                 sys.stdout.flush()
        #                                                 ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s" % (int(z)))

        #                                 if jerksearch_wmax > 0 and flag_jerk_search == 1:
        #                                         tstart_jerksearch = time.time()
        #                                         print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 Jerk 搜索，参数为 zmax=%d, wmax=%d, numharm=%d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm), end=' ')
        #                                         sys.stdout.flush()
        #                                         flag_cuda = ""
        #                                         jerksearch_flags = other_flags_accelsearch + flag_cuda
        #                                         jerksearch(fft_file, work_dir, log_abspath, numharm=jerksearch_numharm, zmax=jerksearch_zmax, wmax=jerksearch_wmax, other_flags=jerksearch_flags, dict_env=dict_env_zmax_0, verbosity_level=verbosity_level, flag_LOG_append=flag_LOG_append)
        #                                         tend_jerksearch = time.time()
        #                                         time_taken_jerksearch_s = tend_jerksearch - tstart_jerksearch
        #                                         print("完成，耗时 %.2f 秒！" % (time_taken_jerksearch_s))
        #                                         sys.stdout.flush()
        #                                         ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s_JERK_%s" % (jerksearch_zmax, jerksearch_wmax))

        #                         else:   
        #                                 print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 文件 '%s' 已成功搜索过，跳过..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' '); sys.stdout.flush()

        #                 if flag_remove_fftfiles == 1 and os.path.exists(fft_file):
        #                         print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 删除 %s 以节省磁盘空间..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, fft_file_nameonly), end=' '); sys.stdout.flush()
        #                         os.remove(fft_file)  # 删除 FFT 文件
        #                         print("完成！");  sys.stdout.flush()


        #########################################################################################################
        #                                     并行化的 Jerk 搜索
        # 如果我们使用多个 CPU 进行 Jerk 搜索，方案会有所不同
        # 我们将首先对所有 dat 文件进行 realfft、去红噪声和消噪处理，然后并行搜索所有 .fft 文件
        #########################################################################################################
        
        # elif num_simultaneous_jerksearches >= 2 and jerksearch_wmax > 0 and flag_jerk_search == 1:
        if 1:
                list_jerksearch_commands = []
                list_jerksearch_workdirs = []
                jerksearch_flags = other_flags_accelsearch
                print("\n正在使用多个 CPU 进行 Jerk 搜索")

                for i in range(N_files_to_search):
                        print()
                        print("周期性搜索 FFT：在循环中，i = %d / %d" % (i, N_files_to_search-1))
                        dat_file = list_files_to_search[i]
                        dat_file_nameonly = os.path.basename(dat_file)
                        fft_file = dat_file.replace(".dat", ".fft")
                        fft_file_nameonly = os.path.basename(fft_file)

                        DM_trial_was_searched = check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax)

                        if dict_flag_steps['flag_step_realfft'] == 1:

                                if DM_trial_was_searched == False:
                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 realfft..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        realfft(dat_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, 0, flag_LOG_append)
                                        print("完成！")
                                        sys.stdout.flush()

                                        if flag_remove_datfiles_of_segments ==1 and (segment_label != "full") and os.path.exists(dat_file):
                                                print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 删除 %s 以节省磁盘空间（使用 \"%s\" 可重新创建它）..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly, dat_file_nameonly+".makecmd"), end=' ')
                                                sys.stdout.flush()
                                                os.remove(dat_file)
                                                print("完成！")
                                                sys.stdout.flush()

                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 rednoise..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        rednoise(fft_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0)
                                        print("完成！")
                                        sys.stdout.flush()

                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在将消噪文件 '%s' 应用到 '%s'..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapfile_nameonly, fft_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        zapped_fft_filename, zapped_inf_filename = zapbirds(fft_file, zapfile, work_dir, log_dir, LOG_basename, presto_env_zmax_0, verbosity_level)
                                        zapped_fft_nameonly = os.path.basename(zapped_fft_filename)
                                        print("完成！")
                                        sys.stdout.flush()
                                else:
                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 已完全搜索过。跳过..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search), end=' ')
                        else:
                                print("STEP_REALFFT = 0，跳过 realfft、rednoise、zapbirds...")

                        if dict_flag_steps['flag_step_periodicity_search'] == 1:
                                if DM_trial_was_searched == False:
                                        if flag_acceleration_search == 1:
                                                for z in list_zmax:
                                                        tstart_accelsearch = time.time()
                                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行加速度搜索，zmax = %4d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, z), end=' ')
                                                        sys.stdout.flush()
                                                        if int(z) == 0:
                                                                dict_env = copy.deepcopy(dict_env_zmax_0)
                                                                flag_cuda = ""
                                                        else:
                                                                if flag_use_cuda == 1:
                                                                        dict_env = copy.deepcopy(dict_env_zmax_any)
                                                                        gpu_id = random.choice(list_cuda_ids)
                                                                        flag_cuda = " -cuda %d " % (gpu_id)
                                                                else:
                                                                        dict_env = copy.deepcopy(dict_env_zmax_0)
                                                                        flag_cuda = ""

                                                        accelsearch_flags = other_flags_accelsearch + flag_cuda  # + " -flo %s -fhi %s" % (frequency_to_search_min, frequency_to_search_max)

                                                        accelsearch(fft_file, work_dir, log_abspath, numharm=numharm, zmax=z, other_flags=accelsearch_flags, dict_env=dict_env, verbosity_level=verbosity_level, flag_LOG_append=flag_LOG_append)
                                                        tend_accelsearch = time.time()
                                                        time_taken_accelsearch_s = tend_accelsearch - tstart_accelsearch
                                                        print("完成，耗时 %.2f 秒！" % (time_taken_accelsearch_s)); sys.stdout.flush()
                                                        ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s" % (int(z)))

                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 将在本块的加速度搜索结束后，对 %s 进行 Jerk 搜索，参数为 zmax=%d, wmax=%d, numharm=%d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm), end=' ')
                                        cmd_jerksearch = "accelsearch %s -zmax %d -wmax %d -numharm %d %s" % (jerksearch_flags, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm, fft_file)
                                        list_jerksearch_commands.append(cmd_jerksearch)
                                        list_jerksearch_workdirs.append(work_dir)
                                        print()

                                
                TP = ThreadPool(num_simultaneous_jerksearches)
                N_commands = len(list_jerksearch_commands)
                print()
                print("现在开始使用 %d 个 CPU 进行并行化的 Jerk 搜索..." % num_simultaneous_jerksearches);  sys.stdout.flush()
                print()
                for k in range(len(list_jerksearch_commands)):
                        print("正在排队第 %d 条 Jerk 搜索命令：'%s'" % (k+1, list_jerksearch_commands[k]))
                        time.sleep(0.1)
                        TP.apply_async(execute_and_log_in_thread_pool, (list_jerksearch_commands[k], log_dir, list_jerksearch_workdirs[k], k, N_commands, 1) )
                print("\n")
                print("同时运行 %d 条 Jerk 搜索命令..." % (num_simultaneous_jerksearches)); sys.stdout.flush()
                TP.close()
                TP.join()
                print()
                print("共完成 %d 条命令！" % (len(list_jerksearch_commands)))


                                        
def make_birds_file(ACCEL_0_filename, out_dir, log_dir, log_filename, width_Hz, flag_grow=1, flag_barycentre=0, sigma_birdies_threshold=4, verbosity_level=0):
        infile_nameonly = os.path.basename(ACCEL_0_filename)  # 获取输入文件的文件名
        infile_basename = infile_nameonly.replace("_ACCEL_0", "")  # 去掉 "_ACCEL_0" 后缀
        birds_filename = ACCEL_0_filename.replace("_ACCEL_0", ".birds")  # 生成鸟频文件名
        log_file = open(log_filename, "a")  # 打开日志文件

        # 跳过文件的前三行
        print("正在打开候选文件：%s" % (ACCEL_0_filename))
        candidate_birdies = sifting.candlist_from_candfile(ACCEL_0_filename)  # 从候选文件中读取候选鸟频
        candidate_birdies.reject_threshold(sigma_birdies_threshold)  # 根据阈值筛选候选鸟频

        # 写入超过特定信噪比阈值的候选鸟频
        list_birdies = candidate_birdies.cands
        print("鸟频数量 = %d" % (len(list_birdies)))
        file_birdies = open(birds_filename, "w")  # 打开鸟频文件
        print("鸟频文件路径：%s" % (birds_filename))
        for cand in list_birdies:  # 写入鸟频文件
                file_birdies.write("%.3f     %.20f     %d     %d     %d\n" % (cand.f, width_Hz, cand.numharm, flag_grow, flag_barycentre))
        file_birdies.close()
        return birds_filename  # 返回鸟频文件路径


def get_Fourier_bin_width(fft_infile):
        inffile_name = fft_infile.replace(".fft", ".inf")
        inffile = infodata.infodata(inffile_name)
        Tobs_s = inffile.dt * inffile.N
        fourier_bin_width_Hz = 1./Tobs_s

        return fourier_bin_width_Hz





def check_prepdata_outfiles(basename, verbosity_level=0):
        dat_filename  = basename + ".dat"
        inf_filename = basename + ".inf"
        try:
                if (os.path.getsize(dat_filename) > 0) and (os.path.getsize(inf_filename) >0): #checks if it exists and its
                        return True
                else:
                        return False
        except OSError:
                return False








def make_rfifind_mask(infile, out_dir, log_dir, LOG_basename, time=0.1, time_intervals_to_zap="", chans_to_zap="", other_flags="", presto_env=os.environ['PRESTO'],search_type= None,obsname = '*fits'):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        parent_dir = os.path.dirname(infile)
        if search_type:
               infile_basename = 'rfi0.1s'
               infile = parent_dir+ '/'+obsname

        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)

        flag_zapints = ""
        flag_zapchan = ""
        if time_intervals_to_zap != "":
                flag_zapints = "-zapints %s" %  (time_intervals_to_zap)
        if chans_to_zap != "":
                flag_zapchan = "-zapchan %s" %  (chans_to_zap)

        cmd_rfifind = "rfifind %s -o %s -time %s %s %s %s" % (other_flags, infile_basename, time, flag_zapints, flag_zapchan, infile)

        sys.stdout.flush()
        print("%s" % (cmd_rfifind))
        sys.stdout.flush()

        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
        #execute_and_log(cmd_rfifind, out_dir, log_abspath, dict_env, 0)
        file_to_check = "%s/%s_rfifind.%s" % (out_dir, infile_basename, 'mask')
        run_cmd(cmd_rfifind,ifok=file_to_check,work_dir=out_dir,log_file=log_abspath,dict_envs=dict_env,flag_append=None)

        if check_rfifind_outfiles(out_dir, infile_basename) == True:
                print("make_rfifind_mask:: %s | rfifind on \"%s\" completed successfully!\n" % (datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
                sys.stdout.flush()
        else:
                print("WARNING (%s) | could not find all the output files from rfifind on \"%s\"!" % (datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
                sys.stdout.flush()
                raise Exception("Your STEP_RFIFIND flag is set to 0, but the rfifind files could not be found!")

        mask_file = "%s/%s_rfifind.mask" % (out_dir, infile_basename)
        result = get_rfifind_result(mask_file, log_abspath)




def check_prepsubband_result_single_scheme(work_dir, DD_scheme, verbosity_level=1):
    # 遍历当前去色散方案中的所有 DM 值
    for dm in np.arange(DD_scheme['loDM'], DD_scheme['highDM'] - 0.5*DD_scheme['dDM'], DD_scheme['dDM']):
        # if verbosity_level >= 2:
        #     # 打印正在检查的文件路径
        #     print("check_prepsubband_result_single_scheme:: 正在查找：", [os.path.join(work_dir, "*DM%.2f.dat" % (dm))], [os.path.join(work_dir, "*DM%.2f.inf" % (dm))])
        #     # 打印实际找到的文件
        #     print("check_prepsubband_result_single_scheme:: 找到以下文件： %s, %s" % (
        #         [x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.dat" % (dm))) if not "_red" in x],
        #         [x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.inf" % (dm))) if not "_red" in x]
        #     ))

        # 检查是否存在对应的 .dat 和 .inf 文件（排除 "_red" 文件）
        if len(
            [x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.dat" % (dm))) if not "_red" in x] +
            [x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.inf" % (dm))) if not "_red" in x]
        ) != 2:
        #     if verbosity_level >= 2:
        #         print("check_prepsubband_result_single_scheme: 返回 False")
            return False
        
    return True










def check_zapbirds_outfiles2(zapped_fft_filename, verbosity_level=0):
        zapped_inf_filename = zapped_fft_filename.replace(".fft", ".inf")

        if ("zapped" in zapped_fft_filename) and ("zapped" in zapped_inf_filename):
                try:
                        if (os.path.getsize(zapped_fft_filename) > 0) and (os.path.getsize(zapped_inf_filename) >0): #checks if it exists and its size is > 0
                                return True
                        else:
                                return False
                except OSError:
                        return False
        else:
                return False


def check_zapbirds_outfiles(fftfile, list_zapped_ffts_abspath, verbosity_level=0):
        fftfile_nameonly = os.path.basename(fftfile)
        try:
                file_list_zapped_ffts = open(list_zapped_ffts_abspath, 'r')
                if "%s\n" % (fftfile_nameonly) in file_list_zapped_ffts.readlines():
                        if verbosity_level >= 1:
                                print("check_zapbirds_outfiles:: NB: File '%s' is already in the list of zapped files (%s)." % (fftfile_nameonly, list_zapped_ffts_abspath))
                        if (os.path.getsize(fftfile) > 0):
                                if verbosity_level >= 1:
                                        print("check_zapbirds_outfiles:: size is > 0. Returning True...")
                                return True
                        else:
                                if verbosity_level >= 1:
                                        print("rednoise:: size is = 0. Returning False...")
                                return False
                else:
                        if verbosity_level >= 1:
                                print("check_zapbirds_outfiles:: File '%s' IS NOT in the list of zapped files (%s). I will zap the file from scratch..." % (fftfile_nameonly, list_zapped_ffts_abspath))
                        return False
        except:
                if verbosity_level >= 1:
                        print("check_zapbirds_outfiles:: File '%s' does not exist. Creating it and returning False..." % (list_zapped_ffts_abspath))
                os.mknod(list_zapped_ffts_abspath)
                return False





def dedisperse_rednoise_and_periodicity_search_FFT(infile, out_dir, root_workdir, log_dir, LOG_basename, flag_search_full, segment_label, chunk_label, list_seg_ck_indices, zapfile, Nsamples, ignorechan_list, mask_file, list_DD_schemes, nchan, subbands=0, num_simultaneous_prepsubbands=1, other_flags_prepsubband="", presto_env_prepsubband=os.environ['PRESTO'], flag_use_cuda=0, list_cuda_ids=[0], flag_acceleration_search=1, numharm=8, list_zmax=[20], flag_jerk_search=0, jerksearch_zmax=10, jerksearch_wmax=30, jerksearch_numharm=4, num_simultaneous_jerksearches=1, period_to_search_min_s=0.001, period_to_search_max_s=20.0, other_flags_accelsearch="", flag_remove_fftfiles=0, flag_remove_datfiles_of_segments=0, presto_env_accelsearch_zmax_0=os.environ['PRESTO'], presto_env_accelsearch_zmax_any=os.environ['PRESTO'], verbosity_level=0, dict_flag_steps = {'flag_step_dedisperse': 1 , 'flag_step_realfft': 1, 'flag_step_periodicity_search': 1}):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]

        # 打印去色散操作的启动信息
        print("dedisperse_rednoise_and_periodicity_search_FFT:: 正在启动去色散操作")
        sys.stdout.flush()
        # 打印当前的list_zmax参数值
        print("dedisperse_rednoise_and_periodicity_search_FFT:: list_zmax = ", list_zmax)

        if dict_flag_steps['flag_step_dedisperse'] == 1:
                if segment_label == "full":
                        dedisperse(infile, out_dir, log_dir, LOG_basename, segment_label, chunk_label, Nsamples, ignorechan_list, mask_file, list_DD_schemes, nchan, subbands, num_simultaneous_prepsubbands, other_flags_prepsubband, presto_env_prepsubband, verbosity_level)
                else:

                        search_string = "%s/03_DEDISPERSION/%s/full/ck00/*.dat" % (root_workdir, infile_basename) #List of datfiles to split
                        list_datfiles_to_split = glob.glob(search_string)

                        # print("dedisperse_rednoise_and_periodicity_search_FFT:: 段标签: '%s'" % (segment_label))
                        # print("搜索字符串 = ", search_string)
                        # print("需要分割的dat文件列表 = ", list_datfiles_to_split)

                        segment_min = np.float64(segment_label.replace("m", ""))
                        i_chunk = int(chunk_label.replace("ck", ""))
                        split_into_chunks(infile, list_datfiles_to_split, log_dir, LOG_basename, out_dir, segment_min, i_chunk, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, presto_env=os.environ['PRESTO'], flag_LOG_append=1, flag_remove_datfiles_of_segments=flag_remove_datfiles_of_segments, verbosity_level=verbosity_level)


                print("dedisperse_rednoise_and_periodicity_search_FFT:: 正在启动周期性搜索")
                sys.stdout.flush()
        else:
                print("dedisperse_rednoise_and_periodicity_search_FFT:: STEP_DEDISPERSE = 0, 跳过预处理子带...")

        if not (segment_label == "full" and flag_search_full == 0):
                periodicity_search_FFT(out_dir, log_dir, LOG_basename, zapfile, segment_label, chunk_label, list_seg_ck_indices, list_DD_schemes, flag_use_cuda, list_cuda_ids, flag_acceleration_search, numharm, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm, num_simultaneous_jerksearches, period_to_search_min_s, period_to_search_max_s, other_flags_accelsearch, flag_remove_fftfiles, flag_remove_datfiles_of_segments, presto_env_accelsearch_zmax_0, presto_env_accelsearch_zmax_any, verbosity_level, 1, dict_flag_steps)

def check_if_cand_is_known(candidate, list_known_pulsars, numharm):
    # 遍历已知脉冲星的周期列表，检查候选信号是否为已知脉冲星
    P_cand_ms = candidate.p * 1000  # 将候选信号周期转换为毫秒
    BOLD = '\033[1m'
    END = '\033[0m'

    for i in range(len(list_known_pulsars)):
        psrname = list_known_pulsars[i].psr_name  # 当前已知脉冲星的名称

        P_ms = list_known_pulsars[i].P0_ms  # 当前已知脉冲星的周期（毫秒）
        P_ms_min = P_ms * (1 - list_known_pulsars[i].doppler_factor)  # 考虑多普勒效应后的最小周期
        P_ms_max = P_ms * (1 + list_known_pulsars[i].doppler_factor)  # 考虑多普勒效应后的最大周期

        if (P_cand_ms > P_ms_min) and (P_cand_ms < P_ms_max):
            # 如果候选信号周期在已知脉冲星周期范围内，标记为已知脉冲星
            str_harm = "基频 (%.7f ms)" % (P_ms)
            return True, psrname, str_harm

        else:
            # 检查候选信号是否为已知脉冲星的谐波
            for nh in range(1, numharm + 1):
                for n in range(1, 16 + 1):
                    P_known_ms_nh_min = P_ms_min * (np.float64(n) / nh)
                    P_known_ms_nh_max = P_ms_max * (np.float64(n) / nh)

                    if (P_cand_ms >= P_known_ms_nh_min) and (P_cand_ms <= P_known_ms_nh_max):
                        # 如果候选信号是已知脉冲星的谐波，返回相关信息
                        str_harm = "%d/%d 的谐波 of %.7f ms" % (n, nh, P_ms)
                        return True, psrname, str_harm

            # 检查候选信号是否为已知脉冲星的亚谐波
            for ns in range(2, numharm + 1):
                for n in range(1, 16 + 1):
                    P_known_ms_ns_min = P_ms_min * (np.float64(ns) / n)
                    P_known_ms_ns_max = P_ms_max * (np.float64(ns) / n)

                    if (P_cand_ms >= P_known_ms_ns_min) and (P_cand_ms <= P_known_ms_ns_max):
                        # 如果候选信号是已知脉冲星的亚谐波，返回相关信息
                        str_harm = "%d/%d 的亚谐波 of %.7f ms" % (ns, n, P_ms)
                        return True, psrname, str_harm

    # 如果候选信号既不是已知脉冲星的基频，也不是谐波或亚谐波，返回False
    return False, "", ""






