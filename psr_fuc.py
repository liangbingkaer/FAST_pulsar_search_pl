
import sys
import os
import os.path
import glob
import subprocess
import multiprocessing
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

def print_log(*args, sep=' ', end='\n', file=None, flush=False, log_files=None, masks=None, color=None):
    """
    打印并记录日志，支持高亮显示特定内容或整个消息。
    masks: 需要高亮显示的内容（如果为 None 且 color 有值，则整个消息高亮）
    color: 高亮显示的颜色代码（如 colors.ERROR）
    """
    default_dir = os.path.join(os.getcwd(), 'logall.txt')
    if log_files is None:
        log_files = [default_dir]
    elif isinstance(log_files, str):
        log_files = [log_files, default_dir]
    else:
        log_files = list(log_files) + [default_dir]

    message = sep.join(str(arg) for arg in args) + end

    for file_path in log_files:
        write2file(message, file_path, add_newline=False)

    if color:
        if masks:
            highlighted_message = message.replace(masks, f"{color}{masks}{colors.ENDC}")
        else:
            highlighted_message = f"{color}{message}{colors.ENDC}"
        print(highlighted_message, end='', file=file or sys.stdout, flush=flush)
    else:
        print(message, end='', file=file or sys.stdout, flush=flush)

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

def time_consum(start_time,cmd):
    """计算并记录执行时间"""
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = format_execution_time(execution_time)
    
    log_message = f"运行时间： {execution_time_str}"
    print_log(f'运行命令：{cmd}',masks=cmd,color=colors.OKCYAN,log_files='logruntime.txt')
    print_log(log_message,masks=log_message,color=colors.OKCYAN,log_files='logruntime.txt')
    time.sleep(2)

# 记录程序开始和结束
def get_current_time_to_minute():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

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

def run_cmd(cmd, ifok=None, work_dir=None, log_file=None, dict_envs={}, flag_append=True):
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
        print_log(f'File {ifok} exists. Skipping command: {cmd}\n', log_file)
        return

    start_time = time.time()
    datetime_start = datetime.now().strftime("%Y/%m/%d %H:%M")
    cwd = os.getcwd()

    if work_dir:
        os.chdir(work_dir)
    else:
        work_dir = cwd

    log_mode = "a" if flag_append else "w"
    log_handle = open(log_file, log_mode) if log_file else None

    print_log(f'程序运行路径为: {work_dir}', log_file)
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
    time_consum(start_time,cmd=cmd)

    if ifok and ifok.endswith(('.txt', '.ifok')):
        with open(ifok, 'a') as f:
            f.write(f"Command executed: {cmd}\n")
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

def check_if_enough_disk_space(root_workdir, num_DMs, T_obs_s, t_samp_s, list_segments_nofull, flag_remove_fftfiles, flag_remove_datfiles_of_segments):
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

    if flag_remove_datfiles_of_segments == 0:
        print_log("是否删除分段的 .dat 文件？否  --> 总占用空间将非常高")
    else:
        print_log("是否删除分段的 .dat 文件？是  --> 分段搜索的磁盘空间占用将可以忽略不计")
    size_G = f"{full_length_search_size_bytes / 1.0e9:4.2f}" 
    print_log(f"全长度搜索：~{size_G} GB       ({num_DMs} DM 试验 * 每次试验 {datfile_full_size_bytes / 1.0e6:5.0f} MB)",masks=size_G,color=colors.OKGREEN)

    # 初始化总搜索所需空间
    total_search_size_bytes = full_length_search_size_bytes

    # 遍历分段长度列表，计算每个分段搜索所需的磁盘空间
    for seg in list_segments_nofull:
        seg_length_s = np.float64(seg) * 60  # 分段长度（秒）
        N_chunks = int(T_obs_s / seg_length_s)  # 分段数量

        # 如果剩余部分超过分段长度的 80%，则增加一个分段
        fraction_left = (T_obs_s % seg_length_s) / seg_length_s
        if fraction_left >= 0.80:
            N_chunks = N_chunks + 1

        # 计算每个分段的数据文件大小
        if flag_remove_datfiles_of_segments == 0:
            N_samples_per_datfile_seg = int(seg_length_s / t_samp_s)
            datfile_seg_size_bytes = N_samples_per_datfile_seg * 4
        else:
            datfile_seg_size_bytes = 0

        if flag_remove_fftfiles == 0:
            datfile_seg_size_bytes = datfile_seg_size_bytes * 2

        # 计算分段搜索所需的磁盘空间
        seg_search_size_bytes = datfile_seg_size_bytes * num_DMs * N_chunks
        total_search_size_bytes = total_search_size_bytes + seg_search_size_bytes

        print("分段 %5s 分钟搜索：~%4d GB        (%d DM 试验 * 每次试验 %5d MB * %2d 分段)" % (seg, seg_search_size_bytes / 1.0e9, num_DMs, datfile_seg_size_bytes / 1.0e6, N_chunks))

    print()

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
    
def execute_and_log(command, work_dir, log_abspath, dict_envs={}, flag_append=0):
        datetime_start = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
        time_start = time.time()
        if flag_append == 1:
                flag_open_mode = "a"
        else:
                flag_open_mode = "w+"
        log_file = open("%s" % (log_abspath), flag_open_mode)
        executable = command.split()[0]

        # 写入日志文件的开头部分，包括启动时间和命令等信息
        log_file.write("****************************************************************\n")
        log_file.write("开始日期和时间：%s\n" % (datetime_start))  # 将 "START DATE AND TIME" 改为中文
        log_file.write("\n命令：\n")  # 将 "COMMAND" 改为中文
        log_file.write("%s\n\n" % (command))  # 输出执行的命令
        log_file.write("工作目录：%s\n" % (work_dir))  # 将 "WORKING DIRECTORY" 改为中文
        log_file.write("****************************************************************\n")
        log_file.flush()  # 刷新日志文件缓冲区，确保内容写入文件

        list_for_Popen = command.split()
        env_subprocess = os.environ.copy()
        if dict_envs:  # If the dictionary is not empty                                                                                                                                                            
                for k in list(dict_envs.keys()):
                        env_subprocess[k] = dict_envs[k]

        proc = subprocess.Popen(list_for_Popen, stdout=log_file, stderr=log_file, cwd=work_dir, env=env_subprocess)
        proc.communicate()  # Wait for the process to complete                                                                                                                                                    

        datetime_end = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
        time_end = time.time()

        #print("execute_and_log:: 命令：%s" % (command))  # 打印执行的命令
        #print("execute_and_log:: 找到的可执行文件路径：%s: " % (executable), get_command_output("which %s" % (executable)))  # 打印可执行文件路径
        #print("execute_and_log:: 工作目录 = ", work_dir)  # 打印工作目录
        #print("execute_and_log:: 查看日志的方式：\"tail -f %s\"" % (log_abspath))  # 提示如何查看日志
        #sys.stdout.flush()  # 刷新标准输出缓冲区
        #print("execute_and_log: 用于 Popen 的命令列表 = ", list_for_Popen)  # 打印用于子进程的命令列表
        #print("execute_and_log: 日志文件 = ", log_file)  # 打印日志文件对象
        #print("execute_and_log: 子进程环境变量 = ", env_subprocess)  # 打印子进程的环境变量

        log_file.write("\n结束日期和时间：%s\n" % (datetime_end))  # 写入结束时间和日期
        log_file.write("\n总耗时：%d 秒\n" % (time_end - time_start))  # 写入总耗时
        log_file.close()  # 关闭日志文件

        with open(os.path.join(os.getcwd(), 'cmd.sh'), 'a') as file:
                file.write(f'#程序运行路径为:{work_dir}  \n')
                file.write(command + '\n')

def execute_and_log_in_thread_pool(command, log_dir, work_dir, id_num, N_ids, flag_log=1):
    # 获取当前时间，用于日志记录
    datetime_start = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
    datetime_start_single_string = (datetime.now()).strftime("%Y%m%d_%H%M")
    time_start = time.time()

    # 提取命令的标签（用于日志文件命名）
    if "/" in command.split()[0]:
        command_label = command.split("/")[-1].split()[0]
    else:
        command_label = command.split()[0]

    # 将命令字符串分割为列表，用于传递给 subprocess.Popen
    list_for_Popen = command.split()

    # 遍历命令列表，处理包含通配符（? 或 *）的参数
    for i in range(len(list_for_Popen)):
        current_piece = list_for_Popen[i]
        if "?" in current_piece or "*" in current_piece:
            # 使用 glob 模块展开通配符，并替换原列表中的对应部分
            new_list_for_Popen = list_for_Popen[:i] + sorted(glob.glob(current_piece)) + list_for_Popen[i+1:]
            list_for_Popen = new_list_for_Popen

    # 根据 flag_log 决定是否记录日志
    if flag_log == 1:
        # 构造日志文件名并打开日志文件
        log_filename = "LOG_%s_%s_%03d.txt" % (command_label, datetime_start_single_string, int(id_num))
        log_abspath = os.path.join(log_dir, log_filename)
        log_file = open(log_abspath, "w+")

        # 写入日志文件的头部信息
        log_file.write("****************************************************************\n")
        log_file.write("开始日期和时间：%s\n" % (datetime_start))
        log_file.write("\n命令：\n")
        log_file.write("%s\n\n" % (command))
        log_file.write("工作目录：%s\n" % (work_dir))
        log_file.write("****************************************************************\n")
        log_file.flush()

        # 在指定工作目录下执行命令，并将输出重定向到日志文件
        proc = subprocess.Popen(list_for_Popen, cwd=work_dir, stdout=log_file, stderr=log_file)
    elif flag_log == 0:
        # 如果不记录日志，直接打印提示信息
        print("未记录日志...")
        proc = subprocess.Popen(list_for_Popen, cwd=work_dir, stdout=subprocess.PIPE)

    # 等待命令执行完成
    proc.communicate()

    # 获取结束时间和总耗时
    datetime_end = (datetime.now()).strftime("%Y/%m/%d  %H:%M")
    time_end = time.time()

    # 如果记录日志，写入结束时间和总耗时
    if flag_log == 1:
        log_file.write("\n结束日期和时间：%s\n" % (datetime_end))
        log_file.write("\n总耗时：%d 秒\n" % (time_end - time_start))
        log_file.close()

    # 打印命令执行完成的提示信息
    print("命令 %4d/%d ('%s') 执行完成。" % (id_num + 1, N_ids, command_label)); sys.stdout.flush()

def get_command_output(command, shell_state=False, work_dir=os.getcwd()):
        print_log(f'运行命令：{command}', masks=command ,color=colors.OKCYAN)
        time.sleep(0.8) 
        list_for_Popen = command.split()
        if shell_state ==False:
                proc = subprocess.Popen(list_for_Popen, stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        else:
                proc = subprocess.Popen([command], stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        out, err = proc.communicate()  
        append_to_script_if_not_exists(os.path.join(os.getcwd(), 'cmd.sh'),f'#程序运行路径为:{work_dir}')
        append_to_script_if_not_exists(os.path.join(os.getcwd(), 'cmd.sh'),command)   
        return out.decode('ascii')

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


def sift_candidates(work_dir, log_dir, LOG_basename,  dedispersion_dir, observation_basename, segment_label, chunk_label, list_zmax, jerksearch_zmax, jerksearch_wmax, flag_remove_duplicates, flag_DM_problems, flag_remove_harmonics, minimum_numDMs_where_detected, minimum_acceptable_DM=2.0, period_to_search_min_s=0.001, period_to_search_max_s=15.0, verbosity_level=0):
        verbosity_level = 3
        work_dir_basename = os.path.basename(work_dir)
        string_ACCEL_files_dir = os.path.join(dedispersion_dir, observation_basename, segment_label, chunk_label)

        best_cands_filename = "%s/best_candidates_%s.siftedcands" % (work_dir, work_dir_basename)
        # if verbosity_level >= 3:
        #         print("sift_candidates:: best_cands_filename = %s" % (best_cands_filename))
        #         print("sift_candidates:: string_ACCEL_files_dir = %s" % (string_ACCEL_files_dir))

        list_ACCEL_files = []
        for z in list_zmax:
                string_glob = "%s/*ACCEL_%d" % (string_ACCEL_files_dir, z)
                print("Reading files '%s'..." % (string_glob), end=' ')
                list_ACCEL_files = list_ACCEL_files + glob.glob(string_glob)
                print("done!")

        string_glob_jerk_files = "%s/*ACCEL_%d_JERK_%d" % (string_ACCEL_files_dir, jerksearch_zmax, jerksearch_wmax)
        # if verbosity_level >= 3:
        #         print("JERK: Also reading files '%s'.." % (string_glob_jerk_files))
        #         print("Found: ", glob.glob(string_glob_jerk_files))
        list_ACCEL_files = list_ACCEL_files + glob.glob(string_glob_jerk_files)
        # if verbosity_level >= 3:
        #         print()
        #         print("ACCEL files found: ", list_ACCEL_files)
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        print("\033[1m >> TIP:\033[0m Check sifting output with '\033[1mcat %s\033[0m'" % (log_abspath))

        list_DMs = [x.split("_ACCEL")[0].split("DM")[-1] for x in list_ACCEL_files]
        candidates = sifting.read_candidates(list_ACCEL_files, track=True)

        print("sift_candidates:: z = %d" % (z))
        print("sift_candidates:: %s/*ACCEL_%d" % (string_ACCEL_files_dir, z))
        #print("sift_candidates:: list_ACCEL_files = %s" % (list_ACCEL_files))
        #print("sift_candidates:: list_DMs = %s" % (list_DMs))
        #print("sift_candidates:: candidates.cands = ", candidates.cands)
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


        if len(candidates.cands) >= 1:  # 检查候选者列表是否为空

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

        # else:
        #        print "sift_candidates:: ERROR: len(candidates.cands) < 1!!! candidates = %s" % (candidates)
        #        exit()

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
                file_to_fold = os.path.join(dir_dedispersion, obs, seg, ck, cand.filename.split("_ACCEL")[0] + ".dat")
                cmd_prepfold = "prepfold %s -noxwin -accelcand %d -accelfile %s/%s.cand -o ts_fold_%s_%s_%s_DM%.2f_%s   %s" % (other_flags_prepfold, cand.candnum, dir_accelfile, cand.filename, obs, seg, ck, cand.DM, str_zmax_wmax, file_to_fold)
                execute_and_log(cmd_prepfold, work_dir, log_abspath, dict_env, flag_LOG_append)
        elif what_fold =="rawdata":
                file_to_fold = observation.file_abspath
                if seg == "full":
                        cmd_prepfold = "prepfold %s -noxwin -accelcand %d -accelfile %s/%s.cand -dm %.2f %s -mask %s -o raw_fold_%s_%s_%s_DM%.2f_%s    %s" % (other_flags_prepfold, cand.candnum, dir_accelfile, cand.filename, cand.DM, flag_ignorechan, observation.mask, obs, seg, ck, cand.DM, str_zmax_wmax, file_to_fold)
                else:
                        segment_min = np.float64(seg.replace("m", ""))
                        i_chunk = int(ck.replace("ck", ""))
                        T_obs_min = observation.T_obs_s / 60.
                        start_frac = (i_chunk * segment_min) / T_obs_min
                        end_frac = ((i_chunk + 1) * segment_min) / T_obs_min
                        if end_frac > 1:
                                end_frac = 1.0

                        cmd_prepfold = "prepfold %s -start %.5f -end %.5f -noxwin -accelcand %d -accelfile %s/%s.cand -dm %.2f %s -mask %s -o raw_fold_%s_%s_%s_DM%.2f_%s    %s" % (other_flags_prepfold, start_frac, end_frac, cand.candnum, dir_accelfile, cand.filename, cand.DM, flag_ignorechan, observation.mask, obs, seg, ck, cand.DM, str_zmax_wmax, file_to_fold)

                file_script_fold.write("%s\n" % cmd_prepfold)
                print(cmd_prepfold)

        # if verbosity_level >= 2:
        #         print("fold_candidates:: cand.filename: ",  cand.filename)
        #         print("file_to_fold = ", file_to_fold)
        #         print("fold_candidates:: cmd_prepfold = %s" % (cmd_prepfold))

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


def check_rfifind_outfiles(out_dir, basename):
        for suffix in ["bytemask", "inf", "mask", "ps", "rfi", "stats"]:
                file_to_check = "%s/%s_rfifind.%s" % (out_dir, basename, suffix)
                if not os.path.exists(file_to_check):
                        print("ERROR: file %s not found!" % (file_to_check))
                        return False
                elif os.stat(file_to_check).st_size == 0:  # If the file has size 0 bytes
                        print("ERROR: file %s has size 0!" % (file_to_check))
                        return False
        return True


def check_rednoise_outfiles(fftfile_rednoise_abspath, verbosity_level=0):
        inffile_rednoise_abspath = fftfile_rednoise_abspath.replace("_red.fft", "_red.inf")

        if os.path.exists(fftfile_rednoise_abspath ) and (os.path.getsize(fftfile_rednoise_abspath) > 0) and os.path.exists(inffile_rednoise_abspath) and (os.path.getsize(inffile_rednoise_abspath) > 0):
                return True
        else:
                return False


def check_accelsearch_result(fft_infile, zmax, verbosity_level=0):
        fft_infile_nameonly = os.path.basename(fft_infile)
        fft_infile_basename = os.path.splitext(fft_infile_nameonly)[0]

        if verbosity_level >= 2:
                print("check_accelsearch_result:: infile_basename: ", fft_infile_basename)
                print("check_accelsearch_result:: ACCEL_filename = ", ACCEL_filename)
                print("check_accelsearch_result:: ACCEL_cand_filename", ACCEL_cand_filename)
                print("check_accelsearch_result:: ACCEL_txtcand_filename = ", ACCEL_txtcand_filename)

        ACCEL_filename =  fft_infile.replace(".fft", "_ACCEL_%d" % (zmax))
        ACCEL_cand_filename =  fft_infile.replace(".fft", "_ACCEL_%d.cand" % (zmax))
        ACCEL_txtcand_filename =  fft_infile.replace(".fft", "_ACCEL_%d.txtcand" % (zmax))

        try:
                if (os.path.getsize(ACCEL_filename) > 0) and (os.path.getsize(ACCEL_cand_filename) > 0) and (os.path.getsize(ACCEL_txtcand_filename) > 0):
                        result_message = "check_accelsearch_result:: Files exist and their size is > 0! Skipping..."
                        check_result = True
                else:
                        result_message = "check_accelsearch_result:: Files exists but at least one of them has size = 0!"
                        check_result = False
        except OSError:
                result_message = "check_accelsearch_result:: OSError: It seems accelsearch has not been executed!"
                check_result = False

        if verbosity_level >= 1:
                print(result_message)

        return check_result


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


def accelsearch(infile, work_dir, log_abspath, numharm=8, zmax=0, other_flags="", dict_env={}, verbosity_level=0, flag_LOG_append=1):
        verbosity_level = 2
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        # 构造空结果文件的路径（用于标记未产生候选结果的情况）
        inffile_empty = infile.replace(".fft", "_ACCEL_%d_empty" % (zmax))

        # 构造 accelsearch 命令
        cmd_accelsearch = "accelsearch %s -zmax %s -numharm %s %s" % (other_flags, zmax, numharm, infile)

        print("\nBEGIN ACCELSEARCH ----------------------------------------------------------------------")
        print("accelsearch:: check_accelsearch_result(infile, int(zmax)) :: %s" % (check_accelsearch_result(infile, int(zmax))))

        # 检查是否已运行过 accelsearch
        if check_accelsearch_result(infile, int(zmax)) == False and check_accelsearch_result(inffile_empty, int(zmax)) == False:
                print("accelsearch:: 正在运行: %s" % (cmd_accelsearch))
                execute_and_log(cmd_accelsearch, work_dir, log_abspath, dict_env, flag_LOG_append)
        else:
                print("accelsearch:: 警告：accelsearch（zmax=%d）似乎已经对文件 %s 执行过。跳过..." % (int(zmax), infile_nameonly))
        if check_accelsearch_result(infile, int(zmax)) == False:
                file_empty = open(inffile_empty, "w")
                print("%s警告%s：accelsearch 没有产生任何候选结果！写入文件 %s 以标记此情况..." % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, inffile_empty), end='')
                file_empty.write("ACCELSEARCH DID NOT PRODUCE ANY CANDIDATES!")
        else:
                print("END ACCELSEARCH ---------------------------------------------------------------------- ")


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

                DM_trial_was_searched = check_if_DM_trial_was_searched(datfile_chunk_name, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level)

                
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


                                
def check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level=1):
        # 获取文件名和对应的 FFT 文件名
        verbosity_level = 1
        dat_file_nameonly = os.path.basename(dat_file)
        fft_file = dat_file.replace(".dat", ".fft")
        fft_file_nameonly = os.path.basename(fft_file)

        # print("check_if_DM_trial_was_searched:: list_zmax =", list_zmax)
        # print("check_if_DM_trial_was_searched:: flag_jerk_search =", flag_jerk_search)
        # print("check_if_DM_trial_was_searched:: jerksearch_zmax = {}, jerksearch_wmax = {}".format(jerksearch_zmax, jerksearch_wmax))

        # 遍历所有 zmax 值，检查对应的加速度搜索结果文件是否存在
        for z in list_zmax:
                ACCEL_filename = dat_file.replace(".dat", "_ACCEL_%s" % (int(z)))  # 加速度搜索结果文件
                ACCEL_filename_empty = dat_file.replace(".dat", "_ACCEL_%s_empty" % (int(z)))  # 空结果文件
                ACCEL_cand_filename = ACCEL_filename + ".cand"  # 候选文件
                ACCEL_txtcand_filename = ACCEL_filename + ".txtcand"  # 文本格式候选文件

        if verbosity_level >= 2:
            print("check_if_DM_trial_was_searched:: 正在检查: {}, {}, {}".format(ACCEL_filename, ACCEL_cand_filename, ACCEL_txtcand_filename))
            print("check_if_DM_trial_was_searched:: 正在检查: {}, {}, {}".format(ACCEL_filename_empty, ACCEL_cand_filename, ACCEL_txtcand_filename))

        # 检查文件是否存在或是否为空
        if (not os.path.exists(ACCEL_filename) or os.path.getsize(ACCEL_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if verbosity_level >= 2:
                print("check_if_DM_trial_was_searched:: 返回 False - 情况 1")
            return False
        if (not os.path.exists(ACCEL_cand_filename) or os.path.getsize(ACCEL_cand_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if verbosity_level >= 2:
                print("check_if_DM_trial_was_searched:: 返回 False - 情况 2")
            return False
        if not os.path.exists(ACCEL_txtcand_filename):
            if verbosity_level >= 2:
                print("check_if_DM_trial_was_searched:: 返回 False - 情况 3")
            return False

        # 如果启用了 Jerk 搜索，检查 Jerk 搜索的结果文件
        if flag_jerk_search == 1 and jerksearch_wmax > 0:
                ACCEL_filename = dat_file.replace(".dat", "_ACCEL_%s_JERK_%s" % (jerksearch_zmax, jerksearch_wmax))  # Jerk 搜索结果文件
                ACCEL_filename_empty = dat_file.replace(".dat", "_ACCEL_%s_JERK_%s_empty" % (jerksearch_zmax, jerksearch_wmax))  # 空结果文件
                ACCEL_cand_filename = ACCEL_filename + ".cand"  # 候选文件
                ACCEL_txtcand_filename = ACCEL_filename + ".txtcand"  # 文本格式候选文件

        # 检查文件是否存在或是否为空
        if (not os.path.exists(ACCEL_filename) or os.path.getsize(ACCEL_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if verbosity_level >= 2:
                print("check_if_DM_trial_was_searched:: 返回 False - 情况 4")
            return False
        if (not os.path.exists(ACCEL_cand_filename) or os.path.getsize(ACCEL_cand_filename) == 0) and \
           (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) == 0):
            if verbosity_level >= 2:
                print("check_if_DM_trial_was_searched:: 返回 False - 情况 5")
            return False
        if not os.path.exists(ACCEL_txtcand_filename):
            if verbosity_level >= 2:
                print("check_if_DM_trial_was_searched:: 返回 False - 情况 6")
            return False

        # 如果所有检查通过，返回 True
        return True


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


        

def periodicity_search_FFT(work_dir, log_dir, LOG_basename, zapfile, segment_label, chunk_label, list_seg_ck_indices, list_DD_scheme, flag_use_cuda=0, list_cuda_ids=[0], flag_acceleration_search=1, numharm=8, list_zmax=[20], flag_jerk_search=1, jerksearch_zmax=0, jerksearch_wmax=0, jerksearch_numharm=4, num_simultaneous_jerksearches=1, period_to_search_min_s=0.001, period_to_search_max_s=20.0, other_flags_accelsearch="", flag_remove_fftfiles=0, flag_remove_datfiles_of_segments=0, presto_env_zmax_0=os.environ['PRESTO'], presto_env_zmax_any=os.environ['PRESTO'], verbosity_level=0, flag_LOG_append=1, dict_flag_steps={'flag_step_dedisperse': 1, 'flag_step_realfft': 1, 'flag_step_periodicity_search': 1}):

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

        #########################################################################################################
        #                                     非并行化的 Jerk 搜索
        #########################################################################################################
        if num_simultaneous_jerksearches == 1 or jerksearch_wmax == 0 or flag_jerk_search == 0:
                for i in range(N_files_to_search):
                        print("\n周期性搜索 FFT：在循环中，i = %d / %d" % (i, N_files_to_search-1))
                        dat_file = list_files_to_search[i]
                        dat_file_nameonly = os.path.basename(dat_file)
                        fft_file = dat_file.replace(".dat", ".fft")
                        fft_file_nameonly = os.path.basename(fft_file)

                        DM_trial_was_searched = check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level)
                        # print("周期性搜索 FFT：DM_trial_was_searched =", DM_trial_was_searched)
                        
                        if dict_flag_steps['flag_step_realfft'] == 1:

                                if DM_trial_was_searched == False:
                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 realfft..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        realfft(dat_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, 0, flag_LOG_append)
                                        print("完成！")
                                        sys.stdout.flush()

                                        if flag_remove_datfiles_of_segments == 1 and (segment_label != "full") and os.path.exists(dat_file):
                                                if verbosity_level >= 1:
                                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 删除 %s 以节省磁盘空间（使用 \"%s\" 可重新创建它）..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly, dat_file_nameonly+".makecmd"), end=' ')
                                                        sys.stdout.flush()
                                                os.remove(dat_file)
                                                if verbosity_level >= 1:
                                                        print("完成！")
                                                        sys.stdout.flush()

                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 rednoise..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        rednoise(fft_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, verbosity_level)
                                        print("完成！")
                                        sys.stdout.flush()

                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在将消噪文件 '%s' 应用到 '%s'..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapfile_nameonly, fft_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        zapped_fft_filename, zapped_inf_filename = zapbirds(fft_file, zapfile, work_dir, log_dir, LOG_basename, presto_env_zmax_0, verbosity_level)
                                        zapped_fft_nameonly = os.path.basename(zapped_fft_filename)
                                        print("完成！")
                                        sys.stdout.flush()
                                else:
                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 文件 '%s' 已成功搜索过，跳过..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly)); sys.stdout.flush()
                                        
                        else:
                                print("STEP_REALFFT = 0，跳过 realfft、rednoise、zapbirds...")

                        # print "\033[1m >> 提示：\033[0m 使用 '\033[1mtail -f %s\033[0m' 跟踪 accelsearch 日志" % (log_abspath)

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
                                                        print("完成，耗时 %.2f 秒！" % (time_taken_accelsearch_s))
                                                        sys.stdout.flush()
                                                        ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s" % (int(z)))

                                        if jerksearch_wmax > 0 and flag_jerk_search == 1:
                                                tstart_jerksearch = time.time()
                                                print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 Jerk 搜索，参数为 zmax=%d, wmax=%d, numharm=%d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm), end=' ')
                                                sys.stdout.flush()
                                                flag_cuda = ""
                                                jerksearch_flags = other_flags_accelsearch + flag_cuda
                                                jerksearch(fft_file, work_dir, log_abspath, numharm=jerksearch_numharm, zmax=jerksearch_zmax, wmax=jerksearch_wmax, other_flags=jerksearch_flags, dict_env=dict_env_zmax_0, verbosity_level=verbosity_level, flag_LOG_append=flag_LOG_append)
                                                tend_jerksearch = time.time()
                                                time_taken_jerksearch_s = tend_jerksearch - tstart_jerksearch
                                                print("完成，耗时 %.2f 秒！" % (time_taken_jerksearch_s))
                                                sys.stdout.flush()
                                                ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s_JERK_%s" % (jerksearch_zmax, jerksearch_wmax))

                                else:   
                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 文件 '%s' 已成功搜索过，跳过..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' '); sys.stdout.flush()

                        if flag_remove_fftfiles == 1 and os.path.exists(fft_file):
                                print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 删除 %s 以节省磁盘空间..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, fft_file_nameonly), end=' '); sys.stdout.flush()
                                os.remove(fft_file)  # 删除 FFT 文件
                                print("完成！");  sys.stdout.flush()


        #########################################################################################################
        #                                     并行化的 Jerk 搜索
        # 如果我们使用多个 CPU 进行 Jerk 搜索，方案会有所不同
        # 我们将首先对所有 dat 文件进行 realfft、去红噪声和消噪处理，然后并行搜索所有 .fft 文件
        #########################################################################################################
        
        elif num_simultaneous_jerksearches >= 2 and jerksearch_wmax > 0 and flag_jerk_search == 1:
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

                        DM_trial_was_searched = check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level)

                        if dict_flag_steps['flag_step_realfft'] == 1:

                                if DM_trial_was_searched == False:
                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 realfft..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        realfft(dat_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, 0, flag_LOG_append)
                                        print("完成！")
                                        sys.stdout.flush()

                                        if flag_remove_datfiles_of_segments ==1 and (segment_label != "full") and os.path.exists(dat_file):
                                                if verbosity_level >= 1:
                                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 删除 %s 以节省磁盘空间（使用 \"%s\" 可重新创建它）..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly, dat_file_nameonly+".makecmd"), end=' ')
                                                        sys.stdout.flush()
                                                os.remove(dat_file)
                                                if verbosity_level >= 1:
                                                        print("完成！")
                                                        sys.stdout.flush()

                                        print("段 '%s' %d/%d | 检查点 %d/%d | DM %d/%d - 正在对 %s 进行 rednoise..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        rednoise(fft_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, verbosity_level)
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


def check_zaplist_outfiles(fft_infile, verbosity_level=0):
        birds_filename   = fft_infile.replace(".fft", ".birds")
        zaplist_filename = fft_infile.replace(".fft", ".zaplist")
        try:
                if (os.path.getsize(birds_filename) > 0) and (os.path.getsize(zaplist_filename) >0): #checks if it exists and its
                        return True
                else:
                        return False
        except OSError:
                return False


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


def make_zaplist(fft_infile, out_dir, log_dir, LOG_basename, common_birdies_filename, birds_numharm=4, other_flags_accelsearch="", presto_env=os.environ['PRESTO'], verbosity_level=0):
    fft_infile_nameonly = os.path.basename(fft_infile)  # 获取输入文件的文件名
    fft_infile_basename = os.path.splitext(fft_infile_nameonly)[0]  # 获取输入文件的基本名称（无扩展名）
    log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)  # 日志文件的绝对路径
    dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}  # 设置环境变量

    # 检查是否已存在zaplist文件
    if check_zaplist_outfiles(fft_infile) == False:
        if verbosity_level >= 2:  # 如果详细级别大于等于2，打印运行信息
            print("正在执行accelsearch...", end=' ')
            sys.stdout.flush()
            print(fft_infile, birds_numharm, 0, other_flags_accelsearch, presto_env, verbosity_level)
        accelsearch(fft_infile, out_dir, log_abspath, birds_numharm, 0, other_flags_accelsearch, dict_env, verbosity_level)  # 执行accelsearch
        if verbosity_level >= 2:
            print("accelsearch完成！")
        ACCEL_0_filename = fft_infile.replace(".fft", "_ACCEL_0")  # 生成的ACCEL_0文件名
        fourier_bin_width_Hz = get_Fourier_bin_width(fft_infile)  # 获取傅里叶频宽
        if verbosity_level >= 2:
            print("傅里叶频宽：", fourier_bin_width_Hz)
            print("正在生成鸟频文件...")
            sys.stdout.flush()
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

        if verbosity_level >= 2:
            print("鸟频文件生成完成！")
            sys.stdout.flush()
        zaplist_filename = fft_infile.replace(".fft", ".zaplist")
        cmd_makezaplist = "makezaplist.py %s" % (birds_filename)  # 构造makezaplist命令
        run_cmd(cmd_makezaplist,ifok=zaplist_filename,work_dir=out_dir,log_file=log_abspath,dict_envs=dict_env,flag_append=None)
        #execute_and_log(cmd_makezaplist, out_dir, log_abspath, dict_env, 0)  # 执行makezaplist命令并记录日志
    else:
        if verbosity_level >= 1:
            print("文件 %s 的zaplist已存在！" % (fft_infile_basename), end=' ')

    zaplist_filename = fft_infile.replace(".fft", ".zaplist")  # 生成的zaplist文件名
    return zaplist_filename


def rednoise(fftfile, out_dir, log_dir, LOG_basename, other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0):
        # 获取文件名和基本路径
        verbosity_level = 2
        fftfile_nameonly = os.path.basename(fftfile)
        fftfile_basename = os.path.splitext(fftfile_nameonly)[0]
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)

        dereddened_ffts_filename = "%s/dereddened_ffts.txt" % (out_dir)  # 已去红噪声文件列表
        fftfile_rednoise_abspath = os.path.join(out_dir, "%s_red.fft" % (fftfile_basename))  # 去红噪声后的.fft文件路径
        inffile_rednoise_abspath = os.path.join(out_dir, "%s_red.inf" % (fftfile_basename))  # 去红噪声后的.inf文件路径
        inffile_original_abspath = os.path.join(out_dir, "%s.inf" % (fftfile_basename))  # 原始.inf文件路径

        cmd_rednoise = "rednoise %s %s" % (other_flags, fftfile)  # 构造rednoise命令
        print("rednoise:: rednoise命令 = ", cmd_rednoise)

        try:  # 尝试打开已去红噪声文件列表
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')
        except:  # 如果文件不存在，则创建
                print("rednoise:: 文件 '%s' 不存在。正在创建..." % (dereddened_ffts_filename), end=' ')
                sys.stdout.flush()
                os.mknod(dereddened_ffts_filename)
                print("完成！")
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')

        # 检查当前fft文件是否已在去红噪声列表中
        if "%s\n" % (fftfile) in file_dereddened_ffts.readlines():
                print("rednoise:: 注意：文件 '%s' 已在去红噪声文件列表中 (%s)。" % (fftfile, dereddened_ffts_filename))
                print("rednoise:: 检查文件大小 '%s'" % (fftfile))
                if os.path.getsize(fftfile) > 0:  # 如果文件大小大于0，则跳过
                        operation = "skip"
                        print("rednoise:: 文件大小大于0，跳过...")
                else:  # 如果文件大小为0，则重新处理
                        operation = "make_from_scratch"
                        print("rednoise:: 文件大小为0，重新处理...")
        else:  # 如果文件不在去红噪声列表中，则重新处理
                operation = "make_from_scratch"
                print("rednoise:: 文件 '%s' 不在去红噪声文件列表中 (%s)，将从头开始处理..." % (fftfile_basename, dereddened_ffts_filename))

        file_dereddened_ffts.close()
        if operation == "make_from_scratch":  # 如果需要从头处理
                print("rednoise:: 正在从头处理...")
                dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
                #execute_and_log(cmd_rednoise, out_dir, log_abspath, dict_env, 0)  # 执行rednoise命令并记录日志
                run_cmd(cmd_rednoise,ifok=None,work_dir=out_dir,log_file=log_abspath,dict_envs=dict_env,flag_append=None)
                print("完成！", end=' ')
                sys.stdout.flush()
                file_dereddened_ffts = open(dereddened_ffts_filename, 'a')  # 将文件添加到去红噪声列表
                file_dereddened_ffts.write("%s\n" % (fftfile))
                file_dereddened_ffts.close()
                os.rename(fftfile_rednoise_abspath, fftfile_rednoise_abspath.replace("_red.", "."))  # 重命名文件
                os.rename(inffile_rednoise_abspath, inffile_rednoise_abspath.replace("_red.", "."))


def realfft(infile, out_dir, log_dir, LOG_basename, other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0, flag_LOG_append=0):
    infile_nameonly = os.path.basename(infile)  
    infile_basename = os.path.splitext(infile_nameonly)[0]  # 获取输入文件的基本名称（无扩展名）
    log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)  
    fftfile_abspath = os.path.join(out_dir, "%s.fft" % (infile_basename))  
    cmd_realfft = "realfft %s %s" % (other_flags, infile)  

    if os.path.exists(fftfile_abspath) and (os.path.getsize(fftfile_abspath) > 0):  
        print_log("警告：文件 %s 已存在。跳过 realfft..." % (fftfile_abspath),color=colors.WARNING)
    else:  
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}  # 设置环境变量
        #execute_and_log(cmd_realfft, out_dir, log_abspath, dict_env, 0)  
        run_cmd(cmd_realfft,ifok=fftfile_abspath,work_dir=out_dir,log_file=log_abspath,dict_envs=dict_env,flag_append=None)
        if os.path.exists(fftfile_abspath) and (os.stat(fftfile_abspath).st_size > 0):  # 如果.fft文件已生成且大小大于0
                print_log("%s | realfft 对文件 \"%s\" 处理成功！" % (datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly),color=colors.OKBLUE)
                sys.stdout.flush()
        else:  
                print_log("警告 (%s) | realfft 对文件 \"%s\" 的处理未找到输出文件！" % (datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly),color=colors.WARNING)
                sys.stdout.flush()


# PREPDATA
def prepdata(infile, out_dir, log_dir, LOG_basename, DM, Nsamples=0, ignorechan_list="", mask="", downsample_factor=1, reference="barycentric", other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        # file_log = open(log_abspath, "w"); file_log.close()
        outfile_basename = "%s_DM%05.2f" % (infile_basename, np.float64(DM))
        datfile_abspath = os.path.join(out_dir, "%s.dat" % (outfile_basename))
        inffile_abspath = os.path.join(out_dir, "%s.inf" % (outfile_basename))

        if reference =="topocentric":
                flag_nobary = "-nobary "
        elif reference =="barycentric":
                flag_nobary = ""
        else:
                print_log("ERROR: Invalid value for barycentering option: \"%s\"" % (reference),color=colors.ERROR)
                exit()

        if Nsamples >= 0:
                flag_numout = "-numout %d " % (make_even_number(int(Nsamples/np.float64(downsample_factor))) )
        else:
                flag_numout = ""

        if mask !="":
                flag_mask = "-mask %s " % (mask)
        else:
                flag_mask = ""

        if ignorechan_list !="":
                flag_ignorechan = "-ignorechan %s " % (ignorechan_list)
        else:
                flag_ignorechan = ""

        cmd_prepdata = "prepdata -o %s %s%s %s%s%s -dm %s -downsamp %s %s" % (outfile_basename, flag_numout, flag_ignorechan, flag_mask, flag_nobary, other_flags, str(DM), downsample_factor, infile)

        # print("%s | 正在运行：" % (datetime.now()).strftime("%Y/%m/%d  %H:%M")); sys.stdout.flush()
        # print("%s" % (cmd_prepdata)); sys.stdout.flush()

        if os.path.exists(datfile_abspath) and os.path.exists(inffile_abspath):
            # 如果文件已存在，则跳过并检查结果
            print_log("\n警告：文件 '%s.dat' 和 '%s.inf' 已存在。跳过并检查结果..." % (outfile_basename, outfile_basename),color=colors.WARNING)
        else:
            # 设置环境变量并执行命令
            dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
            run_cmd(cmd_prepdata,ifok=datfile_abspath,work_dir=out_dir,log_file=log_abspath,dict_envs=dict_env,flag_append=None)        
            #execute_and_log(cmd_prepdata, out_dir, log_abspath, dict_env, 0)
            if os.path.exists(datfile_abspath) and os.path.exists(inffile_abspath):
                # 如果生成的文件存在，打印成功信息
                print("%s | prepdata 对文件 \"%s\" 处理成功！" % (datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly)); sys.stdout.flush()
            else:
                # 如果文件缺失，打印警告信息
                print("警告 (%s) | prepdata 对文件 \"%s\" 的处理未找到所有输出文件！" % (datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly)); sys.stdout.flush()

def make_rfifind_mask(infile, out_dir, log_dir, LOG_basename, time=0.1, time_intervals_to_zap="", chans_to_zap="", other_flags="", presto_env=os.environ['PRESTO']):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]

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


def get_DDplan_scheme(infile, out_dir, log_dir, LOG_basename, loDM, highDM, DM_coherent_dedispersion, N_DMs_per_prepsubband, freq_central_MHz, bw_MHz, nchan, nsubbands, t_samp_s):
    # 获取输入文件的名称和基础名
    infile_nameonly = os.path.basename(infile)
    infile_basename = os.path.splitext(infile_nameonly)[0]
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


def dedisperse(infile, out_dir, log_dir, LOG_basename, segment_label, chunk_label, Nsamples, ignorechan_list, mask_file, list_DD_schemes, nchan, nsubbands=0, num_simultaneous_prepsubbands=1, other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0):
        # 获取输入文件的名称和基础名
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        # 构造 prepsubband 输出文件名
        prepsubband_outfilename = "%s_%s_%s" % (infile_basename, segment_label, chunk_label)
        # 设置环境变量，确保 PRESTO 工具链的路径正确
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
        # 定义 prepsubband 脚本文件的名称和绝对路径
        file_script_prepsubband_name = "script_prepsubband.txt"
        file_script_prepsubband_abspath = "%s/%s" % (out_dir, file_script_prepsubband_name)
        
        # 定义日志文件的绝对路径
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)

        # 获取去色散方案的数量
        N_schemes = len(list_DD_schemes)

        # 构造掩膜文件和忽略通道的参数字符串
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
        file_script_prepsubband = open(file_script_prepsubband_abspath, "w")
        for i in range(N_schemes):
                flag_numout = ""
                if i < N_schemes-1:
                        # 构造 prepsubband 命令（非最后一个方案）
                        cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'], list_DD_schemes[i]['downsamp'], nsubbands, infile)
                elif i == N_schemes-1:
                        # 构造 prepsubband 命令（最后一个方案，DM 数量加 1）
                        cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'] + 1, list_DD_schemes[i]['downsamp'], nsubbands, infile)
                # 将命令写入脚本文件
                file_script_prepsubband.write("%s\n" % cmd_prepsubband)

        # 关闭脚本文件
        file_script_prepsubband.close()

        # 打印子带数量和提示信息
        print(f"使用 {nsubbands} 个子带进行去色散（原始通道数量：{nchan}）")
        print()
        if N_schemes == 1:
            print_log(f"提示：使用 'tail -f {log_abspath}' 查看 prepsubband 的进度",masks=f'tail -f {log_abspath}',color=colors.OKCYAN)
        elif N_schemes > 1:
            print_log(f"提示：使用 'for f in {log_dir}/LOG_prepsubband_*.txt; do tail -1 ${{f}}; echo; done' 查看 prepsubband 的进度",masks=f'for f in {log_dir}/LOG_prepsubband_*.txt; do tail -1 ${{f}}; echo; done',color=colors.OKCYAN)
        print()

        # 初始化已完成的 prepsubband 方案数量和命令列表
        N_prepsubband_schemes_done = 0
        list_prepsubband_commands = []
        list_prepsubband_workdirs = []


        while (N_prepsubband_schemes_done < N_schemes):
                # 打印当前已完成的去色散方案数量
                print("dedisperse:: N_prepsubband_schemes_done =", N_prepsubband_schemes_done)
                for i in range(N_schemes):
                        flag_numout = ""
                        # 获取当前方案的低 DM、DM 步长和高 DM
                        loDM = np.float64(list_DD_schemes[i]['loDM'])
                        dDM  = np.float64(list_DD_schemes[i]['dDM'])
                        hiDM = loDM + int(list_DD_schemes[i]['num_DMs'])*dDM

                        # 根据是否是最后一个方案，设置括号类型
                        if i < N_schemes-1:
                                str_parentesis = ")"
                                # 构造 prepsubband 命令（非最后一个方案）
                                cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'], list_DD_schemes[i]['downsamp'], nsubbands, infile)
                        elif i == N_schemes-1:
                                str_parentesis = "]"
                                # 构造 prepsubband 命令（最后一个方案，DM 数量加 1）
                                cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'] + 1, list_DD_schemes[i]['downsamp'], nsubbands, infile)

                        # 检查当前方案是否已完成
                        if check_prepsubband_result_single_scheme(out_dir, list_DD_schemes[i], verbosity_level) == False:
                                # 如果只运行一个 prepsubband 或只有一个方案
                                if num_simultaneous_prepsubbands == 1 or N_schemes == 1:
                                        # print("正在运行 prepsubband，DM 范围 [%.3f-%.3f%s] pc cm^-3（方案 %d/%d），观测文件 '%s'..." % (loDM, hiDM, str_parentesis, i+1, N_schemes, infile), end=' '); sys.stdout.flush()
                                        # print("dedisperse:: %d) 正在运行： %s" % (i, cmd_prepsubband))

                                        # 执行 prepsubband 命令并记录日志
                                        #execute_and_log("which prepsubband", out_dir, log_abspath, dict_env, 1)
                                        run_cmd(cmd_prepsubband,ifok=None,work_dir=out_dir,log_file=log_abspath,dict_envs=dict_env,flag_append=None)
                                        #execute_and_log(cmd_prepsubband, out_dir, log_abspath, dict_env, 1)
                                        # 检查当前方案是否成功完成
                                        if check_prepsubband_result_single_scheme(out_dir, list_DD_schemes[i], verbosity_level) == True:
                                                N_prepsubband_schemes_done = N_prepsubband_schemes_done + 1
                                        print("完成！"); sys.stdout.flush()

                                # 如果允许多个 prepsubband 同时运行且有多个方案
                                elif num_simultaneous_prepsubbands > 1 and N_schemes > 1:
                                        # 将命令添加到命令列表
                                        list_prepsubband_commands.append(cmd_prepsubband)
                                        list_prepsubband_workdirs.append(out_dir)
                                        # print("list_prepsubband_commands =", list_prepsubband_commands)
                                        N_prepsubband_schemes_done = N_prepsubband_schemes_done + 1

                        else:
                                # 如果当前方案已成功运行，跳过
                                print_log("警告：prepsubband，DM 范围 [%.3f-%.3f%s] pc cm^-3（方案 %d/%d），观测文件 '%s' 已成功运行。跳过..." % (list_DD_schemes[i]['loDM'], hiDM, str_parentesis, i+1, N_schemes, infile),color=colors.WARNING)
                                N_prepsubband_schemes_done = N_prepsubband_schemes_done + 1

        # 如果允许多个 prepsubband 同时运行且有多个方案
        if num_simultaneous_prepsubbands > 1 and N_schemes > 1:
                # 创建线程池并运行命令
                TP = ThreadPool(num_simultaneous_prepsubbands)
                N_commands = len(list_prepsubband_commands)
                print()
                print("总共 %d 条 prepsubband 命令，同时运行 %d 条..." % (N_commands, num_simultaneous_prepsubbands));  sys.stdout.flush()
                print()
                for k in range(len(list_prepsubband_commands)):
                        print("排队 prepsubband 命令 #%d: '%s'" % (k+1, list_prepsubband_commands[k]))
                        time.sleep(0.1)
                        TP.apply_async(execute_and_log_in_thread_pool, (list_prepsubband_commands[k], log_dir, list_prepsubband_workdirs[k], k, N_commands, 1) )
                TP.close()
                TP.join()
                print()
                print("%d 条命令已完成！" % (len(list_prepsubband_commands)))

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


def zapbirds(fft_infile, zapfile_name, work_dir, log_dir, LOG_basename, presto_env, verbosity_level=0):
        fft_infile_nameonly = os.path.basename(fft_infile)
        fft_infile_basename = os.path.splitext(fft_infile_nameonly)[0]
        inffile_filename = fft_infile.replace(".fft", ".inf")
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        # file_log = open(log_abspath, "w"); file_log.close()
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}

        cmd_zapbirds = "zapbirds -zap -zapfile %s %s" % (zapfile_name, fft_infile)
        zapped_fft_filename = fft_infile.replace(".fft", "_zapped.fft")
        zapped_inf_filename = inffile_filename.replace(".inf", "_zapped.inf")

        list_zapped_ffts_abspath = os.path.join(work_dir, "list_zapped_ffts.txt")
        # if verbosity_level >= 2:
        #         print("zapbirds:: list_zapped_ffts_abspath = ", list_zapped_ffts_abspath)

        if check_zapbirds_outfiles(fft_infile, list_zapped_ffts_abspath, verbosity_level=0) == False:
                print("Running ZAPBIRDS: %s" % (cmd_zapbirds))
                sys.stdout.flush()
                execute_and_log(cmd_zapbirds, work_dir, log_abspath, dict_env, 0)
                file_list_zapped_ffts = open(list_zapped_ffts_abspath, 'a')
                file_list_zapped_ffts.write("%s\n" % (fft_infile))
                file_list_zapped_ffts.close()

        return zapped_fft_filename, zapped_inf_filename


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






