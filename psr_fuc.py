
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
import datetime
import numpy as np
import urllib
from presto import filterbank, infodata, parfile, psr_utils, psrfits, rfifind, sifting
from multiprocessing.pool import ThreadPool

def makedir(dir): 
    os.makedirs(dir, exist_ok=True)

def write2file(info, file_path):
    """
    将info写入file_path
    """
    with open(file_path, 'a') as file:
        file.write(info)

def print_log(*args, sep=' ', end='\n', file=None, flush=False, log_files=None, masks=None):
    """
    个人喜欢的能储存日志以及高亮打印的函数。
    默认日志存储在当前工作目录下的 logall.txt。
    mask 的内容会以红色高亮显示。
    """

    if log_files is None:
        log_files = ['logall.txt']
    elif log_files == 'time':
        log_files = ['logall.txt', 'logruntime.txt']

    message = sep.join(str(arg) for arg in args) + end
    log_contents = [message]

    for file_path in log_files:
        write2file(''.join(log_contents), file_path)

    if masks is None:
        print(message, end='', file=sys.stdout, flush=flush)
    else:
        masks = str(masks)  # 确保 masks 为字符串
        if masks in message:
            # 将包含 masks 的部分打印为红色
            highlighted_message = message.replace(
                masks, f"\033[91m{masks}\033[0m"
            )  # 91m 为红色
            print(highlighted_message, end='', file=sys.stdout, flush=flush)
        else:
            print(message, end='', file=sys.stdout, flush=flush)

def check_presto_path(presto_path, key):
        if os.path.exists(presto_path):
                if os.path.exists(presto_path+"/bin/accelsearch"):
                        return True
                else:
                        print("%sERROR%s: %s directory '%s' exists but I could not find 'accelsearch' in %s/bin!" % (colors.ERROR+colors.BOLD, colors.ENDCOLOR, key, presto_path, presto_path))
                        print("Please make sure that your %s installation is actually there and working." % (key))
                        exit()
                
        else:
                print("%sERROR:%s %s directory '%s' does not exist!" % (colors.ERROR+colors.BOLD, colors.ENDCOLOR, key, presto_path))
                print("Please make sure that the path of %s in your configuration file is correct." % (key))
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

    print()
    print("******************************************")
    print("       预估磁盘空间使用情况：")
    print("******************************************")
    if flag_remove_fftfiles == 0:
        print("是否删除 .fft 文件？否  --> 每个 DM 试验将占用双倍空间")
    else:
        print("是否删除 .fft 文件？是")

    if flag_remove_datfiles_of_segments == 0:
        print("是否删除分段的 .dat 文件？否  --> 总占用空间将非常高")
    else:
        print("是否删除分段的 .dat 文件？是  --> 分段搜索的磁盘空间占用将可以忽略不计")

    print()
    print("全长度搜索：~%4d GB        (%d DM 试验 * 每次试验 %5d MB)" % (full_length_search_size_bytes / 1.0e9, num_DMs, datfile_full_size_bytes / 1.0e6))

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
    print("预期磁盘空间使用量：~%5d GB" % (1.1 * total_search_size_bytes / 1.0e9))
    print("可用磁盘空间：~%5d GB" % (disk_space_free_bytes / 1.0e9), end="")
    if disk_space_free_bytes > 1.1 * total_search_size_bytes:
        print("   --> 太好了！磁盘空间足够。")
        return True
    else:
        print("   --> 哎呀！磁盘空间不足！")
        return False
    
def get_command_output(command, shell_state=False, work_dir=os.getcwd()):
        list_for_Popen = command.split()
        if shell_state ==False:
                proc = subprocess.Popen(list_for_Popen, stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        else:
                proc = subprocess.Popen([command], stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        out, err = proc.communicate()

        return out.decode('ascii')

def execute_and_log_in_thread_pool(command, log_dir, work_dir, id_num, N_ids, flag_log=1):
    # 获取当前时间，用于日志记录
    datetime_start = (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M")
    datetime_start_single_string = (datetime.datetime.now()).strftime("%Y%m%d_%H%M")
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
    datetime_end = (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M")
    time_end = time.time()

    # 如果记录日志，写入结束时间和总耗时
    if flag_log == 1:
        log_file.write("\n结束日期和时间：%s\n" % (datetime_end))
        log_file.write("\n总耗时：%d 秒\n" % (time_end - time_start))
        log_file.close()

    # 打印命令执行完成的提示信息
    print("命令 %4d/%d ('%s') 执行完成。" % (id_num + 1, N_ids, command_label)); sys.stdout.flush()

def execute_and_log(command, work_dir, log_abspath, dict_envs={}, flag_append=0, verbosity_level=0):
        datetime_start = (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M")
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

        datetime_end = (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M")
        time_end = time.time()

        print("execute_and_log:: 命令：%s" % (command))  # 打印执行的命令
        print("execute_and_log:: 找到的可执行文件路径：%s: " % (executable), get_command_output("which %s" % (executable)))  # 打印可执行文件路径
        print("execute_and_log:: 工作目录 = ", work_dir)  # 打印工作目录
        print("execute_and_log:: 查看日志的方式：\"tail -f %s\"" % (log_abspath))  # 提示如何查看日志
        sys.stdout.flush()  # 刷新标准输出缓冲区
        print("execute_and_log: 用于 Popen 的命令列表 = ", list_for_Popen)  # 打印用于子进程的命令列表
        print("execute_and_log: 日志文件 = ", log_file)  # 打印日志文件对象
        print("execute_and_log: 子进程环境变量 = ", env_subprocess)  # 打印子进程的环境变量

        log_file.write("\n结束日期和时间：%s\n" % (datetime_end))  # 写入结束时间和日期
        log_file.write("\n总耗时：%d 秒\n" % (time_end - time_start))  # 写入总耗时
        log_file.close()  # 关闭日志文件

def sift_candidates(work_dir, log_dir, LOG_basename,  dedispersion_dir, observation_basename, segment_label, chunk_label, list_zmax, jerksearch_zmax, jerksearch_wmax, flag_remove_duplicates, flag_DM_problems, flag_remove_harmonics, minimum_numDMs_where_detected, minimum_acceptable_DM=2.0, period_to_search_min_s=0.001, period_to_search_max_s=15.0, verbosity_level=0):
        work_dir_basename = os.path.basename(work_dir)
        string_ACCEL_files_dir = os.path.join(dedispersion_dir, observation_basename, segment_label, chunk_label)

        best_cands_filename = "%s/best_candidates_%s.siftedcands" % (work_dir, work_dir_basename)
        if verbosity_level >= 3:
                print("sift_candidates:: best_cands_filename = %s" % (best_cands_filename))
                print("sift_candidates:: string_ACCEL_files_dir = %s" % (string_ACCEL_files_dir))

        list_ACCEL_files = []
        for z in list_zmax:
                string_glob = "%s/*ACCEL_%d" % (string_ACCEL_files_dir, z)
                if verbosity_level >= 1:
                        print("Reading files '%s'..." % (string_glob), end=' ')
                list_ACCEL_files = list_ACCEL_files + glob.glob(string_glob)
                if verbosity_level >= 1:
                        print("done!")

        string_glob_jerk_files = "%s/*ACCEL_%d_JERK_%d" % (string_ACCEL_files_dir, jerksearch_zmax, jerksearch_wmax)
        if verbosity_level >= 3:
                print("JERK: Also reading files '%s'.." % (string_glob_jerk_files))
                print("Found: ", glob.glob(string_glob_jerk_files))

        list_ACCEL_files = list_ACCEL_files + glob.glob(string_glob_jerk_files)

        if verbosity_level >= 3:
                print()
                print("ACCEL files found: ", list_ACCEL_files)
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        if verbosity_level >= 1:
                print("\033[1m >> TIP:\033[0m Check sifting output with '\033[1mcat %s\033[0m'" % (log_abspath))

        list_DMs = [x.split("_ACCEL")[0].split("DM")[-1] for x in list_ACCEL_files]
        candidates = sifting.read_candidates(list_ACCEL_files, track=True)

        print("sift_candidates:: z = %d" % (z))
        print("sift_candidates:: %s/*ACCEL_%d" % (string_ACCEL_files_dir, z))
        print("sift_candidates:: list_ACCEL_files = %s" % (list_ACCEL_file))
        print("sift_candidates:: list_DMs = %s" % (list_DMs))
        print("sift_candidates:: candidates.cands = ", candidates.cands)
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

def fold_candidate(work_dir, log_dir, LOG_basename, observation, dir_dedispersion, obs, seg, ck, candidate, ignorechan_list, other_flags_prepfold="", presto_env=os.environ['PRESTO'], verbosity_level=0, flag_LOG_append=1, what_fold="rawdata", num_simultaneous_prepfolds=1):
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
                if verbosity_level >= 2:
                        print(cmd_prepfold)

        if verbosity_level >= 2:
                print("fold_candidates:: cand.filename: ",  cand.filename)
                print("file_to_fold = ", file_to_fold)
                print("fold_candidates:: cmd_prepfold = %s" % (cmd_prepfold))

        file_script_fold.close()


def make_even_number(number_int):  # 定义一个函数，将输入的数字转换为偶数
        if int(number_int) % 2 == 1:  # 如果数字是奇数
                return int(number_int) - 1  # 返回该数字减1后的偶数
        elif int(number_int) % 2 == 0:  # 如果数字已经是偶数
                return int(number_int)  # 直接返回该数字
        else:  # 如果输入的数字既不是奇数也不是偶数（理论上不可能发生）
                print("ERROR: make_even_number:: 输入的数字既不是偶数也不是奇数！")
                exit()  


def get_command_output(command, shell_state=False, work_dir=os.getcwd()):
        list_for_Popen = command.split()
        if shell_state ==False:
                proc = subprocess.Popen(list_for_Popen, stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        else:
                proc = subprocess.Popen([command], stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        out, err = proc.communicate()

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
    print(cmd)  # 打印命令以便调试

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()  # 获取命令的输出和错误信息
    output_str = output.decode('utf-8').strip()
    return output_str


def get_rfifind_result(file_mask, LOG_file, verbosity_level=0):
        rfifind_mask = rfifind.rfifind(file_mask)  # 加载 rfifind 对象

        N_int = rfifind_mask.nint  # 获取时间积分的数量
        N_chan = rfifind_mask.nchan  # 获取频率通道的数量
        N_int_masked = len(rfifind_mask.mask_zap_ints)  # 获取被屏蔽的时间积分数量
        N_chan_masked = len(rfifind_mask.mask_zap_chans)  # 获取被屏蔽的频率通道数量
        fraction_int_masked = np.float64(N_int_masked / N_int)  # 计算被屏蔽的时间积分比例
        fraction_chan_masked = np.float64(N_chan_masked / N_chan)  # 计算被屏蔽的频率通道比例

        print("get_rfifind_result:: 文件掩膜：%s" % file_mask)  # 打印文件掩膜
        print("get_rfifind_result:: 日志文件：%s" % LOG_file)  # 打印日志文件

        if (fraction_int_masked > 0.5) or (fraction_chan_masked > 0.5):  # 如果屏蔽比例超过 50%
                return "!Mask>50%"  # 返回错误信息

        # 检查日志文件中是否存在第一个块的裁剪问题，并获取有问题的文件名。否则返回 True。
        cmd_grep_problem_clipping = "grep -l 'problem with clipping' %s" % (LOG_file)  # -l 选项返回包含该表达式的文件名
        cmd_grep_inf_results = "grep -l ' inf ' %s" % (LOG_file)
        output = get_command_output(cmd_grep_problem_clipping, True).strip()  # 执行命令并获取输出
        if output != "":
                if verbosity_level >= 1:  # 如果详细级别 >= 1
                        print()
                        print("警告：文件 '%s' 中第一个块存在裁剪问题！" % (LOG_file))  # 提示裁剪问题
                return "!ProbFirstBlock"  # 返回错误信息

        output = get_command_output(cmd_grep_inf_results, True).strip()  # 检查是否存在无穷大结果
        if output != "":
                if verbosity_level >= 1:  # 如果详细级别 >= 1
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


def check_rfifind_outfiles(out_dir, basename, verbosity_level=0):
        for suffix in ["bytemask", "inf", "mask", "ps", "rfi", "stats"]:
                file_to_check = "%s/%s_rfifind.%s" % (out_dir, basename, suffix)
                if not os.path.exists(file_to_check):
                        if verbosity_level >= 1:
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


def accelsearch(infile, work_dir, log_abspath, numharm=8, zmax=0, other_flags="", dict_env= {}, verbosity_level=0, flag_LOG_append=1):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        inffile_empty = infile.replace(".fft", "_ACCEL_%d_empty" % (zmax))

        cmd_accelsearch = "accelsearch %s -zmax %s -numharm %s %s" % (other_flags, zmax, numharm, infile)

        if verbosity_level >= 2:
                print()
                print("BEGIN ACCELSEARCH ----------------------------------------------------------------------")

                print("accelsearch:: cmd_accelsearch: ", cmd_accelsearch)
                print("accelsearch:: ENV: ", dict_env)
                print("accelsearch:: check_accelsearch_result(infile, int(zmax)) :: %s" % (check_accelsearch_result(infile, int(zmax))))
                print("accelsearch:: work_dir = %s" % (work_dir))
                print("accelsearch:: infile = %s" % (infile))

        if check_accelsearch_result(infile, int(zmax)) == False and check_accelsearch_result(inffile_empty, int(zmax)) == False:
                if verbosity_level >= 2:
                        print("accelsearch:: running: %s" % (cmd_accelsearch))
                execute_and_log(cmd_accelsearch, work_dir, log_abspath, dict_env, flag_LOG_append)
        else:
                if verbosity_level >= 2:
                        print("accelsearch:: WARNING: accelsearch with zmax=%d seems to have been already executed on file %s. Skipping..." % (int(zmax), infile_nameonly))

        if verbosity_level >= 2:
                print("accelsearch:: NOW I CHECK THE RESULT OF THE EXECUTION!")

        if check_accelsearch_result(infile, int(zmax)) == False:
                if verbosity_level >= 2:
                        print("False! Then I create a _empty file!")
                file_empty = open(inffile_empty, "w")
                if verbosity_level >=1:
                        print("%sWARNING%s: accelsearch did not produce any candidates! Writing file %s to signal this..." % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, inffile_empty), end='')
                file_empty.write("ACCELSEARCH DID NOT PRODUCE ANY CANDIDATES!")
        else:
                if verbosity_level >= 2:
                        print("accelsearch: GOOD! CANDIDATES HAVE BEEN PRODUCED for %s!" % (infile))

        if verbosity_level >= 2:
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
        dat_file_nameonly = os.path.basename(dat_file)
        fft_file = dat_file.replace(".dat", ".fft")
        fft_file_nameonly = os.path.basename(fft_file)

        if verbosity_level >= 2:
                print("check_if_DM_trial_was_searched:: list_zmax = %s" % list_zmax)
                print("check_if_DM_trial_was_searched:: flag_jerk_search = %s" % flag_jerk_search)
                print("check_if_DM_trial_was_searched:: jerksearch_zmax = %s, jerksearch_wmax = %s" % (jerksearch_zmax,jerksearch_wmax))
                
        for z in list_zmax:
                ACCEL_filename          = dat_file.replace(".dat", "_ACCEL_%s" % (int(z)))
                ACCEL_filename_empty = dat_file.replace(".dat", "_ACCEL_%s_empty" % (int(z)))
                ACCEL_cand_filename = ACCEL_filename + ".cand"
                ACCEL_txtcand_filename = ACCEL_filename + ".txtcand"

                if verbosity_level >= 2:
                        print("check_if_DM_trial_was_searched:: checking: %s, %s, %s" % (ACCEL_filename, ACCEL_cand_filename, ACCEL_txtcand_filename))
                        print("check_if_DM_trial_was_searched:: checking: %s, %s, %s" % (ACCEL_filename_empty, ACCEL_cand_filename, ACCEL_txtcand_filename))

                
                if (not os.path.exists(ACCEL_filename)       or os.path.getsize(ACCEL_filename) ==0        ) and \
                   (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) ==0  ):
                        if verbosity_level >= 2:
                                print("check_if_DM_trial_was_searched:: False - case 1")
                        return False
                if (not os.path.exists(ACCEL_cand_filename) or os.path.getsize(ACCEL_cand_filename) ==0) and \
                   (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) ==0  ):
                        if verbosity_level >= 2:
                                print("check_if_DM_trial_was_searched:: False - case 2")
                        return False
                if not os.path.exists(ACCEL_txtcand_filename):
                        if verbosity_level >= 2:
                                print("check_if_DM_trial_was_searched:: False - case 3")
                        return False

        if flag_jerk_search == 1 and jerksearch_wmax > 0:
                ACCEL_filename          = dat_file.replace(".dat", "_ACCEL_%s_JERK_%s" % (jerksearch_zmax, jerksearch_wmax))
                ACCEL_filename_empty = dat_file.replace(".dat", "_ACCEL_%s_JERK_%s_empty" % (jerksearch_zmax, jerksearch_wmax))
                ACCEL_cand_filename = ACCEL_filename + ".cand"
                ACCEL_txtcand_filename = ACCEL_filename + ".txtcand"
                # print "check_if_DM_trial_was_searched:: checking: %s, %s, %s" % (ACCEL_filename, ACCEL_cand_filename, ACCEL_txtcand_filename)
                if (not os.path.exists(ACCEL_filename)       or os.path.getsize(ACCEL_filename) ==0        ) and \
                   (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) ==0  ):
                        if verbosity_level >= 2:
                                print("check_if_DM_trial_was_searched:: False - case 4")
                        return False
                if (not os.path.exists(ACCEL_cand_filename) or os.path.getsize(ACCEL_cand_filename) ==0) and \
                   (not os.path.exists(ACCEL_filename_empty) or os.path.getsize(ACCEL_filename_empty) ==0  ):
                        if verbosity_level >= 2:
                                print("check_if_DM_trial_was_searched:: False - case 5")
                        return False
                if not os.path.exists(ACCEL_txtcand_filename):
                        if verbosity_level >= 2:
                                print("check_if_DM_trial_was_searched:: False - case 6")
                        return False

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


        

def periodicity_search_FFT(work_dir, log_dir, LOG_basename, zapfile, segment_label, chunk_label, list_seg_ck_indices, list_DD_scheme, flag_use_cuda=0, list_cuda_ids=[0], flag_acceleration_search=1, numharm=8, list_zmax=[20], flag_jerk_search=1, jerksearch_zmax=0, jerksearch_wmax=0, jerksearch_numharm=4, num_simultaneous_jerksearches=1, period_to_search_min_s=0.001, period_to_search_max_s=20.0, other_flags_accelsearch="", flag_remove_fftfiles=0, flag_remove_datfiles_of_segments=0, presto_env_zmax_0=os.environ['PRESTO'], presto_env_zmax_any=os.environ['PRESTO'], verbosity_level=0, flag_LOG_append=1, dict_flag_steps= {'flag_step_dedisperse': 1 , 'flag_step_realfft': 1, 'flag_step_periodicity_search': 1}):

        i_seg, N_seg, i_ck, N_ck = list_seg_ck_indices

        if verbosity_level >= 2:
                print("periodicity_search_FFT:: Files to search: ", "%s/*DM*.*.dat" % (work_dir))
                print("periodicity_search_FFT:: presto_env_zmax_0 = ", presto_env_zmax_0)
                print("periodicity_search_FFT:: presto_env_zmax_any = ", presto_env_zmax_any)

        list_files_to_search = sorted([x for x in glob.glob("%s/*DM*.*.dat" % (work_dir)) ])

        N_DMs_to_search = 0
        for k in range(len(list_DD_scheme)):
                N_DMs_to_search = N_DMs_to_search + list_DD_scheme[k]['num_DMs']
        
        N_files_to_search = len(list_files_to_search)
        N_files_searched  = N_DMs_to_search - N_files_to_search
        
        frequency_to_search_max = 1./period_to_search_min_s
        frequency_to_search_min = 1./period_to_search_max_s
        if verbosity_level >= 2:
                print("frequency_to_search_min, ", frequency_to_search_min)
                print("frequency_to_search_max, ", frequency_to_search_max)

                print("periodicity_search_FFT:: WARNING: -flo and -fhi CURRENTLY DISABLED")
        dict_env_zmax_0 = {'PRESTO': presto_env_zmax_0,   'PATH': "%s/bin:%s" % (presto_env_zmax_0, os.environ['PATH']),   'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env_zmax_0,   os.environ['LD_LIBRARY_PATH'])}
        dict_env_zmax_any = {'PRESTO': presto_env_zmax_any, 'PATH': "%s/bin:%s" % (presto_env_zmax_any, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env_zmax_any, os.environ['LD_LIBRARY_PATH'])}

        if verbosity_level >= 2:
                print("periodicity_search_FFT:: dict_env_zmax_0 = ", dict_env_zmax_0)
                print("periodicity_search_FFT:: dict_env_zmax_any = ", dict_env_zmax_any)
                print("periodicity_search_FFT:: LOG_basename = ", LOG_basename)
                print("periodicity_search_FFT:: list_files_to_search = ", list_files_to_search)

        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        if verbosity_level >= 1:
                print()
                print("\033[1m >> TIP:\033[0m Follow periodicity search with: \033[1mtail -f %s\033[0m" % (log_abspath))

        zapfile_nameonly = os.path.basename(zapfile)

        #########################################################################################################
        #                                     NON-PARALLELIZED JERK SEARCH
        #########################################################################################################
        if num_simultaneous_jerksearches == 1 or jerksearch_wmax == 0 or flag_jerk_search == 0:
                for i in range(N_files_to_search):
                        print()
                        if verbosity_level >= 2:
                                print("periodicity_search_FFT: inside loop with i = %d / %d" % (i, N_files_to_search-1))
                        dat_file = list_files_to_search[i]
                        dat_file_nameonly = os.path.basename(dat_file)
                        fft_file = dat_file.replace(".dat", ".fft")
                        fft_file_nameonly = os.path.basename(fft_file)

                        DM_trial_was_searched = check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level)
                        #print("periodicity_search_FFT:: DM_trial_was_searched = ", DM_trial_was_searched)
                        
                        if dict_flag_steps['flag_step_realfft'] == 1:

                                if DM_trial_was_searched == False:
                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Doing realfft  of %s..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        realfft(dat_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, 0, flag_LOG_append)
                                        print("done!")
                                        sys.stdout.flush()

                                        if flag_remove_datfiles_of_segments ==1 and (segment_label != "full") and os.path.exists(dat_file):
                                                if verbosity_level >= 1:
                                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Removing %s to save disk space (use \"%s\" to recreate it)..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly, dat_file_nameonly+".makecmd"), end=' ')
                                                        sys.stdout.flush()
                                                os.remove(dat_file)
                                                if verbosity_level >= 1:
                                                        print("done!")
                                                        sys.stdout.flush()

                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Doing rednoise of %s..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        rednoise(fft_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, verbosity_level)
                                        print("done!")
                                        sys.stdout.flush()

                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Applying zapfile '%s' to '%s'..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapfile_nameonly, fft_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        zapped_fft_filename, zapped_inf_filename = zapbirds(fft_file, zapfile, work_dir, log_dir, LOG_basename, presto_env_zmax_0, verbosity_level)
                                        zapped_fft_nameonly = os.path.basename(zapped_fft_filename)
                                        print("done!")
                                        sys.stdout.flush()
                                else:
                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - File '%s' was already successfully searched. Skipping..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly)); sys.stdout.flush()
                                        
                        else:
                                print("STEP_REALFFT = 0, skipping realfft, rednoise, zapbirds...")

                        # print "\033[1m >> TIP:\033[0m Follow accelsearch with '\033[1mtail -f %s\033[0m'" % (log_abspath)

                        if dict_flag_steps['flag_step_periodicity_search'] == 1:
                                if DM_trial_was_searched == False:
                                        if flag_acceleration_search == 1:
                                                for z in list_zmax:
                                                        tstart_accelsearch = time.time()
                                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Doing accelsearch of %s with zmax = %4d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, z), end=' ')
                                                        sys.stdout.flush()
                                                        if int(z) == 0:
                                                                dict_env = copy.deepcopy(dict_env_zmax_0)
                                                                if verbosity_level >= 2:
                                                                        print("accelsearch:: zmax == 0 ----> dict_env = %s" % (dict_env))
                                                                flag_cuda = ""
                                                        else:
                                                                if flag_use_cuda == 1:
                                                                        dict_env = copy.deepcopy(dict_env_zmax_any)
                                                                        gpu_id = random.choice(list_cuda_ids)
                                                                        flag_cuda = " -cuda %d " % (gpu_id)
                                                                else:
                                                                        dict_env = copy.deepcopy(dict_env_zmax_0)
                                                                        flag_cuda = ""

                                                                if verbosity_level >= 2:
                                                                        print("periodicity_search_FFT:: zmax == %d ----> dict_env = %s" % (int(z), dict_env))
                                                                        print("periodicity_search_FFT:: Now check CUDA: list_cuda_ids = ", list_cuda_ids)
                                                                        print("periodicity_search_FFT:: flag_use_cuda = ", flag_use_cuda)
                                                                        print("periodicity_search_FFT:: flag_cuda = ", flag_cuda)

                                                        accelsearch_flags = other_flags_accelsearch + flag_cuda  # + " -flo %s -fhi %s" % (frequency_to_search_min, frequency_to_search_max)

                                                        accelsearch(fft_file, work_dir, log_abspath, numharm=numharm, zmax=z, other_flags=accelsearch_flags, dict_env=dict_env, verbosity_level=verbosity_level, flag_LOG_append=flag_LOG_append)
                                                        tend_accelsearch = time.time()
                                                        time_taken_accelsearch_s = tend_accelsearch - tstart_accelsearch
                                                        print("done in %.2f s!" % (time_taken_accelsearch_s))
                                                        sys.stdout.flush()
                                                        ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s" % (int(z)))

                                        if jerksearch_wmax > 0 and flag_jerk_search == 1:
                                                tstart_jerksearch = time.time()
                                                print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Doing jerk search of %s with zmax=%d, wmax=%d, numharm=%d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm), end=' ')
                                                sys.stdout.flush()
                                                flag_cuda = ""
                                                jerksearch_flags = other_flags_accelsearch + flag_cuda
                                                jerksearch(fft_file, work_dir, log_abspath, numharm=jerksearch_numharm, zmax=jerksearch_zmax, wmax=jerksearch_wmax, other_flags=jerksearch_flags, dict_env=dict_env_zmax_0, verbosity_level=verbosity_level, flag_LOG_append=flag_LOG_append)
                                                tend_jerksearch = time.time()
                                                time_taken_jerksearch_s = tend_jerksearch - tstart_jerksearch
                                                print("done in %.2f s!" % (time_taken_jerksearch_s))
                                                sys.stdout.flush()
                                                ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s_JERK_%s" % (jerksearch_zmax, jerksearch_wmax))

                                else:   
                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - File '%s' was already successfully searched. Skipping..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' '); sys.stdout.flush()


                        if flag_remove_fftfiles ==1  and os.path.exists(fft_file):
                                if verbosity_level >= 1:
                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Removing %s to save disk space..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, fft_file_nameonly), end=' '); sys.stdout.flush()
                                os.remove(fft_file)
                                if verbosity_level >= 1:
                                        print("done!");  sys.stdout.flush()


        #########################################################################################################
        #                                     PARALLELIZED JERK SEARCH
        # If we are doing a jerk search with multiple CPUs, the scheme will be different
        # We will first realfft, deredden and zap all the dat files, then search all the .fft files in parallel
        #########################################################################################################
        
        elif num_simultaneous_jerksearches >= 2 and jerksearch_wmax > 0 and flag_jerk_search == 1:
                list_jerksearch_commands = []
                list_jerksearch_workdirs = []
                jerksearch_flags = other_flags_accelsearch
                print("\nJerk search with multiple CPUs active")

                for i in range(N_files_to_search):
                        print()
                        if verbosity_level >= 2:
                                print("periodicity_search_FFT: inside loop with i = %d / %d" % (i, N_files_to_search-1))
                        dat_file = list_files_to_search[i]
                        dat_file_nameonly = os.path.basename(dat_file)
                        fft_file = dat_file.replace(".dat", ".fft")
                        fft_file_nameonly = os.path.basename(fft_file)

                        DM_trial_was_searched = check_if_DM_trial_was_searched(dat_file, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, verbosity_level)

                        if dict_flag_steps['flag_step_realfft'] == 1:

                                if DM_trial_was_searched == False:
                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Doing realfft  of %s..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        realfft(dat_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, 0, flag_LOG_append)
                                        print("done!")
                                        sys.stdout.flush()

                                        if flag_remove_datfiles_of_segments ==1 and (segment_label != "full") and os.path.exists(dat_file):
                                                if verbosity_level >= 1:
                                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Removing %s to save disk space (use \"%s\" to recreate it)..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly, dat_file_nameonly+".makecmd"), end=' ')
                                                        sys.stdout.flush()
                                                os.remove(dat_file)
                                                if verbosity_level >= 1:
                                                        print("done!")
                                                        sys.stdout.flush()

                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Doing rednoise of %s..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, dat_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        rednoise(fft_file, work_dir, log_dir, LOG_basename, "", presto_env_zmax_0, verbosity_level)
                                        print("done!")
                                        sys.stdout.flush()

                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Applying zapfile '%s' to '%s'..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapfile_nameonly, fft_file_nameonly), end=' ')
                                        sys.stdout.flush()
                                        zapped_fft_filename, zapped_inf_filename = zapbirds(fft_file, zapfile, work_dir, log_dir, LOG_basename, presto_env_zmax_0, verbosity_level)
                                        zapped_fft_nameonly = os.path.basename(zapped_fft_filename)
                                        print("done!")
                                        sys.stdout.flush()
                                else:
                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Already fully searched. Skipping..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search), end=' ')
                        else:
                                print("STEP_REALFFT = 0, skipping realfft, rednoise, zapbirds...")

                        if dict_flag_steps['flag_step_periodicity_search'] == 1:
                                if DM_trial_was_searched == False:
                                        if flag_acceleration_search == 1:
                                                for z in list_zmax:
                                                        tstart_accelsearch = time.time()
                                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Doing accelsearch of %s with zmax = %4d..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, z), end=' ')
                                                        sys.stdout.flush()
                                                        if int(z) == 0:
                                                                dict_env = copy.deepcopy(dict_env_zmax_0)
                                                                if verbosity_level >= 2:
                                                                        print("accelsearch:: zmax == 0 ----> dict_env = %s" % (dict_env))
                                                                flag_cuda = ""
                                                        else:
                                                                if flag_use_cuda == 1:
                                                                        dict_env = copy.deepcopy(dict_env_zmax_any)
                                                                        gpu_id = random.choice(list_cuda_ids)
                                                                        flag_cuda = " -cuda %d " % (gpu_id)
                                                                else:
                                                                        dict_env = copy.deepcopy(dict_env_zmax_0)
                                                                        flag_cuda = ""

                                                                if verbosity_level >= 2:
                                                                        print("periodicity_search_FFT:: zmax == %d ----> dict_env = %s" % (int(z), dict_env))
                                                                        print("periodicity_search_FFT:: Now check CUDA: list_cuda_ids = ", list_cuda_ids)
                                                                        print("periodicity_search_FFT:: flag_use_cuda = ", flag_use_cuda)
                                                                        print("periodicity_search_FFT:: flag_cuda = ", flag_cuda)

                                                        accelsearch_flags = other_flags_accelsearch + flag_cuda  # + " -flo %s -fhi %s" % (frequency_to_search_min, frequency_to_search_max)

                                                        accelsearch(fft_file, work_dir, log_abspath, numharm=numharm, zmax=z, other_flags=accelsearch_flags, dict_env=dict_env, verbosity_level=verbosity_level, flag_LOG_append=flag_LOG_append)
                                                        tend_accelsearch = time.time()
                                                        time_taken_accelsearch_s = tend_accelsearch - tstart_accelsearch
                                                        print("done in %.2f s!" % (time_taken_accelsearch_s)); sys.stdout.flush()
                                                        ACCEL_filename = fft_file.replace(".fft", "_ACCEL_%s" % (int(z)))

                                        print("Seg '%s' %d/%d | ck %d/%d | DM %d/%d - Will do jerk search of %s with zmax=%d, wmax=%d, numharm=%d at the end of the acceleration search of this chunk..." % (segment_label, i_seg+1, N_seg, i_ck+1, N_ck, N_files_searched+i+1, N_DMs_to_search, zapped_fft_nameonly, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm), end=' ')
                                        cmd_jerksearch = "accelsearch %s -zmax %d -wmax %d -numharm %d %s" % (jerksearch_flags, jerksearch_zmax, jerksearch_wmax, jerksearch_numharm, fft_file)
                                        list_jerksearch_commands.append(cmd_jerksearch)
                                        list_jerksearch_workdirs.append(work_dir)
                                        print()

                                
                TP = ThreadPool(num_simultaneous_jerksearches)
                N_commands = len(list_jerksearch_commands)
                print()
                print("Now doing parallelized jerk search using %d CPUs..." % num_simultaneous_jerksearches);  sys.stdout.flush()
                print()
                for k in range(len(list_jerksearch_commands)):
                        print("Queing jerk search command #%d: '%s'" % (k+1, list_jerksearch_commands[k]))
                        time.sleep(0.1)
                        TP.apply_async(execute_and_log_in_thread_pool, (list_jerksearch_commands[k], log_dir, list_jerksearch_workdirs[k], k, N_commands, 1) )
                print("\n")
                print("Running %d jerk search commands at once..." % (num_simultaneous_jerksearches)); sys.stdout.flush()
                TP.close()
                TP.join()
                print()
                print("%d commands completed!" % (len(list_jerksearch_commands)))




                                        
def make_birds_file(ACCEL_0_filename, out_dir, log_dir, log_filename, width_Hz, flag_grow=1, flag_barycentre=0, sigma_birdies_threshold=4, verbosity_level=0):
        infile_nameonly = os.path.basename(ACCEL_0_filename)
        infile_basename = infile_nameonly.replace("_ACCEL_0", "")
        birds_filename = ACCEL_0_filename.replace("_ACCEL_0", ".birds")
        log_file = open(log_filename, "a")

        # Skip first three lines
        if verbosity_level >= 1:
                print("make_birds_file:: Opening the candidates: %s" % (ACCEL_0_filename))

        candidate_birdies = sifting.candlist_from_candfile(ACCEL_0_filename)
        candidate_birdies.reject_threshold(sigma_birdies_threshold)

        # Write down candidates above a certain sigma threshold
        list_birdies = candidate_birdies.cands
        if verbosity_level >= 1:
                print("make_birds_file:: Number of birdies = %d" % (len(list_birdies)))
        file_birdies = open(birds_filename, "w")
        if verbosity_level >= 1:
                print("make_birds_file:: File_birdies: %s" % (birds_filename))
        for cand in list_birdies:
                file_birdies.write("%.3f     %.20f     %d     %d     %d\n" % (cand.f, width_Hz, cand.numharm, flag_grow, flag_barycentre))
        file_birdies.close()

        return birds_filename


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
        fft_infile_nameonly = os.path.basename(fft_infile)
        fft_infile_basename = os.path.splitext(fft_infile_nameonly)[0]
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        # file_log = open(log_abspath, "w"); file_log.close()
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}

        # accelsearch

        if check_zaplist_outfiles(fft_infile) == False:
                if verbosity_level >= 2:
                        print("Doing accelsearch...", end=' ')
                        sys.stdout.flush()
                        print(fft_infile, birds_numharm, 0, other_flags_accelsearch, presto_env, verbosity_level)
                accelsearch(fft_infile, out_dir, log_abspath, birds_numharm, 0, other_flags_accelsearch, dict_env, verbosity_level)
                if verbosity_level >= 2:
                        print("Done accelsearch!")
                ACCEL_0_filename = fft_infile.replace(".fft", "_ACCEL_0")
                fourier_bin_width_Hz = get_Fourier_bin_width(fft_infile)
                if verbosity_level >= 2:
                        print("fourier_bin_width_Hz: ", fourier_bin_width_Hz)
                        print("Doing make_birds_file")
                        sys.stdout.flush()
                try:
                        birds_filename = make_birds_file(ACCEL_0_filename=ACCEL_0_filename, out_dir=out_dir, log_dir=log_dir, log_filename=log_abspath, width_Hz=fourier_bin_width_Hz, flag_grow=1, flag_barycentre=0, sigma_birdies_threshold=4, verbosity_level=0)
                except:
                        print()
                        print("WARNING: no further birdies found in the 0-DM time series: very clean band/very good mask?")
                        
                        infile_nameonly = os.path.basename(ACCEL_0_filename)
                        infile_basename = infile_nameonly.replace("_ACCEL_0", "")
                        birds_filename = ACCEL_0_filename.replace("_ACCEL_0", ".birds")

                file_common_birdies = open(common_birdies_filename, 'r')
                file_birds          = open(birds_filename, 'a')

                for line in file_common_birdies:
                        file_birds.write(line)
                file_birds.close()

                if verbosity_level >= 2:
                        print("Done make_birds_file!")
                        sys.stdout.flush()

                
                cmd_makezaplist = "makezaplist.py %s" % (birds_filename)
                if verbosity_level >= 2:
                        print("***********************************************")
                        sys.stdout.flush()
                        print("Doing execute_and_log")
                        sys.stdout.flush()
                        print("cmd_makezaplist = ", cmd_makezaplist)
                        sys.stdout.flush()
                execute_and_log(cmd_makezaplist, out_dir, log_abspath, dict_env, 0)
                if verbosity_level >= 2:
                        print("Done execute_and_log!")
                        sys.stdout.flush()
                        print("***********************************************")

        else:
                if verbosity_level >= 1:
                        print("Zaplist for %s already exists! " % (fft_infile_basename), end=' ')

        zaplist_filename = fft_infile.replace(".fft", ".zaplist")
        return zaplist_filename


def rednoise(fftfile, out_dir, log_dir, LOG_basename, other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0):
        # print "rednoise:: Inside rednoise"
        fftfile_nameonly = os.path.basename(fftfile)
        fftfile_basename = os.path.splitext(fftfile_nameonly)[0]
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)

        dereddened_ffts_filename = "%s/dereddened_ffts.txt" % (out_dir)
        fftfile_rednoise_abspath = os.path.join(out_dir, "%s_red.fft" % (fftfile_basename))
        inffile_rednoise_abspath = os.path.join(out_dir, "%s_red.inf" % (fftfile_basename))
        inffile_original_abspath = os.path.join(out_dir, "%s.inf" % (fftfile_basename))

        cmd_rednoise = "rednoise %s %s" % (other_flags, fftfile)

        if verbosity_level >= 2:
                print("rednoise:: dereddened_ffts_filename = ", dereddened_ffts_filename)
                print("rednoise:: fftfile_rednoise_abspath = ", fftfile_rednoise_abspath)
                print("rednoise:: cmd_rednoise = ", cmd_rednoise)
                # print "%s | Running:" % (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M"); sys.stdout.flush()
                # print "%s" % (cmd_rednoise) ; sys.stdout.flush()
                print("rednoise:: opening '%s'" % (dereddened_ffts_filename))

        try:
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')
        except:
                if verbosity_level >= 2:
                        print("rednoise:: File '%s' does not exist. Creating it..." % (dereddened_ffts_filename), end=' ') ; sys.stdout.flush()
                os.mknod(dereddened_ffts_filename)
                if verbosity_level >= 2:
                        print("done!") ; sys.stdout.flush()
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')

        # If the fftfile is already in the list of dereddened files...
        if "%s\n" % (fftfile) in file_dereddened_ffts.readlines():
                if verbosity_level >= 2:
                        print("rednoise:: NB: File '%s' is already in the list of dereddened files (%s)." % (fftfile, dereddened_ffts_filename))
                        # Then check is the file has size > 0...
                        print("rednoise:: Checking the size of '%s'" % (fftfile))

                if (os.path.getsize(fftfile) > 0):
                        operation = "skip"
                        if verbosity_level >= 2:
                                print("rednoise:: size is > 0. Then skipping...")
                else:
                        operation = "make_from_scratch"
                        if verbosity_level >= 2:
                                print("rednoise:: size is = 0. Making from scratch...")

        else:
                operation = "make_from_scratch"
                if verbosity_level >= 2:
                        print("rednoise:: File '%s' IS NOT in the list of dereddened files (%s). I will make the file from scratch..." % (fftfile_basename, dereddened_ffts_filename))

        file_dereddened_ffts.close()

        if operation =="make_from_scratch":
                if verbosity_level >= 2:
                        print("rednoise:: making the file from scratch...")
                dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
                execute_and_log(cmd_rednoise, out_dir, log_abspath, dict_env, 0)
                if verbosity_level >= 2:
                        print("done!", end=' ')
                        sys.stdout.flush()
                file_dereddened_ffts = open(dereddened_ffts_filename, 'a')
                file_dereddened_ffts.write("%s\n" % (fftfile))
                file_dereddened_ffts.close()
                os.rename(fftfile_rednoise_abspath, fftfile_rednoise_abspath.replace("_red.", "."))
                os.rename(inffile_rednoise_abspath, inffile_rednoise_abspath.replace("_red.", "."))


def realfft(infile, out_dir, log_dir, LOG_basename, other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0, flag_LOG_append=0):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        fftfile_abspath = os.path.join(out_dir, "%s.fft" % (infile_basename))
        cmd_realfft = "realfft %s %s" % (other_flags, infile)
        if verbosity_level >= 2:
                print("%s | realfft:: Running:" % (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M"))
                sys.stdout.flush()
                print("%s" % (cmd_realfft))
                sys.stdout.flush()

        if os.path.exists(fftfile_abspath ) and (os.path.getsize(fftfile_abspath) > 0):
                if verbosity_level >= 1:
                        print()
                        print("WARNING: File %s already present. Skipping realfft..." % (fftfile_abspath), end=' ')
        else:
                dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
                execute_and_log(cmd_realfft, out_dir, log_abspath, dict_env, 0)
                if os.path.exists(fftfile_abspath ) and (os.stat(fftfile_abspath).st_size > 0):
                        if verbosity_level >= 2:
                                print("%s | realfft on \"%s\" completed successfully!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
                                sys.stdout.flush()
                else:
                        print("WARNING (%s) | could not find all the output files from realfft on \"%s\"!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
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
                print("ERROR: Invalid value for barycentering option: \"%s\"" % (reference))
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

        if verbosity_level >= 2:
                print("%s | Running:" % (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M"))
                sys.stdout.flush()
                print("%s" % (cmd_prepdata))
                sys.stdout.flush()



        if os.path.exists(datfile_abspath ) and os.path.exists( inffile_abspath):
                if verbosity_level >= 1:
                        print()
                        print("WARNING: File '%s.dat' and '%s.inf' already present. Skipping and checking results..." % (outfile_basename, outfile_basename), end=' ')
        else:
                dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}

                execute_and_log(cmd_prepdata, out_dir, log_abspath, dict_env, 0)
                if os.path.exists(datfile_abspath ) and os.path.exists( inffile_abspath):
                        if verbosity_level >= 2:
                                print("%s | prepdata on \"%s\" completed successfully!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
                                sys.stdout.flush()
                else:
                        print("WARNING (%s) | could not find all the output files from prepdata on \"%s\"!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
                        sys.stdout.flush()


def make_rfifind_mask(infile, out_dir, log_dir, LOG_basename, time=2.0, time_intervals_to_zap="", chans_to_zap="", other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0):
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
        if verbosity_level >= 2:
                print("%s | Running:" % (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M"))
                sys.stdout.flush()
                print("%s" % (cmd_rfifind))
                sys.stdout.flush()

        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}

        execute_and_log(cmd_rfifind, out_dir, log_abspath, dict_env, 0)
        if verbosity_level >= 1:
                print("done!")

        if check_rfifind_outfiles(out_dir, infile_basename) == True:
                if verbosity_level >= 2:
                        print("make_rfifind_mask:: %s | rfifind on \"%s\" completed successfully!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
                        sys.stdout.flush()
        else:
                print("WARNING (%s) | could not find all the output files from rfifind on \"%s\"!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly))
                sys.stdout.flush()
                raise Exception("Your STEP_RFIFIND flag is set to 0, but the rfifind files could not be found!")

        mask_file = "%s/%s_rfifind.mask" % (out_dir, infile_basename)
        result = get_rfifind_result(mask_file, log_abspath, verbosity_level)

def get_DD_scheme_from_DDplan_output(output_DDplan, N_DMs_per_prepsubband, nsubbands):
        list_dict_schemes = []
        downsamp = 1
        output_DDplan_list_lines = output_DDplan.split("\n")
        if nsubbands == 0:
                index = output_DDplan_list_lines.index("  Low DM    High DM     dDM  DownSamp   #DMs  WorkFract")   + 1
        else:
                index = output_DDplan_list_lines.index("  Low DM    High DM     dDM  DownSamp  dsubDM   #DMs  DMs/call  calls  WorkFract")   + 1
                
        print
        print("+++++++++++++++++++++++++++++++++")
        print(output_DDplan)
        print("+++++++++++++++++++++++++++++++++")

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
                                        print("num_DMs else =", num_DMs)
                                        print("dict_scheme else =", dict_scheme)
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
        for dm in np.arange(DD_scheme['loDM'], DD_scheme['highDM'] - 0.5*DD_scheme['dDM'], DD_scheme['dDM']):
                if verbosity_level >= 2:
                        print("check_prepsubband_result_single_scheme:: Looking for: ", [os.path.join(work_dir, "*DM%.2f.dat"%(dm) )],  [os.path.join(work_dir, "*DM%.2f.inf"%(dm) )] )
                        print("check_prepsubband_result_single_scheme:: This is what I found: %s, %s" % ([ x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.dat"%(dm))) if not "_red" in x]  , [ x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.inf"%(dm))) if not "_red" in x]    ))
                if len([ x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.dat"%(dm))) if not "_red" in x]   + [ x for x in glob.glob(os.path.join(work_dir, "*DM%.2f.inf"%(dm))) if not "_red" in x] ) != 2:
                        if verbosity_level >= 2:
                                print("check_prepsubband_result_single_scheme: False")
                        return False
        if verbosity_level >= 2:
                print("check_prepsubband_result_single_scheme: True")

        return True


def get_DDplan_scheme(infile, out_dir, log_dir, LOG_basename, loDM, highDM, DM_coherent_dedispersion, N_DMs_per_prepsubband, freq_central_MHz, bw_MHz, nchan, nsubbands, t_samp_s):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)
        
        if np.float64(DM_coherent_dedispersion) == 0:
                if nsubbands == 0:                        
                        cmd_DDplan = "DDplan.py -o ddplan_%s -l %s -d %s -f %s -b %s -n %s -t %s" % (infile_basename, loDM, highDM, freq_central_MHz, np.fabs(bw_MHz), nchan, t_samp_s)
                else:
                        cmd_DDplan = "DDplan.py -o ddplan_%s -l %s -d %s -f %s -b %s -n %s -t %s -s %s" % (infile_basename, loDM, highDM, freq_central_MHz, np.fabs(bw_MHz), nchan, t_samp_s, nsubbands)
                        print("Use of subbands enabled with %s subbands (number of channels in the data: %d)" % (nsubbands, nchan))

        elif np.float64(DM_coherent_dedispersion) > 0:
                print("Coherent dedispersion enabled with DM = %.3f pc cm-3" % (np.float64(DM_coherent_dedispersion)))
                if nsubbands == 0: 
                        cmd_DDplan = "DDplan.py -o ddplan_%s -l %s -d %s -c %s -f %s -b %s -n %s -t %s" % (infile_basename, loDM, highDM, DM_coherent_dedispersion, freq_central_MHz, np.fabs(bw_MHz), nchan, t_samp_s)
                else:
                        cmd_DDplan = "DDplan.py -o ddplan_%s -l %s -d %s -c %s -f %s -b %s -n %s -t %s -s %s" % (infile_basename, loDM, highDM, DM_coherent_dedispersion, freq_central_MHz, np.fabs(bw_MHz), nchan, t_samp_s, nsubbands)
                        print("Use of subbands enabled with %s subbands (number of channels in the data: %d)" % (nsubbands, nchan))
                        
        elif np.float64(DM_coherent_dedispersion) < 0:
                print("ERROR: The DM of coherent dedispersion < 0! Exiting...")
                exit()

        print("Running:  \033[1m %s \033[0m" % (cmd_DDplan))
        output_DDplan = get_command_output(cmd_DDplan, shell_state=False, work_dir=out_dir)

        list_DD_schemes = get_DD_scheme_from_DDplan_output(output_DDplan, N_DMs_per_prepsubband, nsubbands)
        
        return list_DD_schemes


def dedisperse(infile, out_dir, log_dir, LOG_basename, segment_label, chunk_label, Nsamples, ignorechan_list, mask_file, list_DD_schemes, nchan, nsubbands=0, num_simultaneous_prepsubbands=1, other_flags="", presto_env=os.environ['PRESTO'], verbosity_level=0):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        prepsubband_outfilename = "%s_%s_%s" % (infile_basename, segment_label, chunk_label)
        dict_env = {'PRESTO': presto_env, 'PATH': "%s/bin:%s" % (presto_env, os.environ['PATH']), 'LD_LIBRARY_PATH': "%s/lib:%s" % (presto_env, os.environ['LD_LIBRARY_PATH'])}
        file_script_prepsubband_name = "script_prepsubband.txt"
        file_script_prepsubband_abspath = "%s/%s" % (out_dir, file_script_prepsubband_name)
        
        log_abspath = "%s/LOG_%s.txt" % (log_dir, LOG_basename)

        
        N_schemes = len(list_DD_schemes)

        string_mask = ""
        if mask_file != "":
                string_mask = "-mask %s" % (mask_file)
        string_ignorechan = ""
        if ignorechan_list != "":
                string_ignorechan = "-ignorechan %s" % (ignorechan_list)

        print("----------------------------------------------------------------------")
        print("prepsubband 将运行 %d 次，使用以下 DM 范围：" % (N_schemes))
        print()
        print("%10s %10s %10s %10s %10s " % ("低 DM", "高 DM", "dDM",  "下采样",   "DM 数量"))
        for i in range(N_schemes):
                offset = 0
                if i == N_schemes-1 : offset = 1
                print("%10.3f %10.3f %10s %10s %10d " % (list_DD_schemes[i]['loDM'], np.float64(list_DD_schemes[i]['loDM']) + int(list_DD_schemes[i]['num_DMs'])*np.float64(list_DD_schemes[i]['dDM']), list_DD_schemes[i]['dDM'],  list_DD_schemes[i]['downsamp'],  list_DD_schemes[i]['num_DMs'] + offset))
        print()
        sys.stdout.flush()
        
        if nsubbands == 0:
                nsubbands = nchan
        elif (nchan % nsubbands != 0):
                print("错误：请求的子带数量为 %d，这不是通道数量 %d 的整数倍！" % (nsubbands, nchan))
                exit()

        file_script_prepsubband = open(file_script_prepsubband_abspath, "w")
        for i in range(N_schemes):
                flag_numout = ""
                if i < N_schemes-1:
                        cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'], list_DD_schemes[i]['downsamp'], nsubbands, infile)
                elif i == N_schemes-1:
                        cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'] + 1, list_DD_schemes[i]['downsamp'], nsubbands, infile)
                file_script_prepsubband.write("%s\n" % cmd_prepsubband)

        file_script_prepsubband.close()

        print("使用 %d 个子带进行去色散（原始通道数量：%d）" % (nsubbands, nchan))
        print()
        if N_schemes == 1:
                print("\033[1m >> 提示：\033[0m 使用 '\033[1mtail -f %s\033[0m' 查看 prepsubband 的进度" % (log_abspath))
        elif N_schemes > 1:
                print("\033[1m >> 提示：\033[0m 使用 '\033[1mfor f in %s/LOG_prepsubband_*.txt; do tail -1 ${f}; echo; done\033[0m' 查看 prepsubband 的进度" % (log_dir))
        print()

        N_prepsubband_schemes_done = 0
        list_prepsubband_commands = []
        list_prepsubband_workdirs = []


        while (N_prepsubband_schemes_done < N_schemes):
                if verbosity_level >= 2:
                        print("dedisperse:: N_prepsubband_schemes_done = ", N_prepsubband_schemes_done)
                for i in range(N_schemes):
                        flag_numout = ""
                        loDM = np.float64(list_DD_schemes[i]['loDM'])
                        dDM  = np.float64(list_DD_schemes[i]['dDM'])
                        hiDM = loDM + int(list_DD_schemes[i]['num_DMs'])*dDM

                        if i < N_schemes-1:
                                str_parentesis = ")"
                                cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs']    , list_DD_schemes[i]['downsamp'], nsubbands, infile)
                        elif i == N_schemes-1:
                                str_parentesis = "]"
                                cmd_prepsubband = "prepsubband %s %s -o %s %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (other_flags, flag_numout, prepsubband_outfilename, string_ignorechan, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'] + 1, list_DD_schemes[i]['downsamp'], nsubbands, infile)

                        if check_prepsubband_result_single_scheme(out_dir, list_DD_schemes[i], verbosity_level) == False:
                                if num_simultaneous_prepsubbands == 1 or N_schemes == 1:
                                        if verbosity_level >= 1:
                                                print("Running  prepsubband for DM range [%.3f-%.3f%s pc cm-3 (scheme %d/%d) on observation '%s'..." % (loDM, hiDM, str_parentesis, i+1, N_schemes, infile), end=' '); sys.stdout.flush()
                                        if verbosity_level >= 2:
                                                print("dedisperse:: %d) RUNNING: %s" % (i, cmd_prepsubband))

                                        execute_and_log("which prepsubband", out_dir, log_abspath, dict_env, 1)
                                        execute_and_log(cmd_prepsubband, out_dir, log_abspath, dict_env, 1)
                                        if check_prepsubband_result_single_scheme(out_dir, list_DD_schemes[i], verbosity_level) == True:
                                                N_prepsubband_schemes_done = N_prepsubband_schemes_done + 1
                                        if verbosity_level >= 1:
                                                print("done!"); sys.stdout.flush()

                                elif num_simultaneous_prepsubbands > 1 and N_schemes > 1:
                                        list_prepsubband_commands.append(cmd_prepsubband)
                                        list_prepsubband_workdirs.append(out_dir)
                                        #print("list_prepsubband_commands = %s", list_prepsubband_commands)
                                        N_prepsubband_schemes_done = N_prepsubband_schemes_done + 1
        

                        else:
                                print("WARNING: prepsubband for DM range [%.3f-%.3f%s pc cm-3 (scheme %d/%d) on observation '%s' already successfully run. Skipping..." % (list_DD_schemes[i]['loDM'], hiDM, str_parentesis, i+1, N_schemes, infile))
                                N_prepsubband_schemes_done = N_prepsubband_schemes_done + 1

        if num_simultaneous_prepsubbands > 1 and N_schemes > 1:
                TP = ThreadPool(num_simultaneous_prepsubbands)
                N_commands = len(list_prepsubband_commands)
                print()
                print("Total of %d prepsubband commands, running %d of them at once..." % (N_commands, num_simultaneous_prepsubbands));  sys.stdout.flush()
                print()
                for k in range(len(list_prepsubband_commands)):
                        print("Queing prepsubband command #%d: '%s'" % (k+1, list_prepsubband_commands[k]))
                        time.sleep(0.1)
                        TP.apply_async(execute_and_log_in_thread_pool, (list_prepsubband_commands[k], log_dir, list_prepsubband_workdirs[k], k, N_commands, 1) )
                TP.close()
                TP.join()
                print()
                print("%d commands completed!" % (len(list_prepsubband_commands)))


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
        if verbosity_level >= 2:
                print("zapbirds:: list_zapped_ffts_abspath = ", list_zapped_ffts_abspath)

        if check_zapbirds_outfiles(fft_infile, list_zapped_ffts_abspath, verbosity_level=0) == False:
                if verbosity_level >= 2:
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

                        print("dedisperse_rednoise_and_periodicity_search_FFT:: 段标签: '%s'" % (segment_label))
                        print("搜索字符串 = ", search_string)
                        print("需要分割的dat文件列表 = ", list_datfiles_to_split)

                        segment_min = np.float64(segment_label.replace("m", ""))
                        i_chunk = int(chunk_label.replace("ck", ""))
                        split_into_chunks(infile, list_datfiles_to_split, log_dir, LOG_basename, out_dir, segment_min, i_chunk, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax, presto_env=os.environ['PRESTO'], flag_LOG_append=1, flag_remove_datfiles_of_segments=flag_remove_datfiles_of_segments, verbosity_level=verbosity_level)


                print("dedisperse_rednoise_and_periodicity_search_FFT:: 正在启动周期性搜索")
                sys.stdout.flush()
                print("dedisperse_rednoise_and_periodicity_search_FFT:: 正在查找 %s/*DM*.dat" % (out_dir))
                print("dedisperse_rednoise_and_periodicity_search_FFT:: CUDA设备ID列表 = %s" % (list_cuda_ids))
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






