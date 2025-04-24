#!/usr/bin/env python3
"""
Created on 2025.3.1
@author: Long Peng
@web page: https://www.plxray.cn/
qq:2107053791

need: 
主程序
"""


import os,sys
import numpy as np
from psr_fuc import *
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools
from datetime import datetime

try:
    from presto import filterbank, infodata, parfile, psr_utils, psrfits, rfifind, sifting
except:
    print("\n错误：无法加载 PRESTO 的 Python 模块！")
    print("请确保您的 PRESTO Python 模块已正确安装并可以正常使用。\n")
    exit()

import warnings
from multiprocessing.pool import ThreadPool
warnings.simplefilter('ignore', UserWarning)

# 获取环境变量
def get_env_var(var_name, default=None, required=False):
    value = os.environ.get(var_name, default)
    if required and value is None:
        print(f"Error: {var_name} is not set.")
        sys.exit(1)
    return value

def write_section_header(f, title, line_length=60, line_char="#"):
    """
    写入一个带有标题的分隔线到文件中。
    """
    f.write("\n#===============================================================\n")
    f.write(f"# {title}\n")
    f.write("#===============================================================\n\n")

def prep_configure(observation_filename):
    """
    生产准备脉冲星搜索处理的配置文件。
    参数：
        input_obs_file (str): 搜寻的文件。
    """

    makedir("known_pulsars")
    makedir("01_RFIFIND")
    default_file_format = "psrfits"  
    default_obs = observation_filename if observation_filename else "*fits"

    # 根据输入的观测文件名判断文件格式并设置默认文件格式
    if observation_filename:
        if observation_filename.endswith(".fil"):
            print(f"输入文件 '{observation_filename}' 似乎是 filterbank 格式，默认格式设置为 'filterbank'。")
            default_file_format = "filterbank"
        elif observation_filename.endswith((".fits", ".sf")) and psrfits.is_PSRFITS(observation_filename):
            print(f"输入文件 '{observation_filename}' 似乎是 PSRFITS 格式，默认格式设置为 'psrfits'。")
        else:
            print(f"\n警告：无法确定输入文件 '{observation_filename}' 的格式，默认格式设置为 'psrfits'。")
    else:
        print("警告：未提供输入观测文件，默认格式设置为 'psrfits'。")

    # 尝试获取 PRESTO 环境变量的路径
    presto_path = os.environ.get('PRESTO', "*** PRESTO 环境变量未定义 ***")
    presto_gpu_path = os.environ.get('PRESTO2_ON_GPU') or os.environ.get('PRESTO_ON_GPU') or presto_path
    use_cuda = '1' if presto_gpu_path and presto_gpu_path != presto_path else '0'

    if use_cuda == '0':
        print("警告：未定义 PRESTO2_ON_GPU 或 PRESTO_ON_GPU，GPU 加速将不可用！")

    # 默认配置参数
    dict_survey_configuration_default_values = {
        'OBSNAME':                               "%s               # 默认请使用*fits ，通过-obs指定文件" %observation_filename,
        'SOURCE_NAME':                           "AQLX-1           # 源名" ,       
        'SEARCH_LABEL':                          "%s               # 当前搜索项目的标签，建议修改标志" % os.path.basename(os.getcwd()),
        'DATA_TYPE':                             "%-18s            # 数据类型选项：filterbank 或 psrfits" % (default_file_format),
        'IF_BARY':                               "1                # 是否执行质心修正？重要参数（1=是，0=否）。1需要给出正确的RA,DEC" ,    
        'RA':                                    " 17:20:54.5063   # 赤经eg: 17:20:54.5063 " ,    
        'DEC':                                   " -08:57:31.29    # 赤纬eg: -08:57:31.29  " ,    
        'POOL_NUM':                              "%s               # 多线程核数。（默认为一半） "%int(cpu_count()/2) ,
        'ROOT_WORKDIR':                          "%s               # 根工作目录的路径。"% os.getcwd(),
        'PRESTO':                                "%s               # 主要的 PRESTO 安装路径" % presto_path,
        'PRESTO_GPU':                            "%s               # PRESTO_ON_GPU 安装路径（如果存在）" % presto_gpu_path,
        'IF_DDPLAN':                             "1                # 是否执行ddplan？（1=是，0=否）",
        'DM_MIN':                                "2.0              # 搜索的最小色散",
        'DM_MAX':                                "100.0            # 搜索的最大色散",
        'DM_STEP':                           "[(20, 30, 0.1)]      # 自定义搜索的色散间隔列表，IF_DDPLAN=0时使用",
        'DM_COHERENT_DEDISPERSION':              "0                # 可能的相干去色散（CDD）的色散值（0 = 不进行 CDD）",
        'N_SUBBANDS':                            "128              # 使用的子带数量（0 = 使用所有通道）",
        'PERIOD_TO_SEARCH_MIN':                  "0.001            # 可接受的最小候选周期（秒）",
        'PERIOD_TO_SEARCH_MAX':                  "20.0             # 可接受的最大候选周期（秒）,毫秒脉冲星可改为0.040",
        'LIST_SEGMENTS':                         "full             # 用于搜索的分段长度（以分钟为单位），用逗号分隔（例如 \"full,20,10\"）,该功能目前不可用",
        'RFIFIND_TIME':                          "0.1              # RFIFIND 的 -time 选项值,FAST默认0.1",
        'RFIFIND_CHANS_TO_ZAP':                  "\"\"             # 在 RFIFIND 掩模中需要消除的通道列表",
        'RFIFIND_TIME_INTERVALS_TO_ZAP':         "\"\"             # 在 RFIFIND 掩模中需要消除的时间间隔列表",
        'IGNORECHAN_LIST':                       "\"\"             # 分析中完全忽略的通道列表（PRESTO -ignorechan 选项）",
        'ZAP_ISOLATED_PULSARS_FROM_FFTS':        "0                # 是否在功率谱中消除已知脉冲星？（1=是，0=否）",
        'ZAP_ISOLATED_PULSARS_MAX_HARM':         "8                # 如果在功率谱中消除已知脉冲星，消除到这个谐波次数",
        'FLAG_ACCELERATION_SEARCH':              "1                # 是否进行加速搜索？（1=是，0=否）",
        'ACCELSEARCH_LIST_ZMAX':                 "0                # 使用 PRESTO accelsearch 时的 zmax 值列表（用逗号分隔）",
        'ACCELSEARCH_NUMHARM':                   "8                # 加速搜索时使用的谐波数量",
        'FLAG_JERK_SEARCH':                      "0                # 是否进行jerk search？（1=是，0=否）",
        'JERKSEARCH_ZMAX':                       "100              # jerk search时使用的 zmax 值",
        'JERKSEARCH_WMAX':                       "300              # jerk search时使用的 wmax 值（0 = 不进行jerk search）",
        'JERKSEARCH_NUMHARM':                    "4                # jerk search时使用的谐波数量",
        'SIFTING_FLAG_REMOVE_DUPLICATES':        "1                # 在筛选时是否移除候选重复项？（1=是，0=否）",
        'SIFTING_FLAG_REMOVE_DM_PROBLEMS':       "1                # 是否移除在少数 DM 值中出现的候选项？（1=是，0=否）",
        'SIFTING_FLAG_REMOVE_HARMONICS':         "1                # 是否移除谐波相关的候选项？（1=是，0=否）",
        'SIFTING_MINIMUM_NUM_DMS':               "3                # 候选项必须出现的最小 DM 值数量，才被认为是“好的”",
        'SIFTING_MINIMUM_DM':                    "2.0              # 候选项必须出现的最小 DM 值，才被认为是“好的”",
        'SIFTING_SIGMA_THRESHOLD':               "4.0              # 候选项的最小可接受显著性",
        'FLAG_FOLD_KNOWN_PULSARS':               "1                # 是否折叠可能是已知脉冲星的候选项？（1=是，0=否）",
        'FLAG_FOLD_TIMESERIES':                  "0                # 是否使用时间序列折叠候选项（超快，但没有频率信息）？（1=是，0=否）",
        'FLAG_FOLD_RAWDATA':                     "1                # 是否使用原始数据文件折叠候选项（慢，但包含所有信息）？（1=是，0=否）",
        'FLAG_NUM':                              "50               # 折叠图片数量",
        'RFIFIND_FLAGS':                         "\"\"             # 为 RFIFIND 提供的其他选项",
        'PREPDATA_FLAGS':                        "\"\"             # 为 PREPDATA 提供的其他选项",
        'PREPSUBBAND_FLAGS':                     "\"-ncpus 4\"     # 为 PREPSUBBAND 提供的其他选项",
        'REALFFT_FLAGS':                         "\"\"             # 为 REALFFT 提供的其他选项",
        'REDNOISE_FLAGS':                        "\"\"             # 为 REDNOISE 提供的其他选项",
        'ACCELSEARCH_FLAGS':                     "\"\"             # 进行加速搜索时为 ACCELSEARCH 提供的其他选项",
        'ACCELSEARCH_GPU_FLAGS':                 "\"\"             # 使用 PRESTO_ON_GPU 进行加速搜索时为 ACCELSEARCH 提供的其他选项",
        'ACCELSEARCH_JERK_FLAGS':                "\"\"             # 进行jerk search时为 ACCELSEARCH 提供的其他选项",
        'PREPFOLD_FLAGS':                        "\"-ncpus %-3d -n 64\"     # 为 PREPFOLD 提供的其他选项" % (multiprocessing.cpu_count() / 4),
        'FLAG_SINGLEPULSE_SEARCH':               "1                # 是否进行单脉冲搜索？（1=是，0=否）",
        'SINGLEPULSE_SEARCH_FLAGS':              "\"\"             # 进行单脉冲搜索时为 SINGLE_PULSE_SEARCH.py 提供的其他选项",
        'USE_CUDA':                              "%s               # 是否使用 GPU 加速？（1=是，0=否）" % use_cuda,
        'CUDA_IDS':                              "0                # 使用的 NVIDIA GPU 的 ID（用逗号分隔，例如 \"0,1,2,3\" - 使用 'nvidia-smi' 检查）",
        'NUM_SIMULTANEOUS_JERKSEARCHES':         "%-4d             # 同时运行的jerk search实例数量" % (multiprocessing.cpu_count()),
        'NUM_SIMULTANEOUS_PREPFOLDS':            "4                # 同时运行的 prepfold 实例的最大数量",
        'NUM_SIMULTANEOUS_PREPSUBBANDS':         "%-4d             # 同时运行的 prepsubband 实例的最大数量" % (multiprocessing.cpu_count() / 4),
        'MAX_SIMULTANEOUS_DMS_PER_PREPSUBBAND':  "1000             # prepsubband 一次处理的最大 DM 值数量（最大 1000）",
        'NUM_SIMULTANEOUS_SINGLEPULSE_SEARCHES': "%-4d             # 同时运行的单脉冲搜索实例数量" % (multiprocessing.cpu_count()),
        'FAST_BUFFER_DIR':                       "\"\"             # 快速内存缓冲区路径（可选，最小化 I/O 瓶颈）",
        'FLAG_KEEP_DATA_IN_BUFFER_DIR':          "0                # 搜索后是否在缓冲区保留观测数据副本？（1=是，0=否）",
        'FLAG_REMOVE_FFTFILES':                  "0                # 搜索后是否删除 FFT 文件以节省磁盘空间？（1=是，0=否）",
        'FLAG_REMOVE_DATFILES_OF_SEGMENTS':      "1                # 搜索后是否删除较短分段的 .dat 文件以节省磁盘空间？（1=是，0=否）",
        'STEP_RFIFIND':                          "1                # 是否运行 RFIFIND 步骤？（1=是，0=否）",
        'STEP_ZAPLIST':                          "1                # 是否运行 ZAPLIST 步骤？（1=是，0=否）",
        'STEP_DEDISPERSE':                       "1                # 是否运行去色散步骤？（1=是，0=否）",
        'STEP_REALFFT':                          "1                # 是否运行 REALFFT 步骤？（1=是）",
        'STEP_PERIODICITY_SEARCH':               "1                # 是否运行周期性搜索步骤？（1=是，0=否）",
        'STEP_SIFTING':                          "1                # 是否运行筛选步骤？（1=是，0=否）",
        'STEP_FOLDING':                          "1                # 是否运行折叠步骤？（1=是，0=否）",
        'STEP_SINGLEPULSE_SEARCH':               "1                # 是否运行单脉冲搜索步骤？（1=是，0=否）"
    }

    default_cfg_filename = "%s.cfg" % (os.path.basename(os.getcwd()))
    if os.path.exists(default_cfg_filename):
            default_cfg_filename_existing = default_cfg_filename
            default_cfg_filename = "%s_2.cfg" % (os.path.basename(os.getcwd()))
            print("******************")
            print_log(f"警告：{default_cfg_filename_existing} 已经存在！正在将默认配置保存到文件{default_cfg_filename}" ,masks=default_cfg_filename,color=colors.WARNING)
            print("******************")
            print()
    with open(default_cfg_filename, "w") as f:
        max_key_length = max(len(key) for key in dict_survey_configuration_default_values)
        i = 0
        for key, value in dict_survey_configuration_default_values.items():
            i += 1
            if '#' in value:
                main_value, comment = value.split('#', 1)
                formatted_value = f"{main_value.strip():<40} # {comment.strip()}"
            else:
                formatted_value = value.strip()
            if i == 1:
                write_section_header(f,'一般参数')
            if i == 12:
                write_section_header(f, '搜寻核心参数')
            if i == 21:
                write_section_header(f, '用PRESTO进行傅里叶域搜索')
            if i == 52:
                write_section_header(f, 'Single pulse search with PRESTO')  
            if i == 9:
                write_section_header(f, '计算/性能参数') 
            if i in [17, 20, 25, 27, 29, 34, 40, 44, 57, 62]:
                f.write("\n")              
            f.write(f"{key.ljust(max_key_length)} {formatted_value}\n")


    print_log(f"\n默认配置已写入{default_cfg_filename}",masks=default_cfg_filename,color=colors.OKBLUE)

    with open("common_birdies.txt", "w") as f:
            f.write("10.00   0.003     2     1     0\n")
            f.write("30.00    0.008     2     1     0\n")
            f.write("50.00    0.08      3     1     0\n")
    print(f"一些常见的干扰频率已写入 'common_birdies.txt'。\n")
    print(f"如果有的话，请将已知脉冲星的参数文件放在 'known_pulsars' 文件夹中。\n")

    exit()

# def main():
cwd = os.getcwd()
ps_pl_path = os.path.abspath(os.path.dirname(__file__))

print(f"用法: {os.path.basename(sys.argv[0])} -obs <观测文件>")
if (len(sys.argv) == 1 or ("-h" in sys.argv) or ("-help" in sys.argv) or ("--help" in sys.argv)):
    # 打印程序的用法
    print('未指定obs，将使用*fits作为默认文件')
    obsname = "*fits"
    prep_configure(obsname)
    print()
    print(f"创建指定的搜寻配置文件请运行: \033[1m{os.path.basename(sys.argv[0])} -obs [<观测文件>]\033[0m")
    print()

else:
    for j in range(1, len(sys.argv)):
        if  (sys.argv[j] == "-obs"):
            obsname = sys.argv[j+1]
            prep_configure(obsname)


