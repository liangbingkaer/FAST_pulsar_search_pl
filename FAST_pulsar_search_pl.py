#!/usr/bin/env python3
"""
Created on 2025.3.1
@author: Long Peng
@web page: https://www.plxray.cn/
qq:2107053791

FAST射电脉冲搜寻主程序
"""

import os,sys
import numpy as np
from psr_fuc import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools
from datetime import datetime
import ast
import json

try:
    from presto import filterbank, infodata, parfile, psr_utils, psrfits, rfifind, sifting
except:
    print("\n错误：无法加载 PRESTO 的 Python 模块！")
    print("请确保您的 PRESTO Python 模块已正确安装并可以正常使用。\n")
    exit()

import warnings
from multiprocessing.pool import ThreadPool
warnings.simplefilter('ignore', UserWarning)

class colors:
    HEADER = '\033[95m'      # 亮紫色（Magenta），通常用于标题或重要提示
    OKBLUE = '\033[94m'      # 亮蓝色，用于正常信息或状态提示（运行各类程序）
    OKCYAN = '\033[96m'      # 亮青色（Cyan），用于正常信息或状态提示
    OKGREEN = '\033[92m'     # 亮绿色，通常用于表示成功或正常状态 （打印运行成功）
    WARNING = '\033[93m'     # 亮黄色，用于警告信息
    ERROR = '\033[91m'       # 亮红色，用于错误信息
    BOLD = '\033[1m'         # 加粗文本（不改变颜色），使文本更突出
    ENDC = '\033[0m'         # 重置文本格式（包括颜色和加粗等），恢复默认显示

##读取星历表文件时有用
class Pulsar(object):
    def __init__(self, parfilename):
        # 光速（单位：CGS，即厘米/秒）
        LIGHT_SPEED = 2.99792458e10  

        # 读取星历表文件参数
        pulsar_parfile = parfile.psr_par(parfilename)
        
        self.parfilename = parfilename  
        # 获取脉冲星名称（优先使用PSR，如果没有则使用PSRJ）
        if hasattr(pulsar_parfile, 'PSR'):
            self.psr_name = pulsar_parfile.PSR
        elif hasattr(pulsar_parfile, 'PSRJ'):
            self.psr_name = pulsar_parfile.PSRJ

        # 获取脉冲星的参考历元（PEPOCH）和自转频率（F0）
        self.PEPOCH = pulsar_parfile.PEPOCH
        self.F0 = pulsar_parfile.F0
        # 获取脉冲星的周期（秒和毫秒）
        self.P0_s = 1. / self.F0
        self.P0_ms = self.P0_s * 1000
        # 获取脉冲星的自转频率导数（F1和F2，如果存在）
        if hasattr(pulsar_parfile, 'F1'):
            self.F1 = pulsar_parfile.F1
        else:
            self.F1 = 0
        if hasattr(pulsar_parfile, 'F2'):
            self.F2 = pulsar_parfile.F2
        else:
            self.F2 = 0

        # 判断脉冲星是否为双星系统
        self.is_binary = hasattr(pulsar_parfile, 'BINARY')

        if self.is_binary:
            # 如果是双星系统，设置相关参数
            self.pulsar_type = "binary"  # 标记为双星系统
            self.binary_model = pulsar_parfile.BINARY  # 获取双星模型

            # 1) 轨道周期
            if hasattr(pulsar_parfile, 'PB'):
                self.Pb_d = pulsar_parfile.PB  # 轨道周期（天）
                self.Pb_s = self.Pb_d * 86400  # 轨道周期（秒）
                self.Fb0 = 1. / self.Pb_s  # 轨道频率
            elif hasattr(pulsar_parfile, 'FB0'):
                self.Fb0 = pulsar_parfile.FB0  # 轨道频率
                self.Pb_s = 1. / self.Fb0  # 轨道周期（秒）
                self.Pb_d = self.Pb_s / 86400.  # 轨道周期（天）

            # 2) 脉冲星轨道的投影半长轴
            self.x_p_lts = pulsar_parfile.A1  # 单位：光秒
            self.x_p_cm = pulsar_parfile.A1 * LIGHT_SPEED  # 单位：厘米

            # 3) 轨道偏心率
            if hasattr(pulsar_parfile, 'E'):
                self.ecc = pulsar_parfile.E
            elif hasattr(pulsar_parfile, 'ECC'):
                self.ecc = pulsar_parfile.ECC
            elif hasattr(pulsar_parfile, 'EPS1') and hasattr(pulsar_parfile, 'EPS2'):
                self.eps1 = pulsar_parfile.EPS1
                self.eps2 = pulsar_parfile.EPS2
                self.ecc = np.sqrt(self.eps1**2 + self.eps2**2)
            else:
                self.ecc = 0

            # 4) 近星点的经度
            if hasattr(pulsar_parfile, 'OM'):
                self.omega_p_deg = pulsar_parfile.OM  # 单位：度
            else:
                self.omega_p_deg = 0
            self.omega_p_rad = self.omega_p_deg * np.pi / 180  # 单位：弧度

            # 5) 近星点/升节点的过境时刻
            if hasattr(pulsar_parfile, 'T0'):
                self.T0 = pulsar_parfile.T0
                self.Tasc = self.T0
            elif hasattr(pulsar_parfile, 'TASC'):
                self.Tasc = pulsar_parfile.TASC
                self.T0 = self.Tasc

            # 计算脉冲星在轨道上的最大视向速度
            self.v_los_max = (2 * np.pi * self.x_p_cm / self.Pb_s)
            # 计算多普勒因子
            self.doppler_factor = self.v_los_max / LIGHT_SPEED

        else:
            # 如果脉冲星是孤立的
            self.pulsar_type = "isolated"
            self.v_los_max = 0
            self.doppler_factor = 1e-4  # 考虑地球绕太阳运动引起的多普勒效应

class Observation(object):
    counter = 0  # 用于控制只打印第一个实例信息

    def __init__(self, file_name, data_type="filterbank"):
        self.__class__.counter += 1
        self.show_log = (self.__class__.counter == 1)

        if self.show_log:
            print_log(f"\n正在读取{file_name}文件的绝对路径、文件名和扩展名....", color=colors.HEADER)
        self.file_abspath = os.path.abspath(file_name)
        self.file_nameonly = self.file_abspath.split("/")[-1]
        self.file_basename, self.file_extension = os.path.splitext(self.file_nameonly)
        self.file_buffer_copy = ""

        if data_type == "filterbank":
            if self.show_log:
                print_log("\n正在读取filterbank文件....", color=colors.HEADER)
            try:
                object_file = filterbank.FilterbankFile(self.file_abspath)
                self.N_samples = object_file.nspec
                self.t_samp_s = object_file.dt
                self.T_obs_s = self.N_samples * self.t_samp_s
                self.nbits = object_file.header['nbits']
                self.nchan = object_file.nchan
                self.chanbw_MHz = object_file.header['foff']
                self.bw_MHz = self.nchan * self.chanbw_MHz
                self.freq_central_MHz = object_file.header['fch1'] + object_file.header['foff'] * 0.5 * object_file.nchan
                self.freq_high_MHz = np.amax(object_file.freqs)
                self.freq_low_MHz = np.amin(object_file.freqs)
                self.MJD_int = int(object_file.header['tstart'])
                self.Tstart_MJD = object_file.header['tstart']
                self.source_name = object_file.header['source_name'].strip()
            except ValueError:
                if self.show_log:
                    print_log("警告：读取时出现值错误！尝试使用PRESTO的'readfile'获取必要信息...", color=colors.WARNING), print()
                try:
                    self.N_samples = np.float64(readfile_with_str(f"readfile {self.file_abspath}", "grep 'Spectra per file'").split("=")[-1].strip())
                    self.t_samp_s = 1.0e-6 * float(readfile_with_str(f"readfile {file_name}", "grep 'Sample time (us)'").split("=")[-1].strip())
                    self.T_obs_s = self.N_samples * self.t_samp_s
                    self.nbits = int(readfile_with_str(f"readfile {file_name}", "grep 'bits per sample'").split("=")[-1].strip())
                    self.nchan = int(readfile_with_str(f"readfile {file_name}", "grep 'Number of channels'").split("=")[-1].strip())
                    self.chanbw_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'Channel width (MHz)'").split("=")[-1].strip())
                    self.bw_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'Total Bandwidth (MHz)'").split("=")[-1].strip())
                    self.Tstart_MJD = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'MJD start time'").split("=")[-1].strip())
                    self.freq_high_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'High channel (MHz)'").split("=")[-1].strip())
                    self.freq_low_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'Low channel (MHz)'").split("=")[-1].strip())
                    self.freq_central_MHz = (self.freq_high_MHz + self.freq_low_MHz) / 2.0
                    if self.show_log:
                        print_log('readfile读取信息成功', color=colors.OKGREEN)
                        print_log(f"N_samples: {self.N_samples}")
                        print_log(f"t_samp_s: {self.t_samp_s}")
                        print_log(f"T_obs_s: {self.T_obs_s}", color=colors.BOLD)
                        print_log(f"nbits: {self.nbits}")
                        print_log(f"nchan: {self.nchan}")
                        print_log(f"chanbw_MHz: {self.chanbw_MHz}")
                        print_log(f"bw_MHz: {self.bw_MHz}", color=colors.BOLD)
                        print_log(f"Tstart_MJD: {self.Tstart_MJD}")
                        print_log(f"freq_high_MHz: {self.freq_high_MHz}")
                        print_log(f"freq_central_MHz: {self.freq_central_MHz}")
                        print_log(f"freq_low_MHz: {self.freq_low_MHz}")
                except:
                    if self.show_log:
                        print_log("警告：'readfile'失败。尝试使用'header'获取必要信息...", color=colors.WARNING)
                    self.N_samples = np.abs(int(get_command_output("header %s -nsamples" % (self.file_abspath)).split()[-1]))
                    self.t_samp_s = np.float64(get_command_output("header %s -tsamp" % (self.file_abspath)).split()[-1]) * 1.0e-6
                    self.T_obs_s = np.float64(get_command_output("header %s -tobs" % (self.file_abspath)).split()[-1])
                    self.nbits = int(get_command_output("header %s -nbits" % (self.file_abspath)).split()[-1])
                    self.nchan = int(get_command_output("header %s -nchans" % (self.file_abspath)).split()[-1])
                    self.chanbw_MHz = np.fabs(np.float64(get_command_output("header %s -foff" % (self.file_abspath)).split()[-1]))
                    self.bw_MHz = self.chanbw_MHz * self.nchan
                    self.backend = get_command_output("header %s -machine" % (self.file_abspath)).split()[-1]
                    self.Tstart_MJD = np.float64(get_command_output("header %s -tstart" % (self.file_abspath)).split()[-1])
                    self.freq_high_MHz = np.float64(get_command_output("header %s -fch1" % (self.file_abspath)).split()[-1]) + 0.5 * self.chanbw_MHz
                    self.freq_central_MHz = self.freq_high_MHz - 0.5 * self.bw_MHz
                    self.freq_low_MHz = self.freq_high_MHz - self.bw_MHz
                    if self.show_log:
                        print_log(f"N_samples: {self.N_samples}")
                        print_log(f"t_samp_s: {self.t_samp_s} s")
                        print_log(f"T_obs_s: {self.T_obs_s} s", color=colors.BOLD)
                        print_log(f"nbits: {self.nbits} bits")
                        print_log(f"nchan: {self.nchan} channels")
                        print_log(f"chanbw_MHz: {self.chanbw_MHz} MHz")
                        print_log(f"bw_MHz: {self.bw_MHz} MHz", color=colors.BOLD)
                        print_log(f"backend: {self.backend}")
                        print_log(f"Tstart_MJD: {self.Tstart_MJD}")
                        print_log(f"freq_high_MHz: {self.freq_high_MHz} MHz")
                        print_log(f"freq_central_MHz: {self.freq_central_MHz} MHz")
                        print_log(f"freq_low_MHz: {self.freq_low_MHz} MHz")

        elif data_type == "psrfits":
            if self.show_log:
                print_log("\n正在读取PSRFITS文件....", color=colors.HEADER)
            if psrfits.is_PSRFITS(file_name):
                if self.show_log:
                    print_log("文件'%s'被正确识别为PSRFITS格式" % (file_name))
                object_file = psrfits.PsrfitsFile(self.file_abspath)
                self.bw_MHz = object_file.specinfo.BW
                self.N_samples = object_file.specinfo.N
                self.T_obs_s = object_file.specinfo.T
                self.backend = object_file.specinfo.backend
                self.nbits = object_file.specinfo.bits_per_sample
                self.date_obs = object_file.specinfo.date_obs
                self.dec_deg = object_file.specinfo.dec2000
                self.dec_str = object_file.specinfo.dec_str
                self.chanbw_MHz = object_file.specinfo.df
                self.t_samp_s = object_file.specinfo.dt
                self.freq_central_MHz = object_file.specinfo.fctr
                self.receiver = object_file.specinfo.frontend
                self.freq_high_MHz = object_file.specinfo.hi_freq
                self.freq_low_MHz = object_file.specinfo.lo_freq
                self.MJD_int = object_file.specinfo.mjd
                self.MJD_sec = object_file.specinfo.secs
                self.Tstart_MJD = self.MJD_int + np.float64(self.MJD_sec / 86400.)
                self.nchan = object_file.specinfo.num_channels
                self.observer = object_file.specinfo.observer
                self.project = object_file.specinfo.project_id
                self.ra_deg = object_file.specinfo.ra2000
                self.ra_str = object_file.specinfo.ra_str
                self.seconds_of_day = object_file.specinfo.secs
                self.source_name = object_file.specinfo.source
                self.telescope = object_file.specinfo.telescope

class SurveyConfiguration(object):
        def __init__(self, config_filename):
                self.config_filename = config_filename
                self.list_datafiles = []
                self.list_survey_configuration_ordered_params = ['OBSNAME',"SOURCE_NAME",'SEARCH_LABEL', 'DATA_TYPE','IF_BARY','RA','DEC','POOL_NUM ', 'ROOT_WORKDIR', 'PRESTO', 'PRESTO_GPU','IF_DDPLAN', 'DM_MIN', 'DM_MAX','DM_STEP', 'DM_COHERENT_DEDISPERSION', 'N_SUBBANDS', 'PERIOD_TO_SEARCH_MIN', 'PERIOD_TO_SEARCH_MAX', 'LIST_SEGMENTS', 'RFIFIND_TIME', 'RFIFIND_CHANS_TO_ZAP', 'RFIFIND_TIME_INTERVALS_TO_ZAP', 'IGNORECHAN_LIST', 'ZAP_ISOLATED_PULSARS_FROM_FFTS', 'ZAP_ISOLATED_PULSARS_MAX_HARM', 'FLAG_ACCELERATION_SEARCH', 'ACCELSEARCH_LIST_ZMAX', 'ACCELSEARCH_NUMHARM', 'FLAG_JERK_SEARCH', 'JERKSEARCH_ZMAX', 'JERKSEARCH_WMAX', 'JERKSEARCH_NUMHARM', 'SIFTING_FLAG_REMOVE_DUPLICATES', 'SIFTING_FLAG_REMOVE_DM_PROBLEMS', 'SIFTING_FLAG_REMOVE_HARMONICS', 'SIFTING_MINIMUM_NUM_DMS', 'SIFTING_MINIMUM_DM', 'SIFTING_SIGMA_THRESHOLD', 'FLAG_FOLD_KNOWN_PULSARS', 'FLAG_FOLD_TIMESERIES', 'FLAG_FOLD_RAWDATA','FLAG_NUM', 'RFIFIND_FLAGS', 'PREPDATA_FLAGS', 'PREPSUBBAND_FLAGS', 'REALFFT_FLAGS', 'REDNOISE_FLAGS', 'ACCELSEARCH_FLAGS', 'ACCELSEARCH_GPU_FLAGS', 'ACCELSEARCH_JERK_FLAGS', 'PREPFOLD_FLAGS', 'FLAG_SINGLEPULSE_SEARCH', 'SINGLEPULSE_SEARCH_FLAGS', 'USE_CUDA', 'CUDA_IDS', 'NUM_SIMULTANEOUS_JERKSEARCHES', 'NUM_SIMULTANEOUS_PREPFOLDS', 'NUM_SIMULTANEOUS_PREPSUBBANDS', 'MAX_SIMULTANEOUS_DMS_PER_PREPSUBBAND', 'FAST_BUFFER_DIR', 'FLAG_KEEP_DATA_IN_BUFFER_DIR', 'FLAG_REMOVE_FFTFILES', 'FLAG_REMOVE_DATFILES_OF_SEGMENTS', 'STEP_RFIFIND', 'STEP_ZAPLIST', 'STEP_DEDISPERSE', 'STEP_REALFFT', 'STEP_PERIODICITY_SEARCH', 'STEP_SIFTING', 'STEP_FOLDING', 'STEP_SINGLEPULSE_SEARCH']
                self.dict_survey_configuration = {}
                config_file = open(config_filename, "r" )

                for line in config_file:
                    line = line.strip()  # 移除行首和行尾的空白字符
                    if line and not line.startswith("#"):  # 过滤空行和注释行
                        line_content = line.split('#', 1)[0].strip()  # 移除行内注释
                        parts = line_content.split(None, 1)  # 按空白字符分割键和值
                        if len(parts) == 2:  # 如果分割后有两部分
                            key, value = parts[0], parts[1].strip().strip('"\'')  # 提取键和值，并移除引号
                        else:  # 如果只有键没有值
                            key, value = parts[0], ""  # 值为空字符串
                        self.dict_survey_configuration[key] = value  # 存储键值对
                                #list_line = shlex.split(line)
                                #self.dict_survey_configuration[list_line[0]] = list_line[1]  # Save parameter key and value in the dictionary 
                for key in list(self.dict_survey_configuration.keys()):
                        if   key == "OBSNAME":                           self.obsname                          = self.dict_survey_configuration[key]
                        elif key == "SOURCE_NAME":                       self.source_name                      = self.dict_survey_configuration[key]                     
                        elif key == "SEARCH_LABEL":                      self.search_label                     = self.dict_survey_configuration[key]
                        elif key == "DATA_TYPE":                         self.data_type                        = self.dict_survey_configuration[key]
                        elif key == "IF_BARY":                           self.ifbary                           = int(self.dict_survey_configuration[key])
                        elif key == "RA":                                self.ra                               = self.dict_survey_configuration[key]
                        elif key == "DEC":                               self.dec                              = self.dict_survey_configuration[key]
                        elif key == "POOL_NUM":                          self.pool_num                         = int(self.dict_survey_configuration[key])
                        elif key == "ROOT_WORKDIR":                      self.root_workdir                     = self.dict_survey_configuration[key]
                        elif key == "PRESTO":
                                if check_presto_path(presto_path=self.dict_survey_configuration[key], key=key) == True:
                                        self.presto_env                       = self.dict_survey_configuration[key]

                        elif key == "PRESTO_GPU":
                                if check_presto_path(presto_path=self.dict_survey_configuration[key], key=key) == True:
                                        self.presto_gpu_env                   = self.dict_survey_configuration[key]

                        elif key == "IF_DDPLAN":                            self.if_ddplan                             = int(self.dict_survey_configuration[key])
                        elif key == "DM_MIN":                               self.dm_min                                = self.dict_survey_configuration[key]
                        elif key == "DM_MAX":                               self.dm_max                                = self.dict_survey_configuration[key]
                        elif key == "DM_STEP":                              self.dm_step                               = ast.literal_eval(self.dict_survey_configuration[key])

                        elif key == "DM_COHERENT_DEDISPERSION":             self.dm_coherent_dedispersion              = self.dict_survey_configuration[key]
                        elif key == "N_SUBBANDS":                           self.nsubbands                             = int(self.dict_survey_configuration[key])

                        elif key == "PERIOD_TO_SEARCH_MIN":                 self.period_to_search_min                  = np.float64(self.dict_survey_configuration[key])
                        elif key == "PERIOD_TO_SEARCH_MAX":                 self.period_to_search_max                  = np.float64(self.dict_survey_configuration[key])
                        elif key == "LIST_SEGMENTS":                        self.list_segments                         = self.dict_survey_configuration[key].split(",")

                        elif key == "RFIFIND_TIME":                         self.rfifind_time                          = self.dict_survey_configuration[key]
                        elif key == "RFIFIND_CHANS_TO_ZAP":                 self.rfifind_chans_to_zap                  = self.dict_survey_configuration[key]
                        elif key == "RFIFIND_TIME_INTERVALS_TO_ZAP":        self.rfifind_time_intervals_to_zap         = self.dict_survey_configuration[key]
                        elif key == "IGNORECHAN_LIST":                      self.ignorechan_list                       = self.dict_survey_configuration[key]
                        elif key == "ZAP_ISOLATED_PULSARS_FROM_FFTS":       self.zap_isolated_pulsars_from_ffts        = int(self.dict_survey_configuration[key])
                        elif key == "ZAP_ISOLATED_PULSARS_MAX_HARM":        self.zap_isolated_pulsars_max_harm         = int(self.dict_survey_configuration[key])
			
                        elif key == "FLAG_ACCELERATION_SEARCH":             self.flag_acceleration_search              = int(self.dict_survey_configuration[key])
                        elif key == "ACCELSEARCH_LIST_ZMAX":                self.accelsearch_list_zmax                 = [int(x) for x in self.dict_survey_configuration[key].split(",")]
                        elif key == "ACCELSEARCH_NUMHARM":                  self.accelsearch_numharm                   = int(self.dict_survey_configuration[key])

                        elif key == "FLAG_JERK_SEARCH":                     self.flag_jerk_search                      = int(self.dict_survey_configuration[key])
                        elif key == "JERKSEARCH_ZMAX":                      self.jerksearch_zmax                       = int(self.dict_survey_configuration[key])
                        elif key == "JERKSEARCH_WMAX":                      self.jerksearch_wmax                       = int(self.dict_survey_configuration[key])
                        elif key == "JERKSEARCH_NUMHARM":                   self.jerksearch_numharm                    = int(self.dict_survey_configuration[key])

                        elif key == "SIFTING_FLAG_REMOVE_DUPLICATES":       self.sifting_flag_remove_duplicates        = int(self.dict_survey_configuration[key])
                        elif key == "SIFTING_FLAG_REMOVE_DM_PROBLEMS":      self.sifting_flag_remove_dm_problems       = int(self.dict_survey_configuration[key])
                        elif key == "SIFTING_FLAG_REMOVE_HARMONICS":        self.sifting_flag_remove_harmonics         = int(self.dict_survey_configuration[key])
                        elif key == "SIFTING_MINIMUM_NUM_DMS":              self.sifting_minimum_num_DMs               = int(self.dict_survey_configuration[key])
                        elif key == "SIFTING_MINIMUM_DM":                   self.sifting_minimum_DM                    = np.float64(self.dict_survey_configuration[key])
                        elif key == "SIFTING_SIGMA_THRESHOLD":              self.sifting_sigma_threshold               = np.float64(self.dict_survey_configuration[key])

                        elif key == "FLAG_FOLD_KNOWN_PULSARS":              self.flag_fold_known_pulsars               = int(self.dict_survey_configuration[key])
                        elif key == "FLAG_FOLD_TIMESERIES":                 self.flag_fold_timeseries                  = int(self.dict_survey_configuration[key])
                        elif key == "FLAG_FOLD_RAWDATA":                    self.flag_fold_rawdata                     = int(self.dict_survey_configuration[key])
                        elif key == "FLAG_NUM":                             self.fold_num                     = int(self.dict_survey_configuration[key])

                        elif key == "RFIFIND_FLAGS":                        self.rfifind_flags                         = self.dict_survey_configuration[key]
                        elif key == "PREPDATA_FLAGS":                       self.prepdata_flags                        = self.dict_survey_configuration[key]
                        elif key == "PREPSUBBAND_FLAGS":                    self.prepsubband_flags                     = self.dict_survey_configuration[key]
                        elif key == "REALFFT_FLAGS":                        self.realfft_flags                         = self.dict_survey_configuration[key]
                        elif key == "REDNOISE_FLAGS":                       self.rednoise_flags                        = self.dict_survey_configuration[key]
                        elif key == "ACCELSEARCH_FLAGS":                    self.accelsearch_flags                     = self.dict_survey_configuration[key]
                        elif key == "ACCELSEARCH_GPU_FLAGS":                self.accelsearch_gpu_flags                 = self.dict_survey_configuration[key]
                        elif key == "ACCELSEARCH_JERK_FLAGS":               self.accelsearch_jerk_flags                = self.dict_survey_configuration[key]
                        elif key == "PREPFOLD_FLAGS":                       self.prepfold_flags                        = self.dict_survey_configuration[key]

                        elif key == "FLAG_SINGLEPULSE_SEARCH":              self.flag_singlepulse_search               = int(self.dict_survey_configuration[key])
                        elif key == "SINGLEPULSE_SEARCH_FLAGS":             self.singlepulse_search_flags              = self.dict_survey_configuration[key]

                        elif key == "USE_CUDA":                             self.flag_use_cuda                         = int(self.dict_survey_configuration[key])
                        elif key == "CUDA_IDS":                             self.list_cuda_ids                         = [int(x) for x in self.dict_survey_configuration[key].split(",")]

                        elif key == "NUM_SIMULTANEOUS_JERKSEARCHES":           self.num_simultaneous_jerksearches           = int(self.dict_survey_configuration[key])
                        elif key == "NUM_SIMULTANEOUS_PREPFOLDS":              self.num_simultaneous_prepfolds              = int(self.dict_survey_configuration[key])
                        elif key == "NUM_SIMULTANEOUS_PREPSUBBANDS":           self.num_simultaneous_prepsubbands           = int(self.dict_survey_configuration[key])
                        elif key == "NUM_SIMULTANEOUS_SINGLEPULSE_SEARCHES":   self.num_simultaneous_singlepulse_searches   = int(self.dict_survey_configuration[key])
                        elif key == "MAX_SIMULTANEOUS_DMS_PER_PREPSUBBAND":    self.max_simultaneous_dms_per_prepsubband    = int(self.dict_survey_configuration[key])

                        elif key == "FAST_BUFFER_DIR":                      self.fast_buffer_dir                       = self.dict_survey_configuration[key]
                        elif key == "FLAG_KEEP_DATA_IN_BUFFER_DIR":         self.flag_keep_data_in_buffer_dir          = int(self.dict_survey_configuration[key])
                        elif key == "FLAG_REMOVE_FFTFILES":                 self.flag_remove_fftfiles                  = int(self.dict_survey_configuration[key])
                        elif key == "FLAG_REMOVE_DATFILES_OF_SEGMENTS":     self.flag_remove_datfiles_of_segments      = int(self.dict_survey_configuration[key])

                        elif key == "STEP_RFIFIND":                         self.flag_step_rfifind                     = int(self.dict_survey_configuration[key])
                        elif key == "STEP_ZAPLIST":                         self.flag_step_zaplist                     = int(self.dict_survey_configuration[key])
                        elif key == "STEP_DEDISPERSE":                      self.flag_step_dedisperse                  = int(self.dict_survey_configuration[key])
                        elif key == "STEP_REALFFT":                         self.flag_step_realfft                     = int(self.dict_survey_configuration[key])
                        elif key == "STEP_PERIODICITY_SEARCH":              self.flag_step_periodicity_search          = int(self.dict_survey_configuration[key])
                        elif key == "STEP_SIFTING":                         self.flag_step_sifting                     = int(self.dict_survey_configuration[key])
                        elif key == "STEP_FOLDING":                         self.flag_step_folding                     = int(self.dict_survey_configuration[key])
                        elif key == "STEP_SINGLEPULSE_SEARCH":              self.flag_step_singlepulse_search          = int(self.dict_survey_configuration[key])

                config_file.close()
                self.log_filename = "%s.log" % (self.search_label)
                self.list_0DM_datfiles = []
                self.list_0DM_fftfiles = []
                self.list_0DM_fftfiles_rednoise = []

                if "full" in self.list_segments:
                        self.list_segments_nofull        = copy.deepcopy(self.list_segments)
                        self.list_segments_nofull.remove("full")
                        self.flag_search_full = 1
                else:
                        self.list_segments_nofull        = copy.deepcopy(self.list_segments)
                        self.flag_search_full = 0


                self.dict_chunks = {}      # {'filename': {'20m':   [0,1,2]}}
                self.dict_search_structure = {}
                if self.presto_gpu_env == "":
                        self.presto_gpu_env = self.presto_env

        def get_list_datafiles(self, list_datafiles_filename):
                list_datafiles_file = open(list_datafiles_filename, "r" )
                list_datafiles = [line.split()[0] for line in list_datafiles_file if not line.startswith("#") ] #Skip commented line
                list_datafiles_file.close()
                print("get_list_datafiles:: list_datafiles = ", list_datafiles)

                return list_datafiles

        def print_configuration(self):
                print_log("\n ====================打印配置信息：  ====================== \n",color=colors.HEADER)
                # 遍历有序参数列表并打印每个参数及其值（按需修改）
                important_param_list = ['OBSNAME',"SOURCE_NAME",'POOL_NUM','IF_BARY','IF_DDPLAN','DM_MIN','DM_MAX','DM_STEP','PERIOD_TO_SEARCH_MIN','PERIOD_TO_SEARCH_MAX','LIST_SEGMENTS','ACCELSEARCH_LIST_ZMAX','FLAG_JERK_SEARCH','SIFTING_MINIMUM_NUM_DMS','FLAG_FOLD_TIMESERIES','PREPSUBBAND_FLAGS','PREPFOLD_FLAGS','FLAG_SINGLEPULSE_SEARCH']
                for param in important_param_list:
                        print("%-32s %s" % (param, self.dict_survey_configuration[param]))
                print()
                time.sleep(2)

print_program_message('start')
t_start = time.time()

config_filename = "%s.cfg" % (os.path.basename(os.getcwd()))
config = SurveyConfiguration(config_filename)

#######确定重要的变量
obsname = config.obsname       #决定搜寻的文件
if obsname == "":
    print_log(f'obsname为空，请在{config_filename}指定文件',color=colors.ERROR)
    exit()
elif obsname != "":
    # 通过 glob 模块获取所有匹配的观测文件
    config.list_datafiles = [os.path.basename(x) for x in glob.glob(obsname)]
    if len(config.list_datafiles) == 0:
        print_log("错误: 未找到观测文件！请确保文件名正确无误。",color=colors.ERROR)
        exit()
    elif len(config.list_datafiles) >= 1:
        # 如果找到一个或多个文件，检查每个文件是否存在以及文件大小是否为零
        for f in config.list_datafiles:
            if not os.path.exists(f):
                print_log(f"错误: 文件{f}不存在！可能是符号链接损坏。" ,color=colors.ERROR)
                exit()
            elif os.path.getsize(f) == 0:
                print_log(f"错误:文件{f}的大小为 0！" ,color=colors.ERROR)
                exit()
            config.folder_datafiles           = os.path.dirname(os.path.abspath(obsname)) 

config.list_datafiles_abspath = [os.path.join(config.folder_datafiles, x) for x in config.list_datafiles]  #每个文件的绝对路径
config.list_Observations = [Observation(x, config.data_type) for x in config.list_datafiles_abspath]  #生成类属性
config.file_common_birdies = os.path.join(config.root_workdir, "common_birdies.txt")
time.sleep(1)

##重要的变量
obsname = config.obsname       #决定搜寻的文件
if obsname == "":
    print_log(f'obsname为空，请在{config_filename}指定文件',color=colors.ERROR)
    exit()
elif obsname != "":
    # 通过 glob 模块获取所有匹配的观测文件
    config.list_datafiles = [os.path.basename(x) for x in glob.glob(obsname)]
    if len(config.list_datafiles) == 0:
        print_log("错误: 未找到观测文件！请确保文件名正确无误。",color=colors.ERROR)
        exit()
    elif len(config.list_datafiles) >= 1:
        # 如果找到一个或多个文件，检查每个文件是否存在以及文件大小是否为零
        for f in config.list_datafiles:
            if not os.path.exists(f):
                print_log(f"错误: 文件{f}不存在！可能是符号链接损坏。" ,color=colors.ERROR)
                exit()
            elif os.path.getsize(f) == 0:
                print_log(f"错误:文件{f}的大小为 0！" ,color=colors.ERROR)
                exit()
            config.folder_datafiles           = os.path.dirname(os.path.abspath(obsname)) 

config.list_datafiles_abspath = [os.path.join(config.folder_datafiles, x) for x in config.list_datafiles]  #每个文件的绝对路径
config.list_Observations = [Observation(x, config.data_type) for x in config.list_datafiles_abspath]  #生成类属性
config.file_common_birdies = os.path.join(config.root_workdir, "common_birdies.txt")

#重要的变量
config.print_configuration()

workdir = config.root_workdir  #主工作目录
data_path = workdir+'/'+obsname #数据绝对路径

sourcename = config.source_name  #源名，同时所生产数据的唯一标签

n_pool = config.pool_num  #多线程核数

fold_num = config.fold_num

print_log(' ====================注意： ====================== \n',color=colors.HEADER)
print_log('源名为：' + sourcename,masks=sourcename,color=colors.WARNING)
print_log(f'待理数据为：{data_path}',masks=obsname,color=colors.WARNING)
target_type = f'计划一共折叠{fold_num}张图'
fits_or_dats = ''
if config.flag_fold_timeseries == 1:
    fits_or_dats += 'dat'
if config.flag_fold_rawdata == 1:
    fits_or_dats += 'fits' if fits_or_dats == '' else '_fits'
print_log(f'对{fits_or_dats}进行折叠: {target_type} zmax:{config.accelsearch_list_zmax}',masks=[fits_or_dats,str(fold_num),config.accelsearch_list_zmax],color=colors.WARNING)
if config.flag_jerk_search == 1:
    fits_or_dats+= '__jerk'
    print_log(f'进行jerks搜寻：zmax:{config.jerksearch_zmax} wmax:{config.jerksearch_wmax} 叠加谐波数：{config.jerksearch_numharm}')
ifbary = config.ifbary
if ifbary == 1:
    fits_or_dats += '__bary'
    ra = config.ra
    dec = config.dec
    print_log(f'进行质心修正\n 注意： ra = {ra}  dec = {dec} \n',masks=[ra,dec],color=colors.WARNING)

#文件夹
ifok_dir = os.path.join(workdir,'00_IFOK')
#打印文件总信息

sifting.sigma_threshold = config.sifting_sigma_threshold
print_log("main:: SIFTING.sigma_threshold = ", sifting.sigma_threshold, color=colors.BOLD)

LOG_dir = os.path.join(config.root_workdir, "LOG")
makedir(LOG_dir)

if config.if_ddplan == 1:
    print_log("\n ====================DDplan去色散计划：  ====================== \n",color=colors.HEADER)
    list_DDplan_scheme = get_DDplan_scheme(config.list_Observations[0].file_abspath,
                                            LOG_dir,
                                            LOG_dir,
                                            "LOG_diskspace",
                                            config.dm_min,
                                            config.dm_max,
                                            config.dm_coherent_dedispersion,
                                            config.max_simultaneous_dms_per_prepsubband,
                                            config.list_Observations[0].freq_central_MHz,
                                            config.list_Observations[0].bw_MHz,
                                            config.list_Observations[0].nchan,
                                            config.nsubbands,
                                            config.list_Observations[0].t_samp_s)
else:
    print_log("\n ====================自定义去色散计划：  ====================== \n",color=colors.HEADER)
    list_DDplan_scheme = []
    ddpl = config.dm_step
    print(ddpl)
    for ddpl_value in ddpl:
        loodm, highdm, ddm = ddpl_value
        ndms = int((highdm - loodm) // ddm)

        scheme = {
            'loDM': loodm,
            'highDM': highdm,
            'dDM': ddm,
            'downsamp': 1,  
            'num_DMs': ndms
        }
        list_DDplan_scheme.append(scheme)
print_log(list_DDplan_scheme)
time.sleep(2)

# 初始化统计变量
total_files = len(config.list_Observations)
need_copy, skipped, no_space = [], [], []
buffer_valid = False

# 检查快速缓冲目录是否可用
if config.fast_buffer_dir:
    if os.path.exists(config.fast_buffer_dir):
        buffer_free = shutil.disk_usage(config.fast_buffer_dir).free
        buffer_valid = True
    else:
        print_log(f"警告：快速缓冲目录 '{config.fast_buffer_dir}' 不存在！", color=colors.WARNING)
        config.fast_buffer_dir = ""

# 分类统计文件
if buffer_valid:
    for obs in config.list_Observations:
        dst_path = os.path.join(config.fast_buffer_dir, obs.file_nameonly)
        if not os.path.exists(dst_path) or os.path.getsize(dst_path) != os.path.getsize(obs.file_abspath):
            if os.path.getsize(obs.file_abspath) <= buffer_free:
                need_copy.append(obs)
                buffer_free -= os.path.getsize(obs.file_abspath)
            else:
                no_space.append(obs.file_nameonly)
        else:
            skipped.append(obs.file_nameonly)

# 统一打印操作摘要
if buffer_valid:
    print_log("\n==== 快速缓冲目录操作摘要 ====", color=colors.OKGREEN)
    print_log(f"位置: {config.fast_buffer_dir}", color=colors.OKGREEN)
    if need_copy:
        print_log(f"\n需复制 {len(need_copy)} 个文件（共 {total_files} 个）:", color=colors.OKGREEN)
        for obs in need_copy:
            print(f"  - {obs.file_nameonly}")
    if skipped:
        print_log(f"\n已跳过 {len(skipped)} 个文件（已存在且大小匹配）:", color=colors.OKGREEN)
        for name in skipped:
            print(f"  - {name}")
    if no_space:
        print_log(f"\n警告：{len(no_space)} 个文件因空间不足未复制！", color=colors.WARNING)

    # 批量复制文件
    if need_copy:
        print("\n开始批量复制...")
        for obs in need_copy:
            dst_path = shutil.copy(obs.file_abspath, config.fast_buffer_dir)
            obs.file_abspath = dst_path  # 更新路径
            print(f"  已完成: {obs.file_nameonly}")
        print("所有文件复制完成！")
else:
    print_log("未使用快速缓冲目录，处理速度可能受影响。", color=colors.WARNING)

data_len = 0
for i, obs in enumerate(config.list_Observations):
    data_len += obs.T_obs_s
formatted_time = format_execution_time(obs.T_obs_s)
print_log(f" {data_path} ({formatted_time})", color=colors.OKGREEN)


time.sleep(1)

print_log("\n****************检查磁盘空间****************\n",masks='检查磁盘空间',color=colors.HEADER)
num_DMs = 0 
for j in range(len(list_DDplan_scheme)):
        num_DMs = num_DMs + list_DDplan_scheme[j]['num_DMs']
        
flag_enough_disk_space = False
flag_enough_disk_space = check_if_enough_disk_space(config.root_workdir, num_DMs, data_len, config.list_Observations[0].t_samp_s, config.flag_remove_fftfiles)

# 如果磁盘空间不足，打印错误信息并退出程序
if flag_enough_disk_space == False:
        print_log(f"错误：磁盘空间不足！请释放空间或更改工作目录。",color=colors.ERROR)
        print("> 提示：为了最小化磁盘使用，请确保在配置文件中将 FLAG_REMOVE_FFTFILES 和 FLAG_REMOVE_DATFILES_OF_SEGMENTS 保留为默认值 1。")
        exit()
time.sleep(1)

################################################################################
#   IMPORT PARFILES OF KNOWN PULSARS
################################################################################
#psrcat -x -c "name Jname RaJ DecJ p0 dm s1400 type binary survey" > knownPSR1.dat
dir_known_pulsars = os.path.join(config.root_workdir, "known_pulsars")


list_known_pulsars = []
if os.path.exists(dir_known_pulsars):
    list_parfilenames = sorted(glob.glob("%s/*.par" % dir_known_pulsars))
    dict_freqs_to_zap = {}

    for k in range(len(list_parfilenames)):
        # 创建脉冲星对象并添加到已知脉冲星列表
        current_pulsar = Pulsar(list_parfilenames[k])
        list_known_pulsars.append(current_pulsar)

        # 如果脉冲星不是双星系统，计算其频率并记录到字典中
        if not current_pulsar.is_binary:
            current_freq = psr_utils.calc_freq(config.list_Observations[0].Tstart_MJD, current_pulsar.PEPOCH, current_pulsar.F0, current_pulsar.F1, current_pulsar.F2)
            dict_freqs_to_zap[current_pulsar.psr_name] = current_freq

        # 打印已读取的脉冲星信息
        print_log("正在读取 '%s' --> 已将 %s 添加到已知脉冲星列表（%s）" % (os.path.basename(list_parfilenames[k]), current_pulsar.psr_name, current_pulsar.pulsar_type),color=colors.HEADER)

        # 如果配置中要求从傅里叶频谱中消除孤立脉冲星的频率，打印警告信息
        if config.zap_isolated_pulsars_from_ffts == 1:
            print_log("\n警告：我将消除孤立脉冲星的傅里叶频率（最多到 %d 阶谐波），具体如下" % (config.zap_isolated_pulsars_max_harm),color=colors.WARNING)
            for key in sorted(dict_freqs_to_zap.keys()):
                print_log("%s  -->  在观测历元的质心频率: %.14f Hz" % (key, dict_freqs_to_zap[key]))


print_log("\n ====================STEP 1 - RFIFIND====================== \n",color=colors.HEADER)

rfifind_masks_dir = os.path.join(config.root_workdir, "01_RFIFIND")
makedir(rfifind_masks_dir)
basename = 'rfi0.1s'
mask_file_path = f"{rfifind_masks_dir}/rfi0.1s_rfifind.mask"

time.sleep(0.2)
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
flag_mask_present = check_rfifind_outfiles(rfifind_masks_dir, basename)

# 情况 1：掩模不存在且不允许自动生成
if not flag_mask_present and config.flag_step_rfifind == 0:
    print_log(f"\n错误！掩模文件 '{mask_file_path}' 未找到，但 STEP_RFIFIND = 0！", color=colors.ERROR)
    print("请将配置中的 STEP_RFIFIND 设置为 1，或手动将掩模文件放入 '01_RFIFIND' 文件夹后重试。\n")
    exit()

# 情况 2：掩模不存在，但允许自动生成
elif not flag_mask_present and config.flag_step_rfifind == 1:
    LOG_basename = f"01_rfifind_{sourcename}"
    log_abspath = f"{LOG_dir}/LOG_{LOG_basename}.txt"
    
    print_log(f"\n未找到掩模文件，正在使用配置文件 '{config_filename}' 中的参数进行生成。", masks=config_filename, color=colors.BOLD)
    print_log(f"提示: 可使用 'tail -f {log_abspath}' 查看运行进度。", color=colors.BOLD)
    print_log(f"正在为观测源 {sourcename} 创建 rfifind 掩模文件...\n")

    sys.stdout.flush()
    make_rfifind_mask(
        config.list_Observations[i].file_abspath,
        rfifind_masks_dir,
        LOG_dir,
        LOG_basename,
        config.rfifind_time,
        config.rfifind_time_intervals_to_zap,
        config.rfifind_chans_to_zap,
        config.rfifind_flags,
        config.presto_env,
        search_type=sourcename,
        obsname=obsname,
    )

# 情况 3 和 4：掩模已存在
else:
    print_log(f"\n掩模文件 '{mask_file_path}' 已存在，不会重新生成。", color=colors.OKBLUE)
    if config.flag_step_rfifind == 0:
        print_log("警告：STEP_RFIFIND = 0，将跳过该步骤，默认当前掩模文件可用。\n", color=colors.WARNING)

# 如果配置允许 rfifind，评估掩蔽频率通道比例
if config.flag_step_rfifind == 1:
    masked_info_file = os.path.join(rfifind_masks_dir,"rfifind_mask_info.json")
    print("正在检查被掩蔽的频带比例...", end=' ')
    sys.stdout.flush()
    
    if os.path.exists(masked_info_file):
        with open(masked_info_file, 'r') as f:
            info = json.load(f)
            fraction_masked_channels = info.get("fraction_masked_channels", 0)
        print("(已从缓存文件读取)", end=' ')
    else:
        mask = rfifind.rfifind(mask_file_path)
        fraction_masked_channels = len(mask.mask_zap_chans) / mask.nchan
        with open(masked_info_file, 'w') as f:
            json.dump({"fraction_masked_channels": fraction_masked_channels}, f)
    mask_str = f"{fraction_masked_channels * 100:.2f}"
    print_log(f"\nRFIFIND：被掩蔽的频率通道比例：{mask_str}%\n", masks=mask_str, color=colors.OKGREEN)

    if 0.5 < fraction_masked_channels <= 0.95:
        print_log(f"!!! 警告：{mask_str}% 的频带被掩蔽，比例偏高 !!!", color=colors.WARNING)
        print("!!! 请考虑调整 RFIFIND 参数（如 RFIFIND_FREQSIG）以减少掩蔽。")
        time.sleep(1)
    elif fraction_masked_channels > 0.95:
        print_log(f"!!! 错误：{mask_str}% 的频带被掩蔽，过高 !!!", color=colors.ERROR)
        print("!!! 请调整配置中的 RFIFIND 参数，使掩蔽比例低于 95%。")
        exit()

# 如果存在 weights 文件，提取并记录被忽略通道
weights_file = mask_file_path.replace(".mask", ".weights")
if os.path.exists(weights_file):
    array_weights = np.loadtxt(weights_file, unpack=True, usecols=(0, 1,), skiprows=1)
    ignored_indices = np.where(array_weights[1] == 0)[0]
    config.ignorechan_list = ",".join(map(str, ignored_indices))
    config.nchan_ignored = len(ignored_indices)

    total_chans = config.list_Observations[i].nchan
    ignored_percent = 100 * config.nchan_ignored / total_chans
    print(f"\n\n已找到 WEIGHTS 文件 '{os.path.basename(weights_file)}'。")
    print(f"共忽略了 {config.nchan_ignored} 个通道（共 {total_chans}，占 {ignored_percent:.2f}%）")
    print(f"被忽略的通道索引： {config.ignorechan_list}")

time.sleep(1)

print_log("\n ====================STEP 2 - BIRDIES AND ZAPLIST   ====================== \n",color=colors.HEADER)

print("STEP_ZAPLIST = %s" % (config.flag_step_zaplist))

sourcename_mask = sourcename+'_'+config.search_label
ifok_dir02 = os.path.join(ifok_dir,'02_BIRDIES')
makedir(ifok_dir02)
LOG_dir02 = os.path.join(LOG_dir,'02_BIRDIES')
makedir(LOG_dir02)

dir_birdies = os.path.join(workdir, "02_BIRDIES")
if config.flag_step_zaplist == 1:
        print_log(f"\n 02a) 使用掩模为{obsname}创建一个 0-DM 质心时间序列。 \n",color=colors.HEADER)
        makedir(dir_birdies)

        time.sleep(0.1)

        sys.stdout.flush()
        LOG_basename = "02a_prepdata_full" 
        log_path = os.path.join(LOG_dir02, f"LOG_{LOG_basename}.txt")
        prepdata(data_path ,sourcename_mask,dir_birdies,ifok_dir02, log_path,0,0,config.ignorechan_list,mask_file_path,1,"topocentric",config.prepdata_flags,config.presto_env) #barycentric为不进行质心修正
        sys.stdout.flush()
                
        print_log("\n 02b) 对所有文件进行傅里叶变换。 \n",color=colors.HEADER)     
        DM0_datfiles = f"{dir_birdies}/{sourcename_mask}_DM00.00.dat"    # 收集 02_BIRDIES_FOLDERS 中的 *.dat 文件
        DM0_datfiles_path = os.path.join(dir_birdies,DM0_datfiles)
        time.sleep(0.1)
        LOG_basename = "02b_realfft_full" 
        log_path = os.path.join(LOG_dir02, f"LOG_{LOG_basename}.txt")
        realfft(DM0_datfiles_path,sourcename_mask,dir_birdies,ifok_dir02,log_path,config.realfft_flags,config.presto_env)

        print_log("\n 02c) 去除红噪声。 \n",color=colors.HEADER)  
        DM0_fftfiles = f"{dir_birdies}/{sourcename_mask}_DM00.00.fft"
        DM0_fftfiles_path = os.path.join(dir_birdies,DM0_fftfiles)
        time.sleep(0.1)
        LOG_basename = "02c_rednoise_full" 
        log_path = os.path.join(LOG_dir02, f"LOG_{LOG_basename}.txt")
        rednoise(DM0_fftfiles_path,sourcename_mask,dir_birdies,ifok_dir02,log_path,config.rednoise_flags,config.presto_env)

        print_log("\n 02d) 加速搜索和创建 zaplist。 \n",color=colors.HEADER)
        DM0_fft_red_files = f"{dir_birdies}/{sourcename_mask}_DM00.00.fft"
        DM0_fft_red_files_path = os.path.join(dir_birdies,DM0_fft_red_files)
        time.sleep(0.1)
        LOG_basename = "02d_makezaplist_full" 
        log_path = os.path.join(LOG_dir02, f"LOG_{LOG_basename}.txt")
        zaplist_filename = make_zaplist(DM0_fft_red_files, sourcename_mask,dir_birdies,ifok_dir02,log_path,config.file_common_birdies,2,config.accelsearch_flags,config.presto_env)


        if config.zap_isolated_pulsars_from_ffts == 1:
                fourier_bin_size =  1./config.list_Observations[0].T_obs_s  # 计算傅里叶变换的频率分辨率
                zaplist_file = open(zaplist_filename, 'a')  # 打开 zaplist 文件以追加内容

                zaplist_file.write("########################################\n")
                zaplist_file.write("#              已知脉冲星              #\n")
                zaplist_file.write("########################################\n")
                for psr in sorted(dict_freqs_to_zap.keys()):  # 遍历已知脉冲星的频率字典
                        zaplist_file.write("# 脉冲星 %s \n" % (psr))
                        for i_harm in range(1, config.zap_isolated_pulsars_max_harm+1):  # 添加谐波频率到 zaplist
                                zaplist_file.write("B%21.14f   %19.17f\n" % (dict_freqs_to_zap[psr]*i_harm, fourier_bin_size*i_harm))
                zaplist_file.close()  # 关闭文件

#指定消色散方案并生成PNG文件夹
if config.if_ddplan == 1:
     
    basename_dd_pl = 'dd'
    log_dd_pl = '03_ddsubbands'
else:   
    basename_dd_pl = 'pl'
    log_dd_pl = '03_subbands'

basename_only = sourcename_mask+'_'+basename_dd_pl+'_'+fits_or_dats
png_dir = os.path.join(workdir,'06_PNG',basename_only)
makedir(png_dir)


dir_dedispersion = os.path.join(config.root_workdir, log_dd_pl)  
print_log("\n ==========STEP 3 - DEDISPERSION, DE-REDDENING AND PERIODICITY SEARCH========== \n",color=colors.HEADER)

LOG_basename = "03_prepsubband_and_search_FFT_%s" % (config.list_Observations[i].file_nameonly)
print("3) 去色散、去红噪声和周期性搜索：", end=' '); sys.stdout.flush()
makedir(dir_dedispersion)  # 创建去色散目录


if config.if_ddplan == 1:
    print_log("\n ====================DDplan去色散计划：  ====================== \n",color=colors.HEADER)
    list_DDplan_scheme = get_DDplan_scheme(config.list_Observations[i].file_abspath,
                                            png_dir,
                                            LOG_dir,
                                            LOG_basename,
                                            config.dm_min,
                                            config.dm_max,
                                            config.dm_coherent_dedispersion,
                                            config.max_simultaneous_dms_per_prepsubband,
                                            config.list_Observations[i].freq_central_MHz,
                                            config.list_Observations[i].bw_MHz,
                                            config.list_Observations[i].nchan,
                                            config.nsubbands,
                                            config.list_Observations[i].t_samp_s)
else:
    print_log("\n ====================自定义去色散计划：  ====================== \n",color=colors.HEADER)
    list_DDplan_scheme = []
    ddpl = config.dm_step
    print(ddpl)
    for ddpl_value in ddpl:
        loodm, highdm, ddm = ddpl_value
        ndms = int((highdm - loodm) // ddm)

        scheme = {
            'loDM': loodm,
            'highDM': highdm,
            'dDM': ddm,
            'downsamp': 1,  
            'num_DMs': ndms
        }
        list_DDplan_scheme.append(scheme)

# 遍历每个方案并生成 dm_list
all_dm_ranges_str = []
for scheme in list_DDplan_scheme:
    lowDM = scheme['loDM']
    highDM = scheme['highDM']
    dDM = scheme['dDM']
    dm_range = np.arange(lowDM, highDM, dDM)
    
    dm_range_str = [f"{dm:.2f}" for dm in dm_range]
    all_dm_ranges_str.extend(dm_range_str)
dm_list = all_dm_ranges_str
N_schemes = len(list_DDplan_scheme)

print("3) 去色散：正在创建工作目录 '%s'..." % (dir_dedispersion), end=' '); sys.stdout.flush()
makedir(dir_dedispersion)
print("完成！"); sys.stdout.flush()

ps2png(os.path.join(png_dir,'*ps'))

print_log("\n ==========STEP 3 -1  PREPSUBBAND 消色散========= \n",color=colors.HEADER)
LOG_dir03 = os.path.join(LOG_dir,log_dd_pl)
makedir(LOG_dir03)

zapfile = "%s/%s_DM00.00.zaplist" % (dir_birdies, sourcename_mask)
dict_flag_steps = {'flag_step_dedisperse': config.flag_step_dedisperse, 'flag_step_realfft': config.flag_step_realfft, 'flag_step_periodicity_search': config.flag_step_periodicity_search}

cpu_count()
ignorechan_list = config.ignorechan_list
nchan = config.list_Observations[0].nchan
subbands = config.nsubbands
num_simultaneous_prepsubbands = config.num_simultaneous_prepsubbands
other_flags_prepsubband = config.prepsubband_flags
presto_env_prepsubband =  config.presto_env

if N_schemes < num_simultaneous_prepsubbands:
        print(f'非并行消色散')
        dedisperse(data_path,basename_dd_pl,sourcename_mask, dir_dedispersion, LOG_dir03, ignorechan_list, mask_file_path, list_DDplan_scheme, nchan, subbands, other_flags_prepsubband, presto_env_prepsubband)

else:   
# if 1:
        print_log(f'并行消色散:核数{num_simultaneous_prepsubbands}/{cpu_count()}',masks=str(num_simultaneous_prepsubbands),color=colors.HEADER)
        prepsubbandcmd_all,ifok_all,log_all=dedisperse2cmd(data_path,basename_dd_pl,sourcename_mask, dir_dedispersion, LOG_dir03, ignorechan_list, mask_file_path, list_DDplan_scheme, nchan, subbands, other_flags_prepsubband, presto_env_prepsubband)
        pool(num_simultaneous_prepsubbands,'prepsubband',prepsubbandcmd_all,ifok_all,log_all,work_dir = dir_dedispersion)
        
print_log("\n ==========STEP 3 -2  prepdata预质心修正 ========= \n",color=colors.HEADER)
if ifbary == 1:
    print_log(f'使用ra = {ra} ,dec = {dec}进行质心修正')
    bary_dir = os.path.join(config.root_workdir, "03_barydata") 
    makedir(bary_dir)
    ifok_dir03b = os.path.join(ifok_dir,'03_barydata')
    makedir(ifok_dir03b)
    LOG_dir03b = os.path.join(LOG_dir,'03_barydata')
    makedir(LOG_dir03b)

    dat_names = sorted([x for x in glob.glob(f"{dir_dedispersion}/*DM*.*.dat")]) 
    inf_names = sorted([os.path.abspath(os.path.join(dir_dedispersion, file)) for file in os.listdir(dir_dedispersion) if file.endswith('.inf') and not file.endswith('_red.inf')])

    print_log('''\n ==================== 修改FAST的inf文件错误  ====================== \n''',color=colors.HEADER)
    for inf in inf_names:
        with open(inf, 'r') as file:
            lines = file.readlines()
            lines = [line for line in lines if 'On/Off bin pair' not in line]

        # 寻找并替换 J2000 Right Ascension 和 J2000 Declination 的行
        for i in range(len(lines)):
            if 'J2000 Right Ascension (hh:mm:ss.ssss)' in lines[i]:
                ra_index = i
                lines[i] = ' J2000 Right Ascension (hh:mm:ss.ssss)  =  ' + ra + "\n"    # 修改赤经
            elif 'J2000 Declination     (dd:mm:ss.ssss)' in lines[i]:
                dec_index = i
                lines[i] = ' J2000 Declination     (dd:mm:ss.ssss)  =  ' + dec + '\n'   # 修改赤纬
            elif 'Any breaks in the data? (1 yes, 0 no)' in lines[i]:
                lines[i] = 'Any breaks in the data? (1 yes, 0 no)  =  0 '+'\n'

        # 将修改后的内容写回文件
        with open(inf, 'w') as file:
            file.writelines(lines)
    print_log('成功！')
    print_log('''\n ==================== ra,dec修正完毕  ====================== \n''',color=colors.HEADER)

    #添加mask_file_path会报错
    prepdata_cmd_list,ifok_list,log_list = prepdata2bary(dat_names,sourcename_mask, bary_dir,ifok_dir03b, LOG_dir03b, Nsamples=0, ignorechan_list="",mask='', downsample_factor=1, other_flags=config.prepdata_flags,presto_env=os.environ['PRESTO'])
    
    print_log('''\n ==================== 3 -3  prepdata质心修正  ====================== \n''',color=colors.OKGREEN) 
    print_log(f'并行质心修正:核数{n_pool}/{cpu_count()}',masks=str(n_pool),color=colors.HEADER)
    pool(n_pool,'prepdata-bary',prepdata_cmd_list,ifok_list,log_list,work_dir = bary_dir)

    dir_dedispersion = bary_dir

else:
    print_log('''\n ==================== 基于给予的参数将跳过质心修正，速度加快  ====================== \n''',color=colors.HEADER)


    
list_zmax = config.accelsearch_list_zmax
numharm = config.accelsearch_numharm

flag_jerk_search = config.flag_jerk_search
jerksearch_zmax = config.jerksearch_zmax
jerksearch_wmax = config.jerksearch_wmax
jerksearch_numharm = config.jerksearch_numharm

if dict_flag_steps['flag_step_realfft'] == 1:

    print_log('''\n ==================== 傅里叶变换  ====================== \n''',color=colors.HEADER) 


    # print("\033[1m >> 提示：\033[0m 可以通过以下命令实时查看周期性搜索的日志：\033[1mtail -f %s\033[0m" % (log_abspath))

    dat_names = sorted([os.path.abspath(os.path.join(dir_dedispersion, file)) for file in os.listdir(dir_dedispersion) if file.endswith('.dat')])
    fft_files = [file.replace(".dat", ".fft") for file in dat_names]

    # DM_trial_was_searched = check_if_DM_trial_was_searched(dat_names, list_zmax, flag_jerk_search, jerksearch_zmax, jerksearch_wmax)


    ifok_dir04 = os.path.join(ifok_dir,'04_FFT')
    makedir(ifok_dir04)
    LOG_dir04 = os.path.join(LOG_dir,'04_FFT')
    makedir(LOG_dir04)
    realfft_cmd_list,ifok_list,log_list = realfft2cmd(dat_names,sourcename_mask, dir_dedispersion,ifok_dir04, LOG_dir04, other_flags=config.realfft_flags,presto_env=os.environ['PRESTO'])
    
    print_log(f'并行质心修正:核数{n_pool}/{cpu_count()}',masks=str(n_pool),color=colors.HEADER)
    pool(n_pool,'realfft',realfft_cmd_list,ifok_list,log_list,work_dir = dir_dedispersion)
    
    print_log('''\n ==================== 去除红噪声  ====================== \n''',color=colors.HEADER) 

    fft_names = sorted([os.path.abspath(os.path.join(dir_dedispersion, file)) for file in os.listdir(dir_dedispersion) if file.endswith('.fft') and not file.endswith('_red.fft')])
    inf_files = [file.replace(".fft", ".inf") for file in fft_names]

    ifok_dir04 = os.path.join(ifok_dir,'04_RED')
    makedir(ifok_dir04)
    LOG_dir04 = os.path.join(LOG_dir,'04_RED')
    makedir(LOG_dir04)
    red_cmd_list,ifok_list,log_list = rednoise2cmd(fft_names,sourcename_mask, dir_dedispersion,ifok_dir04, LOG_dir04, other_flags='',presto_env=os.environ['PRESTO'])

    print_log(f'并行质心修正:核数{n_pool}/{cpu_count()}',masks=str(n_pool),color=colors.HEADER)
    pool(n_pool,'rednoise',red_cmd_list,ifok_list,log_list,work_dir = dir_dedispersion)

    fft_red_names = sorted([os.path.abspath(os.path.join(dir_dedispersion, file)) for file in os.listdir(dir_dedispersion) if file.endswith('_red.fft')])
    inf_files = [file.replace("_red.fft", "_red.inf") for file in fft_red_names]
    for fftfile_rednoise_abspath in fft_red_names:
        os.rename(fftfile_rednoise_abspath, fftfile_rednoise_abspath.replace("_red.", "."))
    for inf_rednoise_abspath in inf_files:
        os.rename(inf_rednoise_abspath, inf_rednoise_abspath.replace("_red.", "."))

    fft_red_names = sorted([os.path.abspath(os.path.join(dir_dedispersion, file)) for file in os.listdir(dir_dedispersion) if file.endswith('_red.fft')])
    if len(fft_red_names) == 0:
        print_log(f'红噪声文件重命名成功',color=colors.OKGREEN)
    else:
        print_log(f'红噪声文件重命名失败,请检查数据',color=colors.ERROR)

    print_log('''\n ==================== 正在将消噪文件应用到FFT  ====================== \n''',color=colors.HEADER) 

    fft_names = sorted([os.path.abspath(os.path.join(dir_dedispersion, file)) for file in os.listdir(dir_dedispersion) if file.endswith('.fft') and not file.endswith('_red.fft')])

    ifok_dir04 = os.path.join(ifok_dir,'04_ZAP')
    makedir(ifok_dir04)
    LOG_dir04 = os.path.join(LOG_dir,'04_ZAP')
    makedir(LOG_dir04)
    zap_cmd_list,ifok_list,log_list = zapbirds2cmd(fft_names, zapfile,ifok_dir04, LOG_dir04)
    
    print_log(f'并行消除ODM噪声:核数{n_pool}/{cpu_count()}',masks=str(n_pool),color=colors.HEADER)
    pool(n_pool,'zap',zap_cmd_list,ifok_list,log_list,work_dir = dir_dedispersion)

else:
    print_log('''\n =============STEP_REALFFT = 0，跳过 realfft、rednoise、zapbirds... ================ \n''',color=colors.HEADER) 

#周期搜寻(耗时最久的部分)
flag_use_cuda = config.flag_use_cuda
list_cuda_ids = config.list_cuda_ids
other_flags_accelsearch = config.accelsearch_flags

presto_env_accelsearch_zmax_0 = os.environ['PRESTO']
presto_env_accelsearch_zmax_any = os.environ['PRESTO']

dict_env_zmax_0 = {'PRESTO': presto_env_accelsearch_zmax_0, 'PATH': f"{presto_env_accelsearch_zmax_0}/bin:{os.environ['PATH']}", 'LD_LIBRARY_PATH': f"{presto_env_accelsearch_zmax_0}/lib:{os.environ['LD_LIBRARY_PATH']}"}
dict_env_zmax_any = {'PRESTO': presto_env_accelsearch_zmax_any, 'PATH': f"{presto_env_accelsearch_zmax_any}/bin:{os.environ['PATH']}", 'LD_LIBRARY_PATH': f"{presto_env_accelsearch_zmax_any}/lib:{os.environ['LD_LIBRARY_PATH']}"}

if dict_flag_steps['flag_step_periodicity_search'] == 1:  

    ifok_dir05 = os.path.join(ifok_dir,'05_search')
    makedir(ifok_dir05)
    LOG_dir05 = os.path.join(LOG_dir,'05_search')
    makedir(LOG_dir05)
    print_log(f'''\n ==================== 加速搜寻：zmax = {list_zmax}  ====================== \n''',color=colors.HEADER)                                                     

    dat_names = sorted([os.path.abspath(os.path.join(dir_dedispersion, file)) for file in os.listdir(dir_dedispersion) if file.endswith('.dat')])
    fft_files = [file.replace(".dat", ".fft") for file in dat_names]
 
    for z in list_zmax:
            print('f检验zmax={zmax}')

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
            search_cmd_list,ifok_list,log_list = accelsearch2cmd(fft_files,ifok_dir05, LOG_dir05, numharm=numharm, zmax=z, other_flags=accelsearch_flags)

            print_log(f'并行周期搜寻:核数{n_pool}/{cpu_count()}',masks=str(n_pool),color=colors.HEADER)
            pool(n_pool,'zap',search_cmd_list,ifok_list,log_list,work_dir = dir_dedispersion)

            for fft_path in fft_files:
                if not check_accelsearch_result(fft_path, int(z),verbosity_level=0):  #打印详细信息verbosity_level=2
                    inffile_empty = fft_path.replace(".fft", "_ACCEL_%d_empty" % (z))
                    with open(inffile_empty, "w") as file_empty:
                        print_log("警告：accelsearch 没有产生任何候选结果！写入文件 %s 以标记此情况..." % (inffile_empty),color=colors.WARNING,mode='p')
                        file_empty.write("ACCELSEARCH DID NOT PRODUCE ANY CANDIDATES!")


oksift = os.path.join(workdir,'ok-sifting')
if config.flag_step_sifting == 1 :
    print_log('''\n ==================== Setp5:ddsifting candidates ====================== \n''',color=colors.HEADER) 
    dir_sifting = os.path.join(config.root_workdir, "04_SIFTING")
    makedir(dir_sifting)

    flag_remove_duplicates = config.sifting_flag_remove_duplicates
    flag_DM_problems =config.sifting_flag_remove_dm_problems
    flag_remove_harmonics = config.sifting_flag_remove_harmonics
    minimum_numDMs_where_detected = config.sifting_minimum_DM
    period_to_search_min_s = config.period_to_search_min
    period_to_search_max_s = config.period_to_search_max

    if not os.path.isfile(oksift):
        # 调用 sift_candidates 函数
        cands = sift_candidates(
                    work_dir=dir_sifting,
                    sourcename=sourcename_mask,
                    log_dir=LOG_dir,
                    dedispersion_dir=dir_dedispersion,
                    list_zmax=list_zmax,
                    jerksearch_zmax=jerksearch_zmax,
                    jerksearch_wmax=jerksearch_wmax,
                    flag_remove_duplicates=flag_remove_duplicates,
                    flag_DM_problems=flag_DM_problems,
                    flag_remove_harmonics=flag_remove_harmonics,
                    minimum_numDMs_where_detected=minimum_numDMs_where_detected,
                    minimum_acceptable_DM=2.0,  # 保持默认值 2.0
                    period_to_search_min_s=period_to_search_min_s,
                    period_to_search_max_s=period_to_search_max_s
        )

        candnumber = len(cands)
        print_log('待折叠候选体个数为：',len(cands))

        best_cands_filename = "%s/best_candidates_%s.siftedcands" % (dir_sifting, sourcename_mask)
        with open(best_cands_filename, "r") as f:
            lines = f.readlines()
            sifting = []
            for line in lines:
                if line.startswith("#"):
                    print_log(line)
                    sifting.append(line)
                if line.startswith(sourcename) or line.startswith('bary'):
                    print_log(line)
                    sifting.append(line) 
        with open(dir_sifting+'/cand_sifting.txt', "w") as f:
            f.write('#待折叠候选体个数为：'+str(candnumber)+'\n')
            for line in sifting:
                f.write(line)
        os.system('touch '+oksift) 
    else:
        print_log(f'请注意!将跳过sifting candidates，如果想重新生成候选，请移除ok-sifting',color=colors.WARNING)


#按信噪比进行排序
input_file_path = os.path.join(dir_sifting,'cand_sifting.txt')  # 请替换为实际的输入文件路径
SNR_file = os.path.join(dir_sifting,'cand_sift_SNR.txt') 

with open(input_file_path, 'r') as infile:
    # 读取所有行
    lines = infile.readlines()
    cand_n = len(lines)
    print_log(f'#待折叠候选体个数为:{cand_n}',masks=str(cand_n),color=colors.OKBLUE)
    # 解析数据，并跳过注释行
    header = lines[1].split()  # 获取列名
    header_str = "{:<2}{:<38} {:<10} {:<10} {:<10} {:<5} {:<10} {:<10} {:<10} {:<15} {:<10} {:<10}".format(*lines[1].split())   # 获取列名
    #print(header_str)
    data = [line.split() for line in lines[2:] if not line.startswith('#')]

    # 将SNR作为浮点数添加到数据中(由于#存在，使用DM代码SNR)
    for entry in data:
        entry[header.index('DM')] = float(entry[header.index('DM')])

    # 按SNR列排序数据
    sorted_data = sorted(data, key=lambda x: x[header.index('DM')], reverse=True)

# 将排序后的数据写入新文件
with open(SNR_file, 'w') as outfile:
    # 写入列名
    outfile.write((header_str) + '\n')
    # 写入数据
    for entry in sorted_data:
        # 将浮点数转换为字符串
        formatted_line = "{:<40} {:<10} {:<10} {:<10} {:<5} {:<10} {:<10} {:<10} {:<15} {:<10} {:<10}".format(*entry)
        #entry_as_str = [str(item) for item in entry]
        #outfile.write('\t'.join(entry_as_str) + '\n')
        outfile.write(formatted_line + '\n')
print_log("排序后的数据已保存到", SNR_file)

   
print_log('''\n ==================== Setp6:folding candidates=  ====================== \n''',color=colors.HEADER) 

# print("\033[1m >> 提示:\033[0m 使用 '\033[1mtail -f %s/LOG_%s.txt\033[0m' 查看折叠进度" % (LOG_dir, LOG_basename))
dir_folding = os.path.join(config.root_workdir, "05_FOLDING")
makedir(dir_folding)
LOG_dir06 = os.path.join(LOG_dir,'06_fold')
makedir(LOG_dir06)

cmd_prepfold_list = []
c1 =[]
c2 =[]
ifok_prepfold_list = []
p1 = []
p2 =[]
log_prepfold_list = []
l1 = []
l2 = []
with open(SNR_file, "r") as f:
    lines = f.readlines()
    n = 0
    for line in lines:
        if line.startswith(sourcename) or line.startswith('bary'):
            parts = line.split()
            candfile = parts[0]
            cand_file = candfile.split(":")[0]
            candnum = int(candfile.split(":")[-1])
            dm = float(parts[1])
            dm ="{:.2f}".format(dm)
            snr = float(parts[2])
            sigma = float(parts[3])
            num_harm = int(parts[4])
            ipow = float(parts[5])
            cpow = float(parts[6])
            p_ms = float(parts[7])
            r = float(parts[8])
            z = float(parts[9])
            num_hits = int(parts[10][1:-1])
            n += 1
            outname ='A'+str(n)+'_'+sourcename_mask
            # print(f'读取第{i+1}个数据')

            cand_zmax = cand_file.split("ACCEL_")[-1].split("_JERK")[0]
            if "JERK_" in os.path.basename(cand_file):
                cand_wmax = cand_file.split("JERK_")[-1]
                str_zmax_wmax = f"z{cand_zmax}_w{cand_wmax}"
            else:
                str_zmax_wmax = f"z{cand_zmax}"

            if ignorechan_list != "":
                flag_ignorechan = f"-ignorechan {ignorechan_list} "
            else:
                flag_ignorechan = ""

            other_flags_prepfold = config.prepfold_flags
            if '-nsub' not in other_flags_prepfold:
                other_flags_prepfold = f"{other_flags_prepfold} -nsub {nchan}"

            if config.flag_fold_timeseries == 1:
                file_script_fold_name = "script_fold_ts.txt"
                file_script_fold_abspath = f"{dir_folding}/{file_script_fold_name}"
                
                file_to_fold = os.path.join(dir_dedispersion, cand_file.split("_ACCEL")[0] + ".dat")
                cmd_prepfold1 = f"prepfold -nosearch {other_flags_prepfold} -noxwin -dm {dm} -accelcand {candnum} -accelfile {dir_dedispersion}/{cand_file}.cand -o {outname}_ts_DM{dm}_{str_zmax_wmax}  {file_to_fold}" #没有添加mask
                #A9_AQLX-1_raw_DM11.50_z0_ACCEL_Cand_4.pfd.png
                png1 = os.path.join(png_dir,f"{outname}_ts_DM{dm}_{str_zmax_wmax}_ACCEL_Cand_{candnum}.pfd.png")
                log1 = os.path.join(LOG_dir06,f'fold_ts-{dm}.ifok')

                c1.append(cmd_prepfold1)
                write2file(cmd_prepfold1,file_script_fold_abspath)
                p1.append(png1)
                l1.append(log1)

            if config.flag_fold_rawdata == 1:
                file_script_fold_name = "script_fold_raw.txt"
                file_script_fold_abspath = f"{png_dir}/{file_script_fold_name}"

                file_to_fold = data_path
                cmd_prepfold2 = f"prepfold -nosearch {other_flags_prepfold} -noxwin -dm {dm} -accelcand {candnum} -accelfile {dir_dedispersion}/{cand_file}.cand  {flag_ignorechan} -mask {mask_file_path} -o {outname}_raw_DM{dm}_{str_zmax_wmax}    {file_to_fold}"
  
                png2 = os.path.join(png_dir,f"{outname}_raw_DM{dm}_{str_zmax_wmax}_ACCEL_Cand_{candnum}.pfd.png")
                log2 = os.path.join(LOG_dir06,f'fold_raw-{dm}.ifok')
                
                c2.append(cmd_prepfold2) 
                file_script_fold_abspath = f"{png_dir}/{file_script_fold_name}"
                write2file(cmd_prepfold2,file_script_fold_abspath)
                p2.append(png2)
                l2.append(log2)               
            
        cmd_prepfold_list = c1 + c2
        ifok_prepfold_list = p1+p2
        log_prepfold_list = l1+l2


def fold_task(cmd, ifok,logfile, work_dir,png_dir):
    whitelist = []
    filename = os.path.basename(ifok)
    png_name = f'{filename[:-4]}.png'
    ps_path = os.path.join(work_dir,f'{filename[:-4]}.ps')
    """子任务执行函数"""
    run_cmd(cmd, ifok = ifok, work_dir=work_dir,log_file=logfile,mode='both')  #根据ifok判断是否运行cmd
    ps2png(png_name)
    handle_files(work_dir, png_dir, 'copy',ps_path )

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


if config.flag_step_folding == 1:
    fold_num_pl = min(len(cmd_prepfold_list),fold_num)
    
    # False_list = [False] * fold_num_pl
    # cmd_prepfold_list1 = cmd_prepfold_list[:fold_num_pl]
    start_time = time.time()
    pool_fold(n_pool,'fold',cmd_prepfold_list[:fold_num_pl],ifok_prepfold_list[:fold_num_pl],log_prepfold_list[:fold_num_pl],work_dir = dir_folding,png_dir=png_dir)

    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = format_execution_time(execution_time)
    print_log( "全部折叠运行时间为： " + execution_time_str + "\n")
    time.sleep(2)   


t_end = time.time()
execution_time = t_end- t_start
execution_time_str = format_execution_time(execution_time)
print_log( "程序完整运行运行时间为： " + execution_time_str + "\n")


print_log('尝试打包文件',color=colors.HEADER)

# 获取 A1 到 A30 开头的 png 文件（使用 glob 和列表推导）
all_png_file = []
for i in range(1, 31):
    pattern = os.path.join(png_dir, f"A{i}*.png")
    matched_files = glob.glob(pattern)
    all_png_file.extend(matched_files)

file_paths = all_png_file[:30]
file_paths.append(SNR_file)

# 构造邮件正文
email_content = '该程序运行成功\n'
email_content += f'源名：{sourcename_mask}\n'
email_content += f'png文件路径：{png_dir}\n'

# 发送邮件
# send_email(email_content, file_paths)

print_program_message('end')

