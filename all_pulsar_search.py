#!/usr/bin/env python3
"""
Created on 2023.11.11
@author: Long Peng
@web page: https://www.plxray.cn/
qq:2107053791

need: 
主程序
"""
import os,sys
import numpy as np
from psr_fuc import *
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

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    BOLD = '\033[1m'
    ENDCOLOR = '\033[0m'

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


class Inffile(object):
    def __init__(self, inffilename):
        # 打开并读取信息文件（.inf 文件）
        inffile = open(inffilename, "r")
        for line in inffile:
            # 逐行解析文件内容，提取关键信息
            if "Data file name without suffix" in line:  
                self.datafilebasename = line.split("=")[-1].strip()
            elif "Telescope used" in line:  # 使用的望远镜
                self.telescope = line.split("=")[-1].strip()
            elif "Instrument used" in line:  # 使用的仪器
                self.instrument = line.split("=")[-1].strip()
            elif "Object being observed" in line:  # 观测目标
                self.source = line.split("=")[-1].strip()
            elif "J2000 Right Ascension" in line:  # J2000赤经
                self.RAJ = line.split("=")[-1].strip()
            elif "J2000 Declination" in line:  # J2000赤纬
                self.DECJ = line.split("=")[-1].strip()
            elif "Data observed by" in line:  # 观测者
                self.observer = line.split("=")[-1].strip()
            elif "Epoch of observation" in line:  # 观测的历元（MJD）
                self.start_MJD = np.float128(line.split("=")[-1].strip())
            elif "Barycentered?" in line:  # 是否已进行质心化处理
                self.barycentered = int(line.split("=")[-1].strip())
            elif "Number of bins in the time series" in line:  # 时间序列的采样点数
                self.nsamples = int(line.split("=")[-1].strip())
            elif "Width of each time series bin" in line:  # 每个采样点的时间宽度（秒）
                self.tsamp_s = np.float128(line.split("=")[-1].strip())
            elif "Any breaks in the data?" in line:  # 数据是否存在间断
                self.breaks_in_data = int(line.split("=")[-1].strip())
            elif "Type of observation" in line:  # 观测类型
                self.obstype = line.split("=")[-1].strip()
            elif "Beam diameter" in line:  # 波束直径
                self.beamdiameter = np.float128(line.split("=")[-1].strip())
            elif "Dispersion measure" in line:  # 色散
                self.DM = np.float128(line.split("=")[-1].strip())
            elif "Central freq of low channel" in line:  # 最低频段的中心频率
                self.freq_ch1 = np.float128(line.split("=")[-1].strip())
            elif "Total bandwidth" in line:  # 总带宽
                self.bw = np.float128(line.split("=")[-1].strip())
            elif "Number of channels" in line:  # 频道数量
                self.nchan = int(line.split("=")[-1].strip())
            elif "Channel bandwidth" in line:  # 每个频道的带宽
                self.bw_chan = np.float128(line.split("=")[-1].strip())
            elif "Data analyzed by" in line:  # 分析数据的人
                self.analyzer = line.split("=")[-1].strip()
        inffile.close()  # 关闭文件

class Observation(object):
    def __init__(self, file_name, data_type="psrfits", verbosity_level=1):
        # 获取文件的绝对路径、文件名和扩展名
        self.file_abspath = os.path.abspath(file_name)
        self.file_nameonly = self.file_abspath.split("/")[-1]
        self.file_basename, self.file_extension = os.path.splitext(self.file_nameonly)
        self.file_buffer_copy = ""  # 初始化文件缓冲区副本

        if data_type == "filterbank":  
            try:
                object_file = filterbank.FilterbankFile(self.file_abspath)  

                # 提取文件的关键信息
                self.N_samples = object_file.nspec  # 总采样点数
                self.t_samp_s = object_file.dt  # 采样时间间隔（秒）
                self.T_obs_s = self.N_samples * self.t_samp_s  # 观测总时长（秒）
                self.nbits = object_file.header['nbits']  # 数据位宽
                self.nchan = object_file.nchan  # 频道数量
                self.chanbw_MHz = object_file.header['foff']  # 每个频道的带宽（MHz）
                self.bw_MHz = self.nchan * self.chanbw_MHz  # 总带宽（MHz）
                self.freq_central_MHz = object_file.header['fch1'] + object_file.header['foff'] * 0.5 * object_file.nchan  # 中心频率（MHz）
                self.freq_high_MHz = np.amax(object_file.freqs)  # 最高频率（MHz）
                self.freq_low_MHz = np.amin(object_file.freqs)  # 最低频率（MHz）
                self.MJD_int = int(object_file.header['tstart'])  # 起始MJD的整数部分
                self.Tstart_MJD = object_file.header['tstart']  # 起始MJD时间

                self.source_name = object_file.header['source_name'].strip()  # 观测源名称

            except ValueError:  # 如果读取失败，尝试使用其他方法
                print("警告：读取时出现值错误！可能是滤波银行数据不是8位、16位或32位。尝试使用PRESTO的'readfile'获取必要信息..."),print()

                try:
                    # 使用PRESTO的'readfile'工具提取信息
                    # a = np.float64(readfile_with_str(f"readfile {self.file_abspath}", "grep 'Spectra per file'").split()[-1])
                    # print(a)
                    self.N_samples = np.float64(readfile_with_str(f"readfile {self.file_abspath}", "grep 'Spectra per file'").split("=")[-1].strip())
                    self.t_samp_s = 1.0e-6 * float(readfile_with_str(f"readfile {file_name}", "grep 'Sample time (us)'").split("=")[-1].strip())
                    self.T_obs_s = self.N_samples * self.t_samp_s
                    self.nbits = int(readfile_with_str(f"readfile {file_name}", "grep 'bits per sample'").split("=")[-1].strip())
                    self.nchan = int(readfile_with_str(f"readfile {file_name}", "grep 'Number of channels'").split("=")[-1].strip())
                    self.chanbw_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'Channel width (MHz)'").split("=")[-1].strip())
                    self.bw_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'Total Bandwidth (MHz)'").split("=")[-1].strip())
                    self.Tstart_MJD = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'MJD start time (STT_\\*)'").split("=")[-1].strip())
                    self.freq_high_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'High channel (MHz)'").split("=")[-1].strip())
                    self.freq_low_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'Low channel (MHz)'").split("=")[-1].strip())
                    self.freq_central_MHz = (self.freq_high_MHz + self.freq_low_MHz) / 2.0
                    print(self.N_samples, self.t_samp_s, self.T_obs_s, self.nbits, self.nchan, self.chanbw_MHz, self.bw_MHz, self.Tstart_MJD, self.freq_high_MHz, self.freq_central_MHz, self.freq_low_MHz)
               
                except:
                    print("警告：'readfile'失败。尝试使用'header'获取必要信息...")

                    # 使用'header'工具提取信息
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

                    print(self.N_samples, self.t_samp_s, self.T_obs_s, self.nbits, self.nchan, self.chanbw_MHz, self.bw_MHz, self.backend, self.Tstart_MJD, self.freq_high_MHz, self.freq_central_MHz, self.freq_low_MHz)

        if data_type == "psrfits":  # 处理PSRFITS文件
            print("\n正在读取PSRFITS文件....")
            if psrfits.is_PSRFITS(file_name):  # 检查文件是否为PSRFITS格式
                print("文件'%s'被正确识别为PSRFITS格式" % (file_name))
                object_file = psrfits.PsrfitsFile(self.file_abspath)  # 使用PSRFITS模块读取文件

                # 提取文件的关键信息
                self.bw_MHz = object_file.specinfo.BW  # 总带宽（MHz）
                self.N_samples = object_file.specinfo.N  # 总采样点数
                self.T_obs_s = object_file.specinfo.T  # 观测总时长（秒）
                self.backend = object_file.specinfo.backend  # 后端设备
                self.nbits = object_file.specinfo.bits_per_sample  # 数据位宽
                self.date_obs = object_file.specinfo.date_obs  # 观测日期
                self.dec_deg = object_file.specinfo.dec2000  # 赤纬（度）
                self.dec_str = object_file.specinfo.dec_str  # 赤纬（字符串格式）
                self.chanbw_MHz = object_file.specinfo.df  # 每个频道的带宽（MHz）
                self.t_samp_s = object_file.specinfo.dt  # 采样时间间隔（秒）
                self.freq_central_MHz = object_file.specinfo.fctr  # 中心频率（MHz）
                self.receiver = object_file.specinfo.frontend  # 接收器
                self.freq_high_MHz = object_file.specinfo.hi_freq  # 最高频率（MHz）
                self.freq_low_MHz = object_file.specinfo.lo_freq  # 最低频率（MHz）
                self.MJD_int = object_file.specinfo.mjd  # 起始MJD的整数部分
                self.MJD_sec = object_file.specinfo.secs  # 起始MJD的小数部分（秒）
                self.Tstart_MJD = self.MJD_int + np.float64(self.MJD_sec / 86400.)  # 起始MJD时间
                self.nchan = object_file.specinfo.num_channels  # 频道
                self.observer = object_file.specinfo.observer
                self.project = object_file.specinfo.project_id
                self.ra_deg = object_file.specinfo.ra2000
                self.ra_str = object_file.specinfo.ra_str
                self.seconds_of_day = object_file.specinfo.secs
                self.source_name = object_file.specinfo.source
                self.telescope = object_file.specinfo.telescope

        else:
                print("\nReading PSRFITS (header only)....")
                self.bw_MHz = np.float64(get_command_output("vap -n -c bw %s" % (file_name)).split()[-1])
                self.N_samples = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Spectra").split("=")[-1])
                self.T_obs_s = np.float64(get_command_output("vap -n -c length %s" % (file_name)).split()[-1])
                self.backend = get_command_output("vap -n -c backend %s" % (file_name)).split()[-1]
                self.nbits = int(get_command_output_with_pipe("readfile %s" % (file_name), "grep bits").split("=")[-1])
                self.chanbw_MHz = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Channel").split("=")[-1])
                self.t_samp_s = np.float64(get_command_output("vap -n -c tsamp %s" % (file_name)).split()[-1])
                self.freq_central_MHz = np.float64(get_command_output("vap -n -c freq %s" % (file_name)).split()[-1])
                self.receiver = get_command_output("vap -n -c rcvr %s" % (file_name)).split()[-1]
                self.freq_high_MHz = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep High").split("=")[-1])
                self.freq_low_MHz = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Low").split("=")[-1])
                self.nchan = int(get_command_output("vap -n -c nchan %s" % (file_name)).split()[-1])
                self.MJD_int = int(get_command_output("psrstat -Q -c ext:stt_imjd %s" % (file_name)).split()[-1])
                self.MJD_sec_int = int(get_command_output("psrstat -Q -c ext:stt_smjd %s" % (file_name)).split()[-1])
                self.MJD_sec_frac = np.float64(get_command_output("psrstat -Q -c ext:stt_offs %s" % (file_name)).split()[-1])
                self.MJD_sec = self.MJD_sec_int + self.MJD_sec_frac
                self.Tstart_MJD       = self.MJD_int + np.float64(self.MJD_sec/86400.)

class SurveyConfiguration(object):
        def __init__(self, config_filename, verbosity_level=1):
                self.config_filename = config_filename
                self.list_datafiles = []
                self.list_survey_configuration_ordered_params = ['SEARCH_LABEL', 'DATA_TYPE', 'ROOT_WORKDIR', 'PRESTO', 'PRESTO_GPU','IF_DDPLAN', 'DM_MIN', 'DM_MAX','DM_STEP', 'DM_COHERENT_DEDISPERSION', 'N_SUBBANDS', 'PERIOD_TO_SEARCH_MIN', 'PERIOD_TO_SEARCH_MAX', 'LIST_SEGMENTS', 'RFIFIND_TIME', 'RFIFIND_CHANS_TO_ZAP', 'RFIFIND_TIME_INTERVALS_TO_ZAP', 'IGNORECHAN_LIST', 'ZAP_ISOLATED_PULSARS_FROM_FFTS', 'ZAP_ISOLATED_PULSARS_MAX_HARM', 'FLAG_ACCELERATION_SEARCH', 'ACCELSEARCH_LIST_ZMAX', 'ACCELSEARCH_NUMHARM', 'FLAG_JERK_SEARCH', 'JERKSEARCH_ZMAX', 'JERKSEARCH_WMAX', 'JERKSEARCH_NUMHARM', 'SIFTING_FLAG_REMOVE_DUPLICATES', 'SIFTING_FLAG_REMOVE_DM_PROBLEMS', 'SIFTING_FLAG_REMOVE_HARMONICS', 'SIFTING_MINIMUM_NUM_DMS', 'SIFTING_MINIMUM_DM', 'SIFTING_SIGMA_THRESHOLD', 'FLAG_FOLD_KNOWN_PULSARS', 'FLAG_FOLD_TIMESERIES', 'FLAG_FOLD_RAWDATA', 'RFIFIND_FLAGS', 'PREPDATA_FLAGS', 'PREPSUBBAND_FLAGS', 'REALFFT_FLAGS', 'REDNOISE_FLAGS', 'ACCELSEARCH_FLAGS', 'ACCELSEARCH_GPU_FLAGS', 'ACCELSEARCH_JERK_FLAGS', 'PREPFOLD_FLAGS', 'FLAG_SINGLEPULSE_SEARCH', 'SINGLEPULSE_SEARCH_FLAGS', 'USE_CUDA', 'CUDA_IDS', 'NUM_SIMULTANEOUS_JERKSEARCHES', 'NUM_SIMULTANEOUS_PREPFOLDS', 'NUM_SIMULTANEOUS_PREPSUBBANDS', 'MAX_SIMULTANEOUS_DMS_PER_PREPSUBBAND', 'FAST_BUFFER_DIR', 'FLAG_KEEP_DATA_IN_BUFFER_DIR', 'FLAG_REMOVE_FFTFILES', 'FLAG_REMOVE_DATFILES_OF_SEGMENTS', 'STEP_RFIFIND', 'STEP_ZAPLIST', 'STEP_DEDISPERSE', 'STEP_REALFFT', 'STEP_PERIODICITY_SEARCH', 'STEP_SIFTING', 'STEP_FOLDING', 'STEP_SINGLEPULSE_SEARCH']
                self.dict_survey_configuration = {}
                config_file = open(config_filename, "r" )

                for line in config_file:
                        if line != "\n" and (not line.startswith("#")):
                                list_line = shlex.split(line)
                                self.dict_survey_configuration[list_line[0]] = list_line[1]  # Save parameter key and value in the dictionary 
                for key in list(self.dict_survey_configuration.keys()):
                        if   key == "SEARCH_LABEL":                      self.search_label                     = self.dict_survey_configuration[key]
                        elif key == "DATA_TYPE":                         self.data_type                        = self.dict_survey_configuration[key]
                        elif key == "ROOT_WORKDIR":
                                if self.dict_survey_configuration[key] == "(cwd)":
                                        self.root_workdir                     = os.getcwd()
                                        print("ROOT_WORKDIR == '(cwd)' --->  ROOT_WORKDIR set as current working directory '%s' \n" % self.root_workdir)
                                else:
                                        self.root_workdir                     = self.dict_survey_configuration[key]
                                        if os.path.exists(self.root_workdir) == False:
                                                print("%sERROR:%s %s directory '%s' does not exist!" % (colors.ERROR+colors.BOLD, colors.ENDCOLOR, key, self.root_workdir ))
                                                print("Please make sure that the path of %s in your configuration file is correct." % (key))
                                                print("Alternatively, you can use '(cwd)' to tell PULSAR_MINER to use the Current Working Directory as the ROOT_WORKDIR")
                                                exit()
                        elif key == "PRESTO":
                                if check_presto_path(presto_path=self.dict_survey_configuration[key], key=key) == True:
                                        self.presto_env                       = self.dict_survey_configuration[key]

                        elif key == "PRESTO_GPU":
                                if check_presto_path(presto_path=self.dict_survey_configuration[key], key=key) == True:
                                        self.presto_gpu_env                   = self.dict_survey_configuration[key]

                        elif key == "IF_DDPLAN":                            self.if_ddplan                             = self.dict_survey_configuration[key]
                        elif key == "DM_MIN":                               self.dm_min                                = self.dict_survey_configuration[key]
                        elif key == "DM_MAX":                               self.dm_max                                = self.dict_survey_configuration[key]
                        elif key == "DM_STEP":                              self.dm_step                                = self.dict_survey_configuration[key]

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
                print("\n ====================打印配置信息：  ====================== \n'")
                # 遍历有序参数列表并打印每个参数及其值
                for param in self.list_survey_configuration_ordered_params:
                        print("%-32s %s" % (param, self.dict_survey_configuration[param]))
                print()

def init_default(observation_filename):

        makedir("known_pulsars")
        makedir("01_RFIFIND")

        default_file_format = "psrfits"  # 默认文件格式设置为 "psrfits"
        default_obs = "<observation>"  # 默认观测文件名占位符

        # 根据输入的观测文件名判断文件格式并设置默认文件格式
        if observation_filename != "":
            default_obs = observation_filename  # 使用提供的观测文件名
            if observation_filename.endswith(".fil"):  
                print("输入文件 '%s' 似乎是filterbank格式。将默认文件格式设置为 'filterbank'。" % (observation_filename))
                default_file_format = "filterbank"  
            elif observation_filename.endswith(".fits") or observation_filename.endswith(".sf"):  # 如果文件以 ".fits" 或 ".sf" 结尾
                if psrfits.is_PSRFITS(observation_filename): 
                    default_file_format = "psrfits"  
                    print("输入文件 '%s' 似乎是 PSRFITS 格式。将默认文件格式设置为 'psrfits'。" % (observation_filename))
            else:
                print("\n警告：无法确定输入文件 '%s' 的格式.... 将默认文件格式设置为 'filterbank'。" % (observation_filename))
        else:
            print("警告：未提供输入观测文件。将默认文件格式设置为 'psrfits'。")

        # 尝试获取 PRESTO 环境变量的路径
        try:
            presto_path = os.environ['PRESTO']
        except:
            presto_path = "*** PRESTO 环境变量未定义 ***"  # 如果未定义，则提示未定义

        try:
            presto_gpu_path = os.environ['PRESTO2_ON_GPU']  # 尝试获取 PRESTO2_ON_GPU 环境变量
            use_cuda = '1'  # 设置使用 CUDA（GPU 加速）
        except:
            try:
                presto_gpu_path = os.environ['PRESTO_ON_GPU']  # 如果 PRESTO2_ON_GPU 未定义，尝试获取 PRESTO_ON_GPU 环境变量
                use_cuda = '1'  # 设置使用 CUDA（GPU 加速）
            except:
                try:
                    presto_gpu_path = os.environ['PRESTO']  # 如果 PRESTO_ON_GPU 也未定义，尝试获取 PRESTO 环境变量
                    use_cuda = '0'  # 设置不使用 CUDA（GPU 加速）
                    print("警告：未定义 PRESTO2_ON_GPU 或 PRESTO_ON_GPU 环境变量 - 将不使用 GPU 加速！")
                except:
                    print("错误：未定义 PRESTO 或 PRESTO_ON_GPU 环境变量！")
                    exit()  # 退出程序

                dict_survey_configuration_default_values = {
                    'SEARCH_LABEL':                          "%s               # 当前搜索项目的标签" % os.path.basename(os.getcwd()),
                    'DATA_TYPE':                             "%-18s            # 数据类型选项：filterbank 或 psrfits" % (default_file_format),
                    'ROOT_WORKDIR':                          "(cwd)            # 根工作目录的路径。选项：/绝对路径 或 '(cwd)' 表示当前工作目录",
                    'PRESTO':                                "%s               # 主要的 PRESTO 安装路径" % presto_path,
                    'PRESTO_GPU':                            "%s               # PRESTO_ON_GPU 安装路径（如果存在）" % presto_gpu_path,
                    'IF_DDPLAN':                             "1                # 是否执行ddplan？（1=是，0=否）",
                    'DM_MIN':                                "2.0              # 搜索的最小色散",
                    'DM_MAX':                                "100.0            # 搜索的最大色散",
                    'DM_STEP':                           "[(20, 30, 0.1)]      # 自定义搜索的色散间隔列表，IF_DDPLAN=0时使用",
                    'DM_COHERENT_DEDISPERSION':              "0                # 可能的相干去色散（CDD）的色散值（0 = 不进行 CDD）",
                    'N_SUBBANDS':                            "0                # 使用的子带数量（0 = 使用所有通道）",
                    'PERIOD_TO_SEARCH_MIN':                  "0.001            # 可接受的最小候选周期（秒）",
                    'PERIOD_TO_SEARCH_MAX':                  "20.0             # 可接受的最大候选周期（秒）,毫秒脉冲星可改为0.040",
                    'LIST_SEGMENTS':                         "full             # 用于搜索的分段长度（以分钟为单位），用逗号分隔（例如 \"full,20,10\"）",
                    'RFIFIND_TIME':                          "0.1              # RFIFIND 的 -time 选项值,FAST默认0.1",
                    'RFIFIND_CHANS_TO_ZAP':                  "\"\"             # 在 RFIFIND 掩模中需要消除的通道列表",
                    'RFIFIND_TIME_INTERVALS_TO_ZAP':         "\"\"             # 在 RFIFIND 掩模中需要消除的时间间隔列表",
                    'IGNORECHAN_LIST':                       "\"\"             # 分析中完全忽略的通道列表（PRESTO -ignorechan 选项）",
                    'ZAP_ISOLATED_PULSARS_FROM_FFTS':        "0                # 是否在功率谱中消除已知脉冲星？（1=是，0=否）",
                    'ZAP_ISOLATED_PULSARS_MAX_HARM':         "8                # 如果在功率谱中消除已知脉冲星，消除到这个谐波次数",
                    'FLAG_ACCELERATION_SEARCH':              "1                # 是否进行加速搜索？（1=是，0=否）",
                    'ACCELSEARCH_LIST_ZMAX':                 "0,200            # 使用 PRESTO accelsearch 时的 zmax 值列表（用逗号分隔）",
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
                    'FLAG_REMOVE_FFTFILES':                  "1                # 搜索后是否删除 FFT 文件以节省磁盘空间？（1=是，0=否）",
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
                print("警告：'%s' 已经存在！正在将默认配置保存到文件 '%s'" % (default_cfg_filename_existing, default_cfg_filename))
                print("******************")
                print()
        with open(default_cfg_filename, "w") as f:
                f.write("#===============================================================\n")
                f.write("# General parameters\n")
                f.write("#===============================================================\n")
                f.write("%-40s %s\n" % ('SEARCH_LABEL', dict_survey_configuration_default_values['SEARCH_LABEL']))
                f.write("%-40s %s\n" % ('DATA_TYPE', dict_survey_configuration_default_values['DATA_TYPE']))
                f.write("%-40s %s\n" % ('ROOT_WORKDIR', dict_survey_configuration_default_values['ROOT_WORKDIR']))
                f.write("%-40s %s\n" % ('PRESTO', dict_survey_configuration_default_values['PRESTO']))
                f.write("%-40s %s\n" % ('PRESTO_GPU', dict_survey_configuration_default_values['PRESTO_GPU']))
                f.write("\n")
                f.write("#===============================================================\n")
                f.write("# Core search parameters\n")
                f.write("#===============================================================\n")
                f.write("%-40s %s\n" % ('IF_DDPLAN', dict_survey_configuration_default_values['IF_DDPLAN']))
                f.write("%-40s %s\n" % ('DM_MIN', dict_survey_configuration_default_values['DM_MIN']))
                f.write("%-40s %s\n" % ('DM_MAX', dict_survey_configuration_default_values['DM_MAX']))
                f.write("%-40s %s\n" % ('DM_STEP', dict_survey_configuration_default_values['DM_STEP']))
                f.write("%-40s %s\n" % ('DM_COHERENT_DEDISPERSION', dict_survey_configuration_default_values['DM_COHERENT_DEDISPERSION']))
                f.write("%-40s %s\n" % ('N_SUBBANDS', dict_survey_configuration_default_values['N_SUBBANDS']))
                f.write("\n")                
                f.write("%-40s %s\n" % ('PERIOD_TO_SEARCH_MIN', dict_survey_configuration_default_values['PERIOD_TO_SEARCH_MIN']))
                f.write("%-40s %s\n" % ('PERIOD_TO_SEARCH_MAX', dict_survey_configuration_default_values['PERIOD_TO_SEARCH_MAX']))
                f.write("\n")
                f.write("%-40s %s\n" % ('LIST_SEGMENTS', dict_survey_configuration_default_values['LIST_SEGMENTS']))
                f.write("\n")
                f.write("#===============================================================\n")
                f.write("# Fourier domain search with PRESTO\n")
                f.write("#===============================================================\n")
                f.write("%-40s %s\n" % ('RFIFIND_TIME', dict_survey_configuration_default_values['RFIFIND_TIME']))
                f.write("%-40s %s\n" % ('RFIFIND_CHANS_TO_ZAP', dict_survey_configuration_default_values['RFIFIND_CHANS_TO_ZAP']))
                f.write("%-40s %s\n" % ('RFIFIND_TIME_INTERVALS_TO_ZAP', dict_survey_configuration_default_values['RFIFIND_TIME_INTERVALS_TO_ZAP']))
                f.write("%-40s %s\n" % ('IGNORECHAN_LIST', dict_survey_configuration_default_values['IGNORECHAN_LIST']))
                f.write("\n")
                f.write("%-40s %s\n" % ('ZAP_ISOLATED_PULSARS_FROM_FFTS', dict_survey_configuration_default_values['ZAP_ISOLATED_PULSARS_FROM_FFTS']))
                f.write("%-40s %s\n" % ('ZAP_ISOLATED_PULSARS_MAX_HARM', dict_survey_configuration_default_values['ZAP_ISOLATED_PULSARS_MAX_HARM']))
                f.write("\n")
                f.write("%-40s %s\n" % ('FLAG_ACCELERATION_SEARCH', dict_survey_configuration_default_values['FLAG_ACCELERATION_SEARCH']))
                f.write("%-40s %s\n" % ('ACCELSEARCH_LIST_ZMAX', dict_survey_configuration_default_values['ACCELSEARCH_LIST_ZMAX']))
                f.write("%-40s %s\n" % ('ACCELSEARCH_NUMHARM', dict_survey_configuration_default_values['ACCELSEARCH_NUMHARM']))
                f.write("\n")
                f.write("%-40s %s\n" % ('FLAG_JERK_SEARCH', dict_survey_configuration_default_values['FLAG_JERK_SEARCH']))
                f.write("%-40s %s\n" % ('JERKSEARCH_ZMAX', dict_survey_configuration_default_values['JERKSEARCH_ZMAX']))
                f.write("%-40s %s\n" % ('JERKSEARCH_WMAX', dict_survey_configuration_default_values['JERKSEARCH_WMAX']))
                f.write("%-40s %s\n" % ('JERKSEARCH_NUMHARM', dict_survey_configuration_default_values['JERKSEARCH_NUMHARM']))
                f.write("\n")
                f.write("%-40s %s\n" % ('SIFTING_FLAG_REMOVE_DUPLICATES', dict_survey_configuration_default_values['SIFTING_FLAG_REMOVE_DUPLICATES']))
                f.write("%-40s %s\n" % ('SIFTING_FLAG_REMOVE_DM_PROBLEMS', dict_survey_configuration_default_values['SIFTING_FLAG_REMOVE_DM_PROBLEMS']))
                f.write("%-40s %s\n" % ('SIFTING_FLAG_REMOVE_HARMONICS', dict_survey_configuration_default_values['SIFTING_FLAG_REMOVE_HARMONICS']))
                f.write("%-40s %s\n" % ('SIFTING_MINIMUM_NUM_DMS', dict_survey_configuration_default_values['SIFTING_MINIMUM_NUM_DMS']))
                f.write("%-40s %s\n" % ('SIFTING_MINIMUM_DM', dict_survey_configuration_default_values['SIFTING_MINIMUM_DM']))
                f.write("%-40s %s\n" % ('SIFTING_SIGMA_THRESHOLD', dict_survey_configuration_default_values['SIFTING_SIGMA_THRESHOLD']))
                f.write("\n")
                f.write("%-40s %s\n" % ('FLAG_FOLD_KNOWN_PULSARS', dict_survey_configuration_default_values['FLAG_FOLD_KNOWN_PULSARS']))
                f.write("%-40s %s\n" % ('FLAG_FOLD_TIMESERIES', dict_survey_configuration_default_values['FLAG_FOLD_TIMESERIES']))
                f.write("%-40s %s\n" % ('FLAG_FOLD_RAWDATA', dict_survey_configuration_default_values['FLAG_FOLD_RAWDATA']))
                f.write("\n")
                f.write("%-40s %s\n" % ('RFIFIND_FLAGS', dict_survey_configuration_default_values['RFIFIND_FLAGS']))
                f.write("%-40s %s\n" % ('PREPDATA_FLAGS', dict_survey_configuration_default_values['PREPDATA_FLAGS']))
                f.write("%-40s %s\n" % ('PREPSUBBAND_FLAGS', dict_survey_configuration_default_values['PREPSUBBAND_FLAGS']))
                f.write("%-40s %s\n" % ('REALFFT_FLAGS', dict_survey_configuration_default_values['REALFFT_FLAGS']))
                f.write("%-40s %s\n" % ('REDNOISE_FLAGS', dict_survey_configuration_default_values['REDNOISE_FLAGS']))
                f.write("%-40s %s\n" % ('ACCELSEARCH_FLAGS', dict_survey_configuration_default_values['ACCELSEARCH_FLAGS']))
                f.write("%-40s %s\n" % ('ACCELSEARCH_GPU_FLAGS', dict_survey_configuration_default_values['ACCELSEARCH_GPU_FLAGS']))
                f.write("%-40s %s\n" % ('ACCELSEARCH_JERK_FLAGS', dict_survey_configuration_default_values['ACCELSEARCH_JERK_FLAGS']))
                f.write("%-40s %s\n" % ('PREPFOLD_FLAGS', dict_survey_configuration_default_values['PREPFOLD_FLAGS']))
                f.write("\n")
                f.write("#===============================================================\n")
                f.write("# Single pulse search with PRESTO\n")
                f.write("#===============================================================\n")
                f.write("%-40s %s\n" % ('FLAG_SINGLEPULSE_SEARCH', dict_survey_configuration_default_values['FLAG_SINGLEPULSE_SEARCH']))
                f.write("%-40s %s\n" % ('SINGLEPULSE_SEARCH_FLAGS', dict_survey_configuration_default_values['SINGLEPULSE_SEARCH_FLAGS']))
                f.write("\n")
                f.write("#===============================================================\n")
                f.write("# Computational/Performance parameters\n")
                f.write("#===============================================================\n")
                f.write("%-40s %s\n" % ('USE_CUDA', dict_survey_configuration_default_values['USE_CUDA']))
                f.write("%-40s %s\n" % ('CUDA_IDS', dict_survey_configuration_default_values['CUDA_IDS']))
                f.write("\n")
                f.write("%-40s %s\n" % ('NUM_SIMULTANEOUS_JERKSEARCHES', dict_survey_configuration_default_values['NUM_SIMULTANEOUS_JERKSEARCHES']))
                f.write("%-40s %s\n" % ('NUM_SIMULTANEOUS_PREPFOLDS', dict_survey_configuration_default_values['NUM_SIMULTANEOUS_PREPFOLDS']))
                f.write("%-40s %s\n" % ('NUM_SIMULTANEOUS_PREPSUBBANDS', dict_survey_configuration_default_values['NUM_SIMULTANEOUS_PREPSUBBANDS']))
                f.write("%-40s %s\n" % ('MAX_SIMULTANEOUS_DMS_PER_PREPSUBBAND', dict_survey_configuration_default_values['MAX_SIMULTANEOUS_DMS_PER_PREPSUBBAND']))
                f.write("%-40s %s\n" % ('NUM_SIMULTANEOUS_SINGLEPULSE_SEARCHES', dict_survey_configuration_default_values['NUM_SIMULTANEOUS_SINGLEPULSE_SEARCHES']))
                f.write("\n")
                f.write("%-40s %s\n" % ('FAST_BUFFER_DIR', dict_survey_configuration_default_values['FAST_BUFFER_DIR']))
                f.write("%-40s %s\n" % ('FLAG_KEEP_DATA_IN_BUFFER_DIR', dict_survey_configuration_default_values['FLAG_KEEP_DATA_IN_BUFFER_DIR'])) 
                f.write("%-40s %s\n" % ('FLAG_REMOVE_FFTFILES', dict_survey_configuration_default_values['FLAG_REMOVE_FFTFILES']))
                f.write("%-40s %s\n" % ('FLAG_REMOVE_DATFILES_OF_SEGMENTS', dict_survey_configuration_default_values['FLAG_REMOVE_DATFILES_OF_SEGMENTS']))
                f.write("\n")                        
                f.write("#===============================================================\n")
                f.write("# Pipeline steps to execute (1=do, 0=skip)\n")
                f.write("#===============================================================\n")
                f.write("%-40s %s\n" % ('STEP_RFIFIND', dict_survey_configuration_default_values['STEP_RFIFIND']))
                f.write("%-40s %s\n" % ('STEP_ZAPLIST', dict_survey_configuration_default_values['STEP_ZAPLIST']))
                f.write("%-40s %s\n" % ('STEP_DEDISPERSE', dict_survey_configuration_default_values['STEP_DEDISPERSE']))
                f.write("%-40s %s\n" % ('STEP_REALFFT', dict_survey_configuration_default_values['STEP_REALFFT']))
                f.write("%-40s %s\n" % ('STEP_PERIODICITY_SEARCH', dict_survey_configuration_default_values['STEP_PERIODICITY_SEARCH']))
                f.write("%-40s %s\n" % ('STEP_SIFTING', dict_survey_configuration_default_values['STEP_SIFTING']))
                f.write("%-40s %s\n" % ('STEP_FOLDING', dict_survey_configuration_default_values['STEP_FOLDING']))
                f.write("%-40s %s\n" % ('STEP_SINGLEPULSE_SEARCH', dict_survey_configuration_default_values['STEP_SINGLEPULSE_SEARCH']))
                f.write("\n")
                f.write("#===============================================================\n")

        print()
        print("默认配置已写入 '%s'。" % (default_cfg_filename))

        with open("common_birdies.txt", "w") as f:
                f.write("10.00   0.003     2     1     0\n")
                f.write("30.00    0.008     2     1     0\n")
                f.write("50.00    0.08      3     1     0\n")
        print("一些常见的干扰频率已写入 'common_birdies.txt'。")
        print()
        print("如果有的话，请将已知脉冲星的参数文件放在 'known_pulsars' 文件夹中。")
        print()
        print("建议单独创建观测数据的 RFIFIND 掩模，以确保合理比例的频带被屏蔽。")
        print("在你对掩模满意后，请将所有相关文件放在 '01_RFIFIND' 文件夹中，并确保它们的文件名与观测数据的文件名一致（例如，'myobs.fil' 应在 '01_RFIFIND' 中有对应的 'myobs_rfifind.mask'、'myobs_rfifind.inf' 等文件），以便 PULSAR_MINER 能够将掩模与观测数据关联并使用它。")
        print()
        print("现在请编辑配置文件，调整参数，并使用以下命令运行管道：")
        print("%s -config %s -obs %s" % (os.path.basename(sys.argv[0]), default_cfg_filename, default_obs))
        print()
        exit()

def common_start_string(sa, sb):
        """ returns the longest common substring from the beginning of sa and sb """
        def _iter():
                for a, b in zip(sa, sb):
                        if a == b:
                                yield a
                        else:
                                return

        return ''.join(_iter())

# def main():
cwd = os.getcwd()
verbosity_level = 3
obsname = ""
ps_pl_path = os.path.abspath( os.path.dirname( __file__ ) )

# SHELL ARGUMENTS
# 检查命令行参数，如果用户请求帮助或未提供参数，则显示帮助信息
if (len(sys.argv) == 1 or ("-h" in sys.argv) or ("-help" in sys.argv) or ("--help" in sys.argv)):
    # 打印程序的用法
    print("用法: %s -config <配置文件> -obs <观测文件> [{-Q | -v | -V}]" % (os.path.basename(sys.argv[0])))
    print()
    # 打印参数说明
    print("%10s  %-32s:  %-50s" % ("-h", "", "打印帮助信息"))
    print("%10s  %-32s:  %-50s %s" % ("-config", "<配置文件>", "输入搜索配置文件", ""))
    print("%10s  %-32s:  %-50s" % ("-obs", "<观测文件>", "要搜索的数据文件"))
    print()
    # 打印如何生成默认配置文件的提示
    print("创建默认配置文件: \033[1m%s -init_default [<观测文件>]\033[0m" % (os.path.basename(sys.argv[0])))
    print()
    exit()

else:
    for j in range(1, len(sys.argv)):
            if (sys.argv[j] == "-config"):
                    config_filename = sys.argv[j+1]
            elif (sys.argv[j] == "-obs"):
                    obsname = sys.argv[j+1]
            elif (sys.argv[j] == "-init_default"):
                    try:
                        observation_filename = sys.argv[j+1]
                    except: observation_filename = ""
                    init_default(observation_filename)
                    print()
                    exit()

print();print_log(f' FAST DATA PULSAR SEARCH PIPELINE '.center(80, '-')); print()
print_log(' Author: Long Peng '.center(80, ' '));print()
print_log(f' See web page: https://www.plxray.cn/ '.center(80, ' '),masks='https://www.plxray.cn/'); print()

print_log(' Adapted from: https://github.com/alex88ridolfi/PULSAR_MINER'.center(80, ' '));print()
print_log(' Script created on Feb 2025'.center(80, ' '));print()
print_log('--'.center(80, '-'));print()
time.sleep(3)

print_log(f'程序开始：当前路径{cwd}\n', log_files='time')
print_log(f'运行日期：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n', log_files='time')
time.sleep(3)


config = SurveyConfiguration(config_filename, 1)

if obsname == "":
    # 如果未指定观测文件，提示用户通过 '-obs' 选项提供文件
    print()
    print("%s错误:%s 请通过 '-obs' 选项指定观测文件。" % (colors.ERROR + colors.BOLD, colors.ENDCOLOR))
    exit()

# 如果指定了观测文件...
elif obsname != "":
    # 通过 glob 模块获取所有匹配的观测文件
    config.list_datafiles = [os.path.basename(x) for x in glob.glob(obsname)]
    if len(config.list_datafiles) == 0:
        # 如果未找到任何观测文件，提示用户检查文件名是否正确
        print()
        print("%s错误:%s 未找到观测文件！请确保文件名正确无误。" % (colors.ERROR + colors.BOLD, colors.ENDCOLOR))
        exit()
    elif len(config.list_datafiles) >= 1:
        # 如果找到一个或多个文件，检查每个文件是否存在以及文件大小是否为零
        for f in config.list_datafiles:
            if not os.path.exists(f):
                # 如果文件不存在，提示用户检查文件路径或符号链接是否损坏
                print()
                print("%s错误:%s 文件 '%s' 不存在！可能是符号链接损坏。" % (colors.ERROR + colors.BOLD, colors.ENDCOLOR, f))
                exit()
            elif os.path.getsize(f) == 0:
                # 如果文件大小为零，提示用户检查文件是否损坏
                print()
                print("%s错误:%s 文件 '%s' 的大小为 0！" % (colors.ERROR + colors.BOLD, colors.ENDCOLOR, f))
                exit()
            config.folder_datafiles           = os.path.dirname(os.path.abspath(obsname)) 

# 现在创建包含绝对路径的观测文件列表
config.list_datafiles_abspath = [os.path.join(config.folder_datafiles, x) for x in config.list_datafiles]
# 创建观测对象列表
config.list_Observations = [Observation(x, config.data_type) for x in config.list_datafiles_abspath]
# 添加通用birdies文件
config.file_common_birdies = os.path.join(config.root_workdir, "common_birdies.txt")

################################################################################
#   IMPORT PARFILES OF KNOWN PULSARS
################################################################################

dir_known_pulsars = os.path.join(config.root_workdir, "known_pulsars")

list_known_pulsars = []
if os.path.exists(dir_known_pulsars):
    # 获取已知脉冲星参数文件（.par）的列表
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
        print("正在读取 '%s' --> 已将 %s 添加到已知脉冲星列表（%s）" % (os.path.basename(list_parfilenames[k]), current_pulsar.psr_name, current_pulsar.pulsar_type))

        # 如果配置中要求从傅里叶频谱中消除孤立脉冲星的频率，打印警告信息
        if config.zap_isolated_pulsars_from_ffts == 1:
            print()
            print()
            print("警告：我将消除孤立脉冲星的傅里叶频率（最多到 %d 阶谐波），具体如下" % (config.zap_isolated_pulsars_max_harm))
            print()
            for key in sorted(dict_freqs_to_zap.keys()):
                print("%s  -->  在观测历元的质心频率: %.14f Hz" % (key, dict_freqs_to_zap[key]))
            print()

list_segments_to_remove = []
for seg in config.list_segments_nofull:
        if (np.float64(seg)*60) >= config.list_Observations[0].T_obs_s:
                print("%s警告%s：段 %sm 的长度超过了完整观测的长度 (%d 分钟)。将被忽略。" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, seg, config.list_Observations[0].T_obs_s / 60.) )
                list_segments_to_remove.append(seg)
        elif (np.float64(seg)*60) >= 0.80 * config.list_Observations[0].T_obs_s:
                print("%s警告%s：段 %sm 的长度超过了完整观测长度的 80%% (%d 分钟)。将被忽略。" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, seg, config.list_Observations[0].T_obs_s / 60.) )
                list_segments_to_remove.append(seg)

# 删除过长的段
for seg in list_segments_to_remove:
        config.list_segments_nofull.remove(seg)
        config.list_segments.remove(seg)


config.print_configuration()
sifting.sigma_threshold = config.sifting_sigma_threshold
print("main:: SIFTING.sigma_threshold = ", sifting.sigma_threshold)

LOG_dir = os.path.join(config.root_workdir, "LOG")


for i in range(len(config.list_Observations)):
        print()
        print("%s: \033[1m %s \033[0m  (%.2f s)" % ("观测", config.list_Observations[i].file_nameonly, config.list_Observations[i].T_obs_s))
        print()

        # # 检查观测文件的基本名称是否超过32个字符，以防止prepfold截断文件名
        # if len(config.list_Observations[i].file_basename) > 32:
        #         print("%s错误：%s 观测的基本名称 ('%s') 过长！" % (colors.ERROR+colors.BOLD, colors.ENDCOLOR, config.list_Observations[i].file_basename))
        #         print("       请将文件重命名，使其基本名称（即不带扩展名的文件名）长度不超过32个字符。")
        #         exit()
        
        # 如果指定了快速缓冲目录
        if config.fast_buffer_dir != "":
                if os.path.exists(config.fast_buffer_dir):
                        # 构造快速缓冲目录中的文件绝对路径
                        file_buffer_abspath = os.path.join(config.fast_buffer_dir, config.list_Observations[i].file_nameonly)
                        # 检查文件是否已存在，或文件大小是否一致
                        if (not os.path.exists(file_buffer_abspath) or (os.path.getsize(file_buffer_abspath) != os.path.getsize(config.list_Observations[i].file_abspath))):

                                # 检查快速缓冲目录是否有足够空间
                                if (os.path.getsize(config.list_Observations[i].file_abspath) < shutil.disk_usage(config.fast_buffer_dir).free):
                                        print()
                                        print("正在将 '%s' 复制到快速缓冲目录 '%s'（这可能需要一些时间）..." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir), end=""); sys.stdout.flush()
                                        file_buffer_abspath = shutil.copy(config.list_Observations[i].file_abspath, config.fast_buffer_dir)
                                        print("完成！")
                                
                                        config.list_Observations[i].file_abspath = file_buffer_abspath
                                        config.list_Observations[i].file_buffer_copy = file_buffer_abspath

                                        print("现在 config.list_Observations[i].file_abspath = ", config.list_Observations[i].file_abspath)
                                else:
                                        print()
                                        print("%s警告：%s 快速缓冲目录 '%s' 空间不足！" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, config.fast_buffer_dir))
                                        print("    -->  不使用快速缓冲目录。这可能导致处理速度变慢...")
                                        print()
                                        # time.sleep(10)

                        else:
                                print("'%s' 的副本已存在于快速缓冲目录 '%s' 中。跳过..." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir))
                                file_buffer_abspath = os.path.join(config.fast_buffer_dir, config.list_Observations[i].file_nameonly)
                                config.list_Observations[i].file_abspath = file_buffer_abspath
                                config.list_Observations[i].file_buffer_copy = file_buffer_abspath
                                print()
                                print("当前使用的观测文件为 '%s'。" % (config.list_Observations[i].file_abspath))

                else:
                        print("%s警告：%s 快速缓冲目录 '%s' 不存在！" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, config.fast_buffer_dir))
                        print("    -->  不使用快速缓冲目录。这可能导致处理速度变慢...")
                        print()
                        config.fast_buffer_dir = ""
                        # time.sleep(10)

        print()
        print()
        print("*******************************************************************")
        print("搜索方案：")
        print("*******************************************************************")

        config.dict_search_structure[config.list_Observations[i].file_basename] = {}
        for s in config.list_segments:
                print("Segment = %s of %s" % (s, config.list_segments))
                if s == "full":
                        segment_length_s             = config.list_Observations[i].T_obs_s
                        segment_length_min    = config.list_Observations[i].T_obs_s / 60.
                        segment_label = s
                else:
                        segment_length_min  = np.float64(s)
                        segment_length_s = np.float64(s) * 60
                        segment_label = "%dm" % (segment_length_min)
                        
                config.dict_search_structure[config.list_Observations[i].file_basename][segment_label] = {}

                N_chunks = int(config.list_Observations[i].T_obs_s / segment_length_s)
                fraction_left = (config.list_Observations[i].T_obs_s % segment_length_s) / segment_length_s
                if fraction_left >= 0.80:
                        N_chunks = N_chunks + 1
                                
                for ck in range(N_chunks):
                        chunk_label = "ck%02d" % (ck)
                        config.dict_search_structure[config.list_Observations[i].file_basename][segment_label][chunk_label] = {'candidates': []}

                print("    段: %8s     ---> %2d 块 (%s)" % (segment_label, N_chunks, ", ".join(sorted(config.dict_search_structure[config.list_Observations[i].file_basename][segment_label].keys()))), end=' ')

                if fraction_left >= 0.80:
                        print(" --> %s警告：%s 段 '%sm' 的最后一个块实际上稍短一些 (%.2f 分钟)！" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, s, fraction_left*segment_length_min))
                elif fraction_left > 0.10 and fraction_left < 0.80:
                        print("--> %s警告：%s 段 '%sm' 的最后一个块 (ck%02d) 只有 %d 分钟，将被忽略！" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, s, ck+1, fraction_left*segment_length_min))
                else:
                        print()


makedir(LOG_dir)


# if config.if
list_DDplan_scheme = get_DDplan_scheme(config.list_Observations[i].file_abspath,
                                       LOG_dir,
                                       LOG_dir,
                                       "LOG_diskspace",
                                       config.dm_min,
                                       config.dm_max,
                                       config.dm_coherent_dedispersion,
                                       config.max_simultaneous_dms_per_prepsubband,
                                       config.list_Observations[i].freq_central_MHz,
                                       config.list_Observations[i].bw_MHz,
                                       config.list_Observations[i].nchan,
                                       config.nsubbands,
                                       config.list_Observations[i].t_samp_s)

############################################################################################
#    检查磁盘空间
############################################################################################

num_DMs = 0 
for j in range(len(list_DDplan_scheme)):
        num_DMs = num_DMs + list_DDplan_scheme[j]['num_DMs']
        
# 调用函数检查磁盘空间是否足够
flag_enough_disk_space = False
flag_enough_disk_space = check_if_enough_disk_space(config.root_workdir, num_DMs, config.list_Observations[i].T_obs_s, config.list_Observations[i].t_samp_s, config.list_segments_nofull, config.flag_remove_fftfiles, config.flag_remove_datfiles_of_segments)

# 如果磁盘空间不足，打印错误信息并退出程序
if flag_enough_disk_space == False:
        print()
        print("%s错误：%s 磁盘空间不足！请释放空间或更改工作目录。" % (colors.ERROR+colors.BOLD, colors.ENDCOLOR))
        print("> 提示：为了最小化磁盘使用，请确保在配置文件中将 FLAG_REMOVE_FFTFILES 和 FLAG_REMOVE_DATFILES_OF_SEGMENTS 保留为默认值 1。")
        exit()

###########################################################################################

print()
print("##################################################################################################")
print("                                           STEP 1 - RFIFIND                                       ")
print("##################################################################################################")
print()


rfifind_masks_dir = os.path.join(config.root_workdir, "01_RFIFIND")

if not os.path.exists(rfifind_masks_dir):
        os.mkdir(rfifind_masks_dir)


for i in range(len(config.list_Observations)):
        time.sleep(0.2)
        config.list_Observations[i].mask = "%s/%s_rfifind.mask" % (rfifind_masks_dir, config.list_Observations[i].file_basename)

        flag_mask_present = check_rfifind_outfiles(rfifind_masks_dir, config.list_Observations[i].file_basename)

        # CASE 1: mask not present, STEP_RFIFIND = 0 
        if flag_mask_present == False and config.flag_step_rfifind == 0:
            # 如果掩模文件不存在且配置文件中 STEP_RFIFIND = 0，则提示错误并退出程序
            print()
            print()
            print("\033[1m  错误！掩模文件 '%s' 未找到，但 STEP_RFIFIND = 0！\033[0m" % (config.list_Observations[i].mask))
            print()
            print("您需要为您的观测创建掩模文件，才能运行该流程。")
            print()
            print("请在配置文件中将 STEP_RFIFIND 设置为 1，或者单独创建掩模文件，并将相关文件复制到 '01_RFIFIND' 目录中，然后重试。")
            print()
            exit()

        
        # CASE 2: mask not present, STEP_RFIFIND = 1
        if flag_mask_present == False and config.flag_step_rfifind == 1:
                LOG_basename = "01_rfifind_%s" % (config.list_Observations[i].file_nameonly)
                log_abspath = "%s/LOG_%s.txt" % (LOG_dir, LOG_basename)
                
                print()
                print("在 01_RFIFIND 文件夹中未找到掩模文件。将使用配置文件 '%s' 中指定的参数生成掩模文件。" % (config_filename))
                print()
                print("\033[1m >> 提示:\033[0m 使用 '\033[1mtail -f %s\033[0m' 查看 rfifind 的进度。" % (log_abspath))
                print()
                print("正在为观测 %3d/%d: '%s' 创建 rfifind 掩模文件..." % (i+1, len(config.list_Observations), config.list_Observations[i].file_nameonly), end=' ')
                sys.stdout.flush()

                make_rfifind_mask(config.list_Observations[i].file_abspath,
                                   rfifind_masks_dir,
                                   LOG_dir,
                                   LOG_basename,
                                   config.rfifind_time,
                                   config.rfifind_time_intervals_to_zap,
                                   config.rfifind_chans_to_zap,
                                   config.rfifind_flags,
                                   config.presto_env,
                                   verbosity_level
                                   )
        
        # CASE 3: mask is already present, STEP_RFIFIND = 1
        elif flag_mask_present == True and config.flag_step_rfifind == 1:
                if verbosity_level >= 1:
                        print()
                        print("GOOD! Mask '%s' already present! Will not create a new one." % (config.list_Observations[i].mask)  )
                        print()
                        print("STEP_RFIFIND=1. Will check that the mask found is ok.")
                        print()
                else:
                        pass

                
        # CASE 4: mask is already present, STEP_RFIFIND = 0
        elif flag_mask_present == True and config.flag_step_rfifind == 0:
                if verbosity_level >= 1:
                        print()
                        print("GOOD! Mask '%s' already present! Will not create a new one." % (config.list_Observations[i].mask)  )
                        print()
                        print("%sWARNING:%s%s STEP_RFIFIND=0. Will skip the step, and trust that the mask found is ok.%s" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, colors.BOLD, colors.ENDCOLOR))
                        print()

                else:
                        pass


        # If STEP_RFIFIND = 1, check the mask before continuing 
        if  config.flag_step_rfifind == 1:
            print("正在检查被掩蔽的频带比例（这可能需要一些时间，具体取决于掩模文件的大小）...", end=' '); sys.stdout.flush()
            mask = rfifind.rfifind(config.list_Observations[i].mask)
            fraction_masked_channels = np.float64(len(mask.mask_zap_chans))/mask.nchan
            print("完成！"); sys.stdout.flush()
            if verbosity_level >= 1:
                    print()
                    print("RFIFIND：被掩蔽的频率通道比例：%.2f%%" % (fraction_masked_channels * 100.))
                    print()
            if fraction_masked_channels > 0.5 and fraction_masked_channels <= 0.95:
                    print()
                    print("************************************************************************************************")
                    print("!!! %s警告%s%s：%.2f%% 的频带被掩蔽了！这似乎有点多！%s !!!" % (colors.WARNING+colors.BOLD, colors.ENDCOLOR, colors.BOLD, fraction_masked_channels * 100., colors.ENDCOLOR))
                    print("!!! 如果您认为太多，请尝试调整配置文件中的 RFIFIND 参数（例如增加 RFIFIND_FREQSIG）")
                    print("************************************************************************************************")
                    time.sleep(1)

            if fraction_masked_channels > 0.95:
                    print()
                    print("************************************************************************************************")
                    print("!!! %s错误%s%s：%.2f%% 的频带被掩蔽了！这太多了。%s!!!" % (colors.ERROR+colors.BOLD, colors.ENDCOLOR, colors.BOLD, fraction_masked_channels * 100., colors.ENDCOLOR))
                    print("!!!")
                    print("!!! %s请调整配置文件中的 RFIFIND 参数，使被掩蔽的通道比例（可能）远小于 95%%，然后重试。%s" % (colors.BOLD, colors.ENDCOLOR))
                    print("************************************************************************************************")
                    exit()


        weights_file = config.list_Observations[i].mask.replace(".mask", ".weights")
        if os.path.exists(weights_file):
                array_weights = np.loadtxt(weights_file, unpack=True, usecols=(0, 1,), skiprows=1)
                config.ignorechan_list = ",".join([str(x) for x in np.where(array_weights[1] == 0)[0] ])
                config.nchan_ignored = len(config.ignorechan_list.split(","))
                if verbosity_level >= 1:
                        print()
                        print()
                        print("找到 WEIGHTS 文件 '%s'。使用该文件忽略 %d 个通道，总共 %d 个通道（%.2f%%）" % (os.path.basename(weights_file), config.nchan_ignored, config.list_Observations[i].nchan, 100*config.nchan_ignored/np.float64(config.list_Observations[i].nchan)))
                        print("被忽略的通道： %s" % (config.ignorechan_list))
                                      

##################################################################################################
# 2) BIRDIES AND ZAPLIST
##################################################################################################

print()
print("##################################################################################################")
print("                                   STEP 2 - BIRDIES AND ZAPLIST                                   ")
print("##################################################################################################")
print()
print("STEP_ZAPLIST = %s" % (config.flag_step_zaplist))

dir_birdies = os.path.join(config.root_workdir, "02_BIRDIES")

if config.flag_step_zaplist == 1:
        print("# =====================================================================================")
        print("# a) 使用掩模为每个文件创建一个 0-DM 质心时间序列。")
        print("# =====================================================================================")
        makedir(dir_birdies)
        for i in range(len(config.list_Observations)):
                time.sleep(0.1)
                print()
                print("正在运行 prepdata 为 \"%s\" 创建 0-DM 和质心时间序列..." % (config.list_Observations[i].file_nameonly), end=' ')
                sys.stdout.flush()
                LOG_basename = "02a_prepdata_%s" % (config.list_Observations[i].file_nameonly)
                prepdata(config.list_Observations[i].file_abspath,
                          dir_birdies,
                          LOG_dir,
                          LOG_basename,
                          0,
                          config.list_Observations[i].N_samples,
                          config.ignorechan_list,
                          config.list_Observations[i].mask,
                          1,
                          "topocentric",
                          config.prepdata_flags,
                          config.presto_env,
                          verbosity_level
                          )
                if verbosity_level >= 1:
                        print("完成！"); sys.stdout.flush()
                        
        print("# ===============================================")
        print("# b) 对所有文件进行傅里叶变换")
        print("# ===============================================")
        print()

        config.list_0DM_datfiles = glob.glob("%s/*%s*.dat" % (dir_birdies, config.list_Observations[i].file_basename))   # 收集 02_BIRDIES_FOLDERS 中的 *.dat 文件
        for i in range(len(config.list_0DM_datfiles)):
                time.sleep(0.1)
                if verbosity_level >= 1:
                        print("正在对 0-DM 质心时间序列 '%s' 运行 realfft..." % (os.path.basename(config.list_0DM_datfiles[i])), end=' ')
                        sys.stdout.flush()

                LOG_basename = "02b_realfft_%s" % (os.path.basename(config.list_0DM_datfiles[i]))
                realfft(config.list_0DM_datfiles[i],
                        dir_birdies,
                        LOG_dir,
                        LOG_basename,
                        config.realfft_flags,
                        config.presto_env,
                        verbosity_level,
                        flag_LOG_append=0
                        )

                if verbosity_level >= 1:
                        print("完成！")
                        sys.stdout.flush()



        print()
        print("# ===============================================")
        print("# 02c) 去除红噪声")
        print("# ===============================================")
        print()
        config.list_0DM_fftfiles = [x for x in glob.glob("%s/*%s*DM00.00.fft" % (dir_birdies, config.list_Observations[i].file_basename)) if not "_red" in x]  # 收集 02_BIRDIES_FOLDERS 中的 *.fft 文件，排除已处理红噪声的文件

        # print "len(config.list_0DM_datfiles), len(config.list_0DM_fftfiles) = ", len(config.list_0DM_datfiles), len(config.list_0DM_fftfiles)

        for i in range(len(config.list_0DM_fftfiles)):
                time.sleep(0.1)
                print("正在对 FFT 文件 \"%s\" 运行 rednoise..." % (os.path.basename(config.list_0DM_datfiles[i])), end=' ')
                sys.stdout.flush()
                LOG_basename = "02c_rednoise_%s" % (os.path.basename(config.list_0DM_fftfiles[i]))
                rednoise(config.list_0DM_fftfiles[i],
                         dir_birdies,
                         LOG_dir,
                         LOG_basename,
                         config.rednoise_flags,
                         config.presto_env,
                         verbosity_level
                         )
                if verbosity_level >= 1:
                        print("完成！")
                        sys.stdout.flush()



        print()
        print("# ===============================================")
        print("# 02d) 加速搜索和创建 zaplist")
        print("# ===============================================")
        print()

        config.list_0DM_fft_rednoise_files = glob.glob("%s/*%s*_DM00.00.fft" % (dir_birdies, config.list_Observations[i].file_basename))  # 收集经过红噪声处理的 0-DM FFT 文件
        for i in range(len(config.list_0DM_fft_rednoise_files)):
                time.sleep(0.1)
                print("正在为 0-DM 质心时间序列 \"%s\" 创建 zaplist..." % (os.path.basename(config.list_0DM_datfiles[i])), end=' ')
                sys.stdout.flush() 
                LOG_basename = "02d_makezaplist_%s" % (os.path.basename(config.list_0DM_fft_rednoise_files[i]))
                zaplist_filename = make_zaplist(config.list_0DM_fft_rednoise_files[i],
                                                dir_birdies,
                                                LOG_dir,
                                                LOG_basename,
                                                config.file_common_birdies,
                                                2,
                                                config.accelsearch_flags,
                                                config.presto_env,
                                                verbosity_level
                                                )
                if verbosity_level >= 1:
                        print("完成！")
                        sys.stdout.flush()

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


dir_dedispersion = os.path.join(config.root_workdir, "03_DEDISPERSION")

print()
print()
print("##################################################################################################")
print("#                 STEP 3 - DEDISPERSION, DE-REDDENING AND PERIODICITY SEARCH")
print("##################################################################################################")
print()

LOG_basename = "03_prepsubband_and_search_FFT_%s" % (config.list_Observations[i].file_nameonly)
print("3) 去色散、去红噪声和周期性搜索：", end=' '); sys.stdout.flush()
makedir(dir_dedispersion)  # 创建去色散目录
print("get_DDplan_scheme(config.list_Observations[i].file_abspath, = ", config.list_Observations[i].file_abspath)  # 打印文件路径
print("LOG_basename = %s" % LOG_basename)  # 打印日志文件基础名称

list_DDplan_scheme = get_DDplan_scheme(config.list_Observations[i].file_abspath,
                                        dir_dedispersion,
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

################################################################################
# 1) 遍历每个观测
for i in range(len(config.list_Observations)):
        obs = config.list_Observations[i].file_basename
        time.sleep(1.0)
        work_dir_obs = os.path.join(dir_dedispersion, config.list_Observations[i].file_basename)
        print("3) 去色散、去红噪声和周期性搜索：正在创建工作目录 '%s'..." % (work_dir_obs), end=' '); sys.stdout.flush()
        makedir(work_dir_obs)
        print("完成！"); sys.stdout.flush()


# 2) 遍历每个分段
if not "full" in list(config.dict_search_structure[obs].keys()):
        print("提示：不会对完整长度的观测进行搜索！")
        config.dict_search_structure[obs]['full'] = {'ck00': {'candidates': []}}
list_segments = ['full'] + ["%sm" % (x) for x in sorted(config.list_segments_nofull)]
# else:
#        list_segments =  ["%sm" % (x) for x in sorted(config.list_segments)]

N_seg = len(list_segments)
for seg, i_seg in zip(list_segments, list(range(N_seg))):
        work_dir_segment = os.path.join(work_dir_obs, "%s" % seg)
        print("\n3) 去色散、去红噪声和周期性搜索：正在创建工作目录 '%s'..." % (work_dir_segment), end=' '); sys.stdout.flush()
        makedir(work_dir_segment)
        print("完成！"); sys.stdout.flush()

# 3) 遍历块
N_ck = len(list(config.dict_search_structure[obs][seg].keys()))
for ck, i_ck in zip(sorted(config.dict_search_structure[obs][seg].keys()), list(range(N_ck))):
    print()
    print("**************************************************************")
    print("分段 %s 的 %s  -- 块 %s 的 %s" % (seg, sorted(config.dict_search_structure[obs].keys()), ck, sorted(config.dict_search_structure[obs][seg].keys())))
    print("**************************************************************")
    work_dir_chunk = os.path.join(work_dir_segment, ck)
    print("3) 去色散、去红化和周期性搜索：正在创建工作目录 '%s'..." % (work_dir_chunk), end=' '); sys.stdout.flush()
    makedir(work_dir_chunk)
    print("完成!"); sys.stdout.flush()

    zapfile = "%s/%s_DM00.00.zaplist" % (dir_birdies, config.list_Observations[i].file_basename)

    print("掩膜::: ", config.list_Observations[i].mask)
    print()
    print("**********************")
    print()
    print("config.list_cuda_ids = ", config.list_cuda_ids)
    print()
    print("config.presto_env = ", config.presto_env)
    print("config.presto_gpu_env = ", config.presto_gpu_env)
    print("**********************")

    dict_flag_steps = {'flag_step_dedisperse': config.flag_step_dedisperse, 'flag_step_realfft': config.flag_step_realfft, 'flag_step_periodicity_search': config.flag_step_periodicity_search}
    dedisperse_rednoise_and_periodicity_search_FFT(config.list_Observations[i].file_abspath,
                                                    work_dir_chunk,
                                                    config.root_workdir,
                                                    LOG_dir,
                                                    LOG_basename,
                                                    config.flag_search_full,
                                                    seg,
                                                    ck,
                                                    [i_seg, N_seg, i_ck, N_ck],
                                                    zapfile,
                                                    make_even_number(config.list_Observations[i].N_samples/1.0),
                                                    config.ignorechan_list,
                                                    config.list_Observations[i].mask,
                                                    list_DDplan_scheme,
                                                    config.list_Observations[i].nchan,
                                                    config.nsubbands,
                                                    config.num_simultaneous_prepsubbands,
                                                    config.prepsubband_flags,
                                                    config.presto_env,
                                                    config.flag_use_cuda,
                                                    config.list_cuda_ids,
                                                    config.flag_acceleration_search,
                                                    config.accelsearch_numharm,
                                                    config.accelsearch_list_zmax,
                                                    config.flag_jerk_search,
                                                    config.jerksearch_zmax,
                                                    config.jerksearch_wmax,
                                                    config.jerksearch_numharm,
                                                    config.num_simultaneous_jerksearches,
                                                    config.period_to_search_min,
                                                    config.period_to_search_max,
                                                    config.accelsearch_flags,
                                                    config.flag_remove_fftfiles,
                                                    config.flag_remove_datfiles_of_segments,
                                                    config.presto_env,
                                                    config.presto_gpu_env,
                                                    verbosity_level,
                                                    dict_flag_steps)


if config.flag_step_sifting == 1:
        print()
        print("##################################################################################################")
        print("#                                  STEP 4 - CANDIDATE SIFTING ")
        print("##################################################################################################")

        dir_sifting = os.path.join(config.root_workdir, "04_SIFTING")
        print("4) 候选体筛选：正在创建工作目录...", end=' '); sys.stdout.flush()
        makedir(dir_sifting)
        print("完成！")

        dict_candidate_lists = {}

        for i in range(len(config.list_Observations)):
                obs = config.list_Observations[i].file_basename
                print("Sifting candidates for observation %3d/%d '%s'." % (i+1, len(config.list_Observations), obs)) 
                for seg in sorted(config.dict_search_structure[obs].keys()):
                        work_dir_segment = os.path.join(dir_sifting, config.list_Observations[i].file_basename, "%s" % seg)
                        makedir(work_dir_segment)

                        for ck in sorted(config.dict_search_structure[obs][seg].keys()):
                                work_dir_chunk = os.path.join(work_dir_segment, ck)
                                makedir(work_dir_chunk)
                                LOG_basename = "04_sifting_%s_%s_%s" % (obs, seg, ck)
                                work_dir_candidate_sifting = os.path.join(dir_sifting, obs, seg, ck)

                                print("4) CANDIDATE SIFTING: Creating working directory '%s'..." % (work_dir_candidate_sifting), end=' '); sys.stdout.flush()
                                makedir(work_dir_candidate_sifting)
                                print("done!")
                                print("4) CANDIDATE SIFTING: Sifting observation %d) \"%s\" / %s / %s..." % (i+1, obs, seg, ck), end=' ')
                                sys.stdout.flush()

                                config.dict_search_structure[obs][seg][ck]['candidates'] = sift_candidates(work_dir_chunk,
                                                                                                            LOG_dir,
                                                                                                            LOG_basename,
                                                                                                            dir_dedispersion,
                                                                                                            obs,
                                                                                                            seg,
                                                                                                            ck,
                                                                                                            config.accelsearch_list_zmax,
                                                                                                            config.jerksearch_zmax,
                                                                                                            config.jerksearch_wmax,
                                                                                                            config.sifting_flag_remove_duplicates,
                                                                                                            config.sifting_flag_remove_dm_problems,
                                                                                                            config.sifting_flag_remove_harmonics,
                                                                                                            config.sifting_minimum_num_DMs,
                                                                                                            config.sifting_minimum_DM,
                                                                                                            config.period_to_search_min,
                                                                                                            config.period_to_search_max
                                )

        for i in range(len(config.list_Observations)):
            # 构造候选体汇总文件的路径和文件名
            candidates_summary_filename = "%s/%s_cands.summary" % (dir_sifting, config.list_Observations[i].file_basename)
            candidates_summary_file = open(candidates_summary_filename, 'w')

            # 初始化需要折叠的候选体总数
            count_candidates_to_fold_all = 0
            # 写入文件分隔符
            candidates_summary_file.write("\n*****************************************************************")
            # 写入当前观测文件中找到的候选体信息
            candidates_summary_file.write("\n在 %s 中找到的候选体：\n\n" % (config.list_Observations[i].file_nameonly))
            # 遍历所有段和块，统计候选体数量
            for seg in sorted(config.dict_search_structure[obs].keys()):
                for ck in sorted(config.dict_search_structure[obs][seg].keys()):
                    Ncands_seg_ck = len(config.dict_search_structure[obs][seg][ck]['candidates'])
                    # 写入每个段和块的候选体数量
                    candidates_summary_file.write("%20s  |  %10s  ---> %4d 候选体\n" % (seg, ck, Ncands_seg_ck))
                    count_candidates_to_fold_all = count_candidates_to_fold_all + Ncands_seg_ck
            # 写入总候选体数量
            candidates_summary_file.write("\n总计 = %d 候选体\n" % (count_candidates_to_fold_all))
            candidates_summary_file.write("*****************************************************************\n\n")

            count_candidates_to_fold_redet = 0
            count_candidates_to_fold_new = 0
            list_all_cands = []
            for seg in sorted(config.dict_search_structure[obs].keys()):
                    for ck in sorted(config.dict_search_structure[obs][seg].keys()):
                            config.dict_search_structure[obs][seg][ck]['candidates_redetections'] = []
                            config.dict_search_structure[obs][seg][ck]['candidates_new'] = []

                            for j in range(len(config.dict_search_structure[obs][seg][ck]['candidates'])):
                                    candidate = config.dict_search_structure[obs][seg][ck]['candidates'][j]

                                    flag_is_know, known_psrname, str_harmonic = check_if_cand_is_known(candidate, list_known_pulsars, numharm=16)

                                    if flag_is_know == True:
                                            config.dict_search_structure[obs][seg][ck]['candidates_redetections'].append(candidate)
                                            count_candidates_to_fold_redet = count_candidates_to_fold_redet + 1
                                    elif flag_is_know == False:
                                            config.dict_search_structure[obs][seg][ck]['candidates_new'].append(candidate)
                                            count_candidates_to_fold_new = count_candidates_to_fold_new + 1

                                    dict_cand = {'cand': candidate, 'obs': obs, 'seg': seg, 'ck': ck, 'is_known': flag_is_know, 'known_psrname': known_psrname, 'str_harmonic': str_harmonic}
                                    list_all_cands.append(dict_cand)
            N_cands_all = len(list_all_cands)

            for i_cand, dict_cand in zip(list(range(0, N_cands_all)), sorted(list_all_cands, key=lambda k: k['cand'].p, reverse=False)):
                    if dict_cand['cand'].DM < 2:
                            candidates_summary_file.write("Cand %4d/%d: %12.6f ms    |  DM: %7.2f pc cm-3    (%4s / %4s | sigma: %5.2f)  ---> Likely RFI\n" % (i_cand+1, N_cands_all, dict_cand['cand'].p * 1000., dict_cand['cand'].DM, dict_cand['seg'], dict_cand['ck'], dict_cand['cand'].sigma))
                    else:
                            if dict_cand['is_known'] == True:
                                    candidates_summary_file.write("Cand %4d/%d:  %12.6f ms  |  DM: %7.2f pc cm-3    (%4s / %4s | sigma: %5.2f)  ---> Likely %s - %s\n" % (i_cand+1, N_cands_all, dict_cand['cand'].p * 1000., dict_cand['cand'].DM, dict_cand['seg'], dict_cand['ck'], dict_cand['cand'].sigma, dict_cand['known_psrname'], dict_cand['str_harmonic']))
                            elif dict_cand['is_known'] == False:
                                    candidates_summary_file.write("Cand %4d/%d:  %12.6f ms  |  DM: %7.2f pc cm-3    (%4s / %4s | sigma: %5.2f)\n" % (i_cand+1, N_cands_all, dict_cand['cand'].p * 1000., dict_cand['cand'].DM, dict_cand['seg'], dict_cand['ck'], dict_cand['cand'].sigma))

            candidates_summary_file.close()

            if verbosity_level >= 1:
                    candidates_summary_file = open(candidates_summary_filename, 'r')
                    for line in candidates_summary_file:
                            print(line, end=' ')
                    candidates_summary_file.close()


if config.flag_step_folding == 1:
        print()
        print()
        print("##################################################################################################")
        print("#                                        STEP 5 - FOLDING ")
        print("##################################################################################################")
        print()

        dir_folding = os.path.join(config.root_workdir, "05_FOLDING")
        if verbosity_level >= 1:
                print("5) 折叠：正在创建工作目录...", end=' '); sys.stdout.flush()
        if not os.path.exists(dir_folding):
                os.mkdir(dir_folding)
        if verbosity_level >= 1:
                print("完成！")

        for i in range(len(config.list_Observations)):
                obs = config.list_Observations[i].file_basename
                print("正在折叠观测 '%s'" % (obs))
                print()

                work_dir_candidate_folding = os.path.join(dir_folding, config.list_Observations[i].file_basename)
                if verbosity_level >= 1:
                        print("5) 候选体折叠：正在创建工作目录 '%s'..." % (work_dir_candidate_folding), end=' '); sys.stdout.flush()
                if not os.path.exists(work_dir_candidate_folding):
                        os.mkdir(work_dir_candidate_folding)
                if verbosity_level >= 1:
                        print("完成！")

                file_script_fold_name = "script_fold.txt"
                file_script_fold_abspath = "%s/%s" % (work_dir_candidate_folding, file_script_fold_name)
                file_script_fold = open(file_script_fold_abspath, "w")
                file_script_fold.close()

                if config.flag_fold_known_pulsars == 1:
                        key_cands_to_fold = 'candidates'

                        print()
                        print("5) 候选体折叠：我将折叠所有 %d 个候选体（包括 %s 个可能是重复检测的候选体）" % (N_cands_all, count_candidates_to_fold_redet))
                        N_cands_to_fold = N_cands_all

                elif config.flag_fold_known_pulsars == 0:
                        key_cands_to_fold = 'candidates_new'
                        print()
                        print("5) 候选体折叠：我将仅折叠 %d 个可能是新脉冲星的候选体（%s 个可能是重复检测的候选体将不被折叠）" % (count_candidates_to_fold_new, count_candidates_to_fold_redet))
                        N_cands_to_fold = count_candidates_to_fold_new
                count_folded_ts = 1
                if config.flag_fold_timeseries == 1:

                        LOG_basename = "05_folding_%s_timeseries" % (obs)
                        print()
                        print("正在折叠时序数据...")
                        print()
                        print("\033[1m >> 提示:\033[0m 使用 '\033[1mtail -f %s/LOG_%s.txt\033[0m' 查看折叠进度" % (LOG_dir, LOG_basename))
                        print()
                        for seg in sorted(config.dict_search_structure[obs].keys()):
                                for ck in sorted(config.dict_search_structure[obs][seg].keys()):
                                        for j in range(len(config.dict_search_structure[obs][seg][ck][key_cands_to_fold])):
                                                candidate = config.dict_search_structure[obs][seg][ck][key_cands_to_fold][j]

                                                print("正在折叠候选体时序数据 %d/%d 的 %s: 段 %s / %s..." % (count_folded_ts, N_cands_to_fold, obs, seg, ck), end=' ')
                                                sys.stdout.flush()

                                                tstart_folding_cand_ts = time.time()
                                                file_to_fold = os.path.join(dir_dedispersion, obs, seg, ck, candidate.filename.split("_ACCEL")[0] + ".dat")
                                                flag_remove_dat_after_folding = 0
                                                if os.path.exists(file_to_fold):

                                                        fold_candidate(work_dir_candidate_folding,
                                                                LOG_dir,
                                                                LOG_basename,
                                                                config.list_Observations[i],
                                                                dir_dedispersion,
                                                                obs,
                                                                seg,
                                                                ck,
                                                                candidate,
                                                                config.ignorechan_list,
                                                                config.prepfold_flags,
                                                                config.presto_env,
                                                                verbosity_level,
                                                                1,
                                                                "timeseries",
                                                               config.num_simultaneous_prepfolds
                                                        )

                                                        tend_folding_cand_ts = time.time()
                                                        time_taken_folding_cand_ts_s = tend_folding_cand_ts - tstart_folding_cand_ts
                                                        print("done in %.2f s!" % (time_taken_folding_cand_ts_s))
                                                        sys.stdout.flush()
                                                        count_folded_ts = count_folded_ts + 1
                                                else:
                                                        print("dat文件不存在！可能是因为你在配置文件中设置了FLAG_REMOVE_DATFILES_OF_SEGMENTS = 1。跳过...")
                count_folded_raw = 1
                if config.flag_fold_rawdata == 1:
                        LOG_basename = "05_folding_%s_rawdata" % (obs)
                        print()
                        print("正在折叠原始数据 \033[1m >> 提示:\033[0m 使用 '\033[1mtail -f %s/LOG_%s.txt\033[0m' 查看折叠进度" % (LOG_dir, LOG_basename))
                        for seg in sorted(list(config.dict_search_structure[obs].keys()), reverse=True):
                                for ck in sorted(config.dict_search_structure[obs][seg].keys()):
                                        for j in range(len(config.dict_search_structure[obs][seg][ck][key_cands_to_fold])):
                                                candidate = config.dict_search_structure[obs][seg][ck][key_cands_to_fold][j]
                                                LOG_basename = "05_folding_%s_%s_%s_rawdata" % (obs, seg, ck)

                                                fold_candidate(work_dir_candidate_folding,
                                                                LOG_dir,
                                                                LOG_basename,
                                                                config.list_Observations[i],
                                                                dir_dedispersion,
                                                                obs,
                                                                seg,
                                                                ck,
                                                                candidate,
                                                                config.ignorechan_list,
                                                                config.prepfold_flags,
                                                                config.presto_env,
                                                                verbosity_level,
                                                                1,
                                                                "rawdata",
                                                               config.num_simultaneous_prepfolds
                                                )

                                                count_folded_raw = count_folded_raw + 1

                os.chdir(work_dir_candidate_folding)
                cmd_pm_run_multithread = "%s/pm_run_multithread -cmdfile %s -ncpus %d" % (os.path.dirname(sys.argv[0]), file_script_fold_abspath, config.num_simultaneous_prepfolds)
                print()
                print()
                print("5) CANDIDATE FOLDING - Now running:")
                print("%s" % cmd_pm_run_multithread)
                os.system(cmd_pm_run_multithread)

if config.flag_singlepulse_search == 1 and config.flag_step_singlepulse_search == 1:
        print()
        print()
        print("##################################################################################################")
        print("#                                        STEP 6 - SINGLE-PULSE SEARCH (PRESTO) ")
        print("##################################################################################################")
        print()

        dir_singlepulse_search = os.path.join(config.root_workdir, "06_SINGLEPULSE")
        if verbosity_level >= 1:
                print("6) 单脉冲搜索：正在创建工作目录...", end=' '); sys.stdout.flush()
        if not os.path.exists(dir_singlepulse_search):
                os.mkdir(dir_singlepulse_search)

        for i in range(len(config.list_Observations)):
                obs = config.list_Observations[i].file_basename
                time.sleep(1.0)
                work_dir_singlepulse_search_obs = os.path.join(dir_singlepulse_search, config.list_Observations[i].file_basename)
                if verbosity_level >= 2:
                        print("6) 单脉冲搜索：正在创建工作目录 '%s'..." % (work_dir_singlepulse_search_obs), end=' '); sys.stdout.flush()
                if not os.path.exists(work_dir_singlepulse_search_obs):
                        os.mkdir(work_dir_singlepulse_search_obs)
                if verbosity_level >= 2:
                        print("完成！"); sys.stdout.flush()

        if verbosity_level >= 1:
                print("完成！")


        # Go into the 06_SINGLEPULSE directory
        os.chdir(work_dir_singlepulse_search_obs)

        # Create symbolic links to all the full-length *.dat and corresponding *.inf files
        search_string_datfiles_full_length = "%s/03_DEDISPERSION/%s/full/ck00/*.dat" % (config.root_workdir, config.list_Observations[0].file_basename) #List of datfiles
        search_string_inffiles_full_length = "%s/03_DEDISPERSION/%s/full/ck00/*.inf" % (config.root_workdir, config.list_Observations[0].file_basename) #List of inffiles
        list_datfiles_full_length = glob.glob(search_string_datfiles_full_length)
        list_inffiles_full_length = glob.glob(search_string_inffiles_full_length)

        for f in list_datfiles_full_length + list_inffiles_full_length:
                symlink_filename = os.path.basename(f)
                if os.path.exists(symlink_filename) and os.path.islink(symlink_filename):
                        print("Symlink %s already exists. Skipping..." % (symlink_filename))
                else:
                        print("Making symbolic link of '%s'..." % (symlink_filename), end=''); sys.stdout.flush()
                        os.symlink(f, symlink_filename)
                        print("done!"); sys.stdout.flush()

        LOG_singlepulse_search_basename = "06_singlepulse_search_%s" % (config.list_Observations[0].file_basename)
        LOG_singlepulse_search_abspath  = "%s/LOG_%s.txt" % (LOG_dir, LOG_singlepulse_search_basename)
        
        list_datfiles_to_singlepulse_search = glob.glob("%s/*.dat" % work_dir_singlepulse_search_obs)


        singlepulse_search(work_dir_singlepulse_search_obs,
                           LOG_dir,
                           LOG_singlepulse_search_basename,
                           list_datfiles_to_singlepulse_search,
                           config.singlepulse_search_flags,
                           config.num_simultaneous_singlepulse_searches,
                           config.presto_env,
                           verbosity_level,
                           config.flag_step_singlepulse_search)

                
if config.list_Observations[i].file_buffer_copy != "":
        if config.flag_keep_data_in_buffer_dir == 1:
                print()
                print("Keeping a copy of '%s' from the buffer directory (%s)." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir))
                print("Remember to delete if you are not using it further.")
        else:
                print("Removing copy of '%s' from the buffer directory (%s)..." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir), end=""), ; sys.stdout.flush()
                os.remove(config.list_Observations[i].file_buffer_copy)
                print("done!")

print()



# if __name__ == "__main__":
#     main()

