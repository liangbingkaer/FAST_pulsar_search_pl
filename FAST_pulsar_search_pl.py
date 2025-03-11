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
    HEADER = '\033[95m'      # 亮紫色（Magenta），通常用于标题或重要提示
    OKBLUE = '\033[94m'      # 亮蓝色，用于正常信息或状态提示
    OKCYAN = '\033[96m'      # 亮青色（Cyan），用于正常信息或状态提示
    OKGREEN = '\033[92m'     # 亮绿色，通常用于表示成功或正常状态
    WARNING = '\033[93m'     # 亮黄色，用于警告信息
    ERROR = '\033[91m'       # 亮红色，用于错误信息
    BOLD = '\033[1m'         # 加粗文本（不改变颜色），使文本更突出
    ENDC = '\033[0m'     # 重置文本格式（包括颜色和加粗等），恢复默认显示

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
    def __init__(self, file_name, data_type="filterbank"):
        # 获取文件的绝对路径、文件名和扩展名
        self.file_abspath = os.path.abspath(file_name)
        self.file_nameonly = self.file_abspath.split("/")[-1]
        self.file_basename, self.file_extension = os.path.splitext(self.file_nameonly)
        self.file_buffer_copy = ""  # 初始化文件缓冲区副本

        if data_type == "filterbank":  
            print_log("\n正在读取filterbank文件....",color=colors.HEADER)
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
                print_log("警告：读取时出现值错误！可能是filterbank数据不是8位、16位或32位。尝试使用PRESTO的'readfile'获取必要信息...",color=colors.WARNING),print()

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
                    self.Tstart_MJD = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'MJD start time'").split("=")[-1].strip())
                    #self.Tstart_MJD = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'MJD start time (STT_\\*)'").split("=")[-1].strip())
                    self.freq_high_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'High channel (MHz)'").split("=")[-1].strip())
                    self.freq_low_MHz = np.float64(readfile_with_str(f"readfile {file_name}", "grep 'Low channel (MHz)'").split("=")[-1].strip())
                    self.freq_central_MHz = (self.freq_high_MHz + self.freq_low_MHz) / 2.0
                    print_log('readfile读取信息成功',color=colors.OKCYAN)
                    print_log(f"N_samples: {self.N_samples}")
                    print_log(f"t_samp_s: {self.t_samp_s}")
                    print_log(f"T_obs_s: {self.T_obs_s}",color=colors.BOLD)
                    print_log(f"nbits: {self.nbits}")
                    print_log(f"nchan: {self.nchan}")
                    print_log(f"chanbw_MHz: {self.chanbw_MHz}")
                    print_log(f"bw_MHz: {self.bw_MHz}",color=colors.BOLD)
                    print_log(f"Tstart_MJD: {self.Tstart_MJD}")
                    print_log(f"freq_high_MHz: {self.freq_high_MHz}")
                    print_log(f"freq_central_MHz: {self.freq_central_MHz}")
                    print_log(f"freq_low_MHz: {self.freq_low_MHz}")
               
                except:
                    print_log("警告：'readfile'失败。尝试使用'header'获取必要信息...",color=colors.WARNING)

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

                    print_log(f"N_samples: {self.N_samples}")
                    print_log(f"t_samp_s: {self.t_samp_s} s")
                    print_log(f"T_obs_s: {self.T_obs_s} s",color=colors.BOLD)
                    print_log(f"nbits: {self.nbits} bits")
                    print_log(f"nchan: {self.nchan} channels")
                    print_log(f"chanbw_MHz: {self.chanbw_MHz} MHz")
                    print_log(f"bw_MHz: {self.bw_MHz} MHz",color=colors.BOLD)
                    print_log(f"backend: {self.backend}")
                    print_log(f"Tstart_MJD: {self.Tstart_MJD}")
                    print_log(f"freq_high_MHz: {self.freq_high_MHz} MHz")
                    print_log(f"freq_central_MHz: {self.freq_central_MHz} MHz")
                    print_log(f"freq_low_MHz: {self.freq_low_MHz} MHz")

        if data_type == "psrfits":  # 处理PSRFITS文件
            print_log("\n正在读取PSRFITS文件....",color=colors.HEADER)
            if psrfits.is_PSRFITS(file_name):  # 检查文件是否为PSRFITS格式
                print_log("文件'%s'被正确识别为PSRFITS格式" % (file_name))
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
                print_log("\nReading PSRFITS (header only)....")
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

import ast
class SurveyConfiguration(object):
        def __init__(self, config_filename):
                self.config_filename = config_filename
                self.list_datafiles = []
                self.list_survey_configuration_ordered_params = ['OBSNAME','SEARCH_LABEL', 'DATA_TYPE', 'ROOT_WORKDIR', 'PRESTO', 'PRESTO_GPU','IF_DDPLAN', 'DM_MIN', 'DM_MAX','DM_STEP', 'DM_COHERENT_DEDISPERSION', 'N_SUBBANDS', 'PERIOD_TO_SEARCH_MIN', 'PERIOD_TO_SEARCH_MAX', 'LIST_SEGMENTS', 'RFIFIND_TIME', 'RFIFIND_CHANS_TO_ZAP', 'RFIFIND_TIME_INTERVALS_TO_ZAP', 'IGNORECHAN_LIST', 'ZAP_ISOLATED_PULSARS_FROM_FFTS', 'ZAP_ISOLATED_PULSARS_MAX_HARM', 'FLAG_ACCELERATION_SEARCH', 'ACCELSEARCH_LIST_ZMAX', 'ACCELSEARCH_NUMHARM', 'FLAG_JERK_SEARCH', 'JERKSEARCH_ZMAX', 'JERKSEARCH_WMAX', 'JERKSEARCH_NUMHARM', 'SIFTING_FLAG_REMOVE_DUPLICATES', 'SIFTING_FLAG_REMOVE_DM_PROBLEMS', 'SIFTING_FLAG_REMOVE_HARMONICS', 'SIFTING_MINIMUM_NUM_DMS', 'SIFTING_MINIMUM_DM', 'SIFTING_SIGMA_THRESHOLD', 'FLAG_FOLD_KNOWN_PULSARS', 'FLAG_FOLD_TIMESERIES', 'FLAG_FOLD_RAWDATA', 'RFIFIND_FLAGS', 'PREPDATA_FLAGS', 'PREPSUBBAND_FLAGS', 'REALFFT_FLAGS', 'REDNOISE_FLAGS', 'ACCELSEARCH_FLAGS', 'ACCELSEARCH_GPU_FLAGS', 'ACCELSEARCH_JERK_FLAGS', 'PREPFOLD_FLAGS', 'FLAG_SINGLEPULSE_SEARCH', 'SINGLEPULSE_SEARCH_FLAGS', 'USE_CUDA', 'CUDA_IDS', 'NUM_SIMULTANEOUS_JERKSEARCHES', 'NUM_SIMULTANEOUS_PREPFOLDS', 'NUM_SIMULTANEOUS_PREPSUBBANDS', 'MAX_SIMULTANEOUS_DMS_PER_PREPSUBBAND', 'FAST_BUFFER_DIR', 'FLAG_KEEP_DATA_IN_BUFFER_DIR', 'FLAG_REMOVE_FFTFILES', 'FLAG_REMOVE_DATFILES_OF_SEGMENTS', 'STEP_RFIFIND', 'STEP_ZAPLIST', 'STEP_DEDISPERSE', 'STEP_REALFFT', 'STEP_PERIODICITY_SEARCH', 'STEP_SIFTING', 'STEP_FOLDING', 'STEP_SINGLEPULSE_SEARCH']
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
                        elif key == "SEARCH_LABEL":                      self.search_label                     = self.dict_survey_configuration[key]
                        elif key == "DATA_TYPE":                         self.data_type                        = self.dict_survey_configuration[key]
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
                # 遍历有序参数列表并打印每个参数及其值
                important_param_list = ['OBSNAME','IF_DDPLAN','DM_MIN','DM_MAX','DM_STEP','PERIOD_TO_SEARCH_MIN','PERIOD_TO_SEARCH_MAX','LIST_SEGMENTS','ACCELSEARCH_LIST_ZMAX','FLAG_JERK_SEARCH','SIFTING_MINIMUM_NUM_DMS','FLAG_FOLD_TIMESERIES','PREPSUBBAND_FLAGS','PREPFOLD_FLAGS','FLAG_SINGLEPULSE_SEARCH']
                for param in important_param_list:
                        print("%-32s %s" % (param, self.dict_survey_configuration[param]))
                print()
                time.sleep(2)

print_program_message('start')
config_filename = "%s.cfg" % (os.path.basename(os.getcwd()))
config = SurveyConfiguration(config_filename)
obsname = config.obsname
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
                print(f"错误: 文件{f}不存在！可能是符号链接损坏。" ,color=colors.ERROR)
                exit()
            elif os.path.getsize(f) == 0:
                print(f"错误:文件{f}的大小为 0！" ,color=colors.ERROR)
                exit()
            config.folder_datafiles           = os.path.dirname(os.path.abspath(obsname)) 

config.list_datafiles_abspath = [os.path.join(config.folder_datafiles, x) for x in config.list_datafiles]
config.list_Observations = [Observation(x, config.data_type) for x in config.list_datafiles_abspath]
config.file_common_birdies = os.path.join(config.root_workdir, "common_birdies.txt")

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

list_segments_to_remove = []
for seg in config.list_segments_nofull:
        if (np.float64(seg)*60) >= config.list_Observations[0].T_obs_s:
                print_log(f"警告：段 {seg}m 的长度超过了完整观测的长度 ({config.list_Observations[0].T_obs_s / 60} 分钟)。将被忽略。",color=colors.WARNING)
                list_segments_to_remove.append(seg)
        elif (np.float64(seg)*60) >= 0.80 * config.list_Observations[0].T_obs_s:
                print_log(f"警告：段 {seg}m 的长度超过了完整观测长度的 80% ({config.list_Observations[0].T_obs_s / 60} 分钟)。将被忽略。",color=colors.WARNING)
                list_segments_to_remove.append(seg)
# 删除过长的段
for seg in list_segments_to_remove:
        config.list_segments_nofull.remove(seg)
        config.list_segments.remove(seg)

time.sleep(1)
config.print_configuration()
sifting.sigma_threshold = config.sifting_sigma_threshold
print_log("main:: SIFTING.sigma_threshold = ", sifting.sigma_threshold,color=colors.BOLD)
#添加全部文件叠加的总时间
LOG_dir = os.path.join(config.root_workdir, "LOG")

for i in range(len(config.list_Observations)):
        print_log(f"观测: {config.list_Observations[i].file_nameonly} ({config.list_Observations[i].T_obs_s:.2f} s)",color=colors.OKGREEN)
        if config.fast_buffer_dir != "":
                if os.path.exists(config.fast_buffer_dir):
                        file_buffer_abspath = os.path.join(config.fast_buffer_dir, config.list_Observations[i].file_nameonly)
                        if (not os.path.exists(file_buffer_abspath) or (os.path.getsize(file_buffer_abspath) != os.path.getsize(config.list_Observations[i].file_abspath))):
                                if (os.path.getsize(config.list_Observations[i].file_abspath) < shutil.disk_usage(config.fast_buffer_dir).free):
                                        print("\n正在将 '%s' 复制到快速缓冲目录 '%s'（这可能需要一些时间）..." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir), end=""); sys.stdout.flush()
                                        file_buffer_abspath = shutil.copy(config.list_Observations[i].file_abspath, config.fast_buffer_dir)                               
                                        config.list_Observations[i].file_abspath = file_buffer_abspath
                                        config.list_Observations[i].file_buffer_copy = file_buffer_abspath
                                        print("现在 config.list_Observations[i].file_abspath = ", config.list_Observations[i].file_abspath)
                                else:
                                        print_log(f"\n警告：快速缓冲目录 '{config.fast_buffer_dir}' 空间不足！",color=colors.WARNING)
                                        print("    -->  不使用快速缓冲目录。这可能导致处理速度变慢...")
                                        time.sleep(3)
                        else:
                                print("'%s' 的副本已存在于快速缓冲目录 '%s' 中。跳过..." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir))
                                file_buffer_abspath = os.path.join(config.fast_buffer_dir, config.list_Observations[i].file_nameonly)
                                config.list_Observations[i].file_abspath = file_buffer_abspath
                                config.list_Observations[i].file_buffer_copy = file_buffer_abspath
                                print("\n当前使用的观测文件为 '%s'。" % (config.list_Observations[i].file_abspath))

                else:
                        print_log(f"警告：快速缓冲目录 '{config.fast_buffer_dir}' 不存在！",color=colors.WARNING)
                        print("    -->  不使用快速缓冲目录。这可能导致处理速度变慢...")
                        config.fast_buffer_dir = ""
                        # time.sleep(10)

        print_log("\n****************搜索方案：****************\n",masks='搜索方案：',color=colors.HEADER)

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

                print_log(f"    段: {segment_label:8s}     ---> {N_chunks:2d} 块 ({', '.join(sorted(config.dict_search_structure[config.list_Observations[i].file_basename][segment_label].keys()))})",color=colors.BOLD)

                if fraction_left >= 0.80:
                        print_log(f" --> 警告：段 '{s}m' 的最后一个块实际上稍短一些 ({fraction_left * segment_length_min:.2f} 分钟)！",color=colors.WARNING)
                elif fraction_left > 0.10 and fraction_left < 0.80:
                        print_log(f"--> 警告：段 '{s}m' 的最后一个块 (ck{ck+1:02d}) 只有 {int(fraction_left * segment_length_min)} 分钟，将被忽略！",color=colors.WARNING)
                else:
                        print()

makedir(LOG_dir)

#ddplancmd = f'DDplan.py -d {maxDM} -n {Nchan} -b {BandWidth} -t {tsamp} -f {fcenter} -s {Nsub} -o DDplan.ps'
#DDplan.py -o ddplan_GBT_Lband_PSR -l 2.0 -d 100.0 -f 1400.0 -b 96.0 -n 96 -t 7.2e-05
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

print(list_DDplan_scheme)

print_log("\n****************检查磁盘空间****************\n",masks='检查磁盘空间',color=colors.HEADER)
num_DMs = 0 
for j in range(len(list_DDplan_scheme)):
        num_DMs = num_DMs + list_DDplan_scheme[j]['num_DMs']
        
flag_enough_disk_space = False
flag_enough_disk_space = check_if_enough_disk_space(config.root_workdir, num_DMs, config.list_Observations[i].T_obs_s, config.list_Observations[i].t_samp_s, config.list_segments_nofull, config.flag_remove_fftfiles, config.flag_remove_datfiles_of_segments)

# 如果磁盘空间不足，打印错误信息并退出程序
if flag_enough_disk_space == False:
        print_log(f"错误：磁盘空间不足！请释放空间或更改工作目录。",color=colors.ERROR)
        print("> 提示：为了最小化磁盘使用，请确保在配置文件中将 FLAG_REMOVE_FFTFILES 和 FLAG_REMOVE_DATFILES_OF_SEGMENTS 保留为默认值 1。")
        exit()

print_log("\n ====================STEP 1 - RFIFIND====================== \n",color=colors.HEADER)

rfifind_masks_dir = os.path.join(config.root_workdir, "01_RFIFIND")
makedir(rfifind_masks_dir)

for i in range(len(config.list_Observations)):
        time.sleep(0.2)
        config.list_Observations[i].mask = "%s/%s_rfifind.mask" % (rfifind_masks_dir, config.list_Observations[i].file_basename)

        flag_mask_present = check_rfifind_outfiles(rfifind_masks_dir, config.list_Observations[i].file_basename)

        # CASE 1: mask not present, STEP_RFIFIND = 0 
        if flag_mask_present == False and config.flag_step_rfifind == 0:
            # 如果掩模文件不存在且配置文件中 STEP_RFIFIND = 0，则提示错误并退出程序
            print_log(f"\n错误！掩模文件 '{config.list_Observations[i].mask}' 未找到，但 STEP_RFIFIND = 0！",color=colors.ERROR)
            print("请在配置文件中将 STEP_RFIFIND 设置为 1，或创建掩模文件并复制到 '01_RFIFIND' 目录中，然后重试。\n")
            exit()

        # CASE 2: mask not present, STEP_RFIFIND = 1
        if flag_mask_present == False and config.flag_step_rfifind == 1:
                LOG_basename = "01_rfifind_%s" % (config.list_Observations[i].file_nameonly)
                log_abspath = "%s/LOG_%s.txt" % (LOG_dir, LOG_basename)
                
                print_log("\n在 01_RFIFIND 文件夹中未找到掩模文件。将使用配置文件 '%s' 中指定的参数生成掩模文件。" % (config_filename),masks=config_filename,color=colors.BOLD)
                print_log(f"提示: 使用 'tail -f {log_abspath}' 查看 rfifind 的进度。",color=colors.OKCYAN)
                print("正在为观测 %3d/%d: '%s' 创建 rfifind 掩模文件...\n" % (i+1, len(config.list_Observations), config.list_Observations[i].file_nameonly), end=' ')
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
                                   )
        
        # CASE 3: mask is already present, STEP_RFIFIND = 1
        elif flag_mask_present == True and config.flag_step_rfifind == 1:
            print_log(f"\n很好！掩模文件 '{config.list_Observations[i].mask}' 已存在！不会创建新的掩模文件。",color=colors.OKBLUE)

        # 情况 4：掩模文件已存在，STEP_RFIFIND = 0
        elif flag_mask_present == True and config.flag_step_rfifind == 0:
            # 掩模文件已存在，但 STEP_RFIFIND = 0
            print_log(f"\n很好！掩模文件 '{config.list_Observations[i].mask}' 已存在！不会创建新的掩模文件。",color=colors.OKBLUE)
            print_log("警告：STEP_RFIFIND=0。将跳过该步骤，并信任找到的掩模文件是可用的。\n",color=colors.WARNING)
        # If STEP_RFIFIND = 1, check the mask before continuing 
        if  config.flag_step_rfifind == 1:
            print("正在检查被掩蔽的频带比例（这可能需要一些时间，具体取决于掩模文件的大小）...", end=' '); sys.stdout.flush()
            mask = rfifind.rfifind(config.list_Observations[i].mask)
            fraction_masked_channels = np.float64(len(mask.mask_zap_chans))/mask.nchan
        mask_str = f"{fraction_masked_channels * 100:.2f}"
        print_log(f"\nRFIFIND：被掩蔽的频率通道比例：{mask_str}%\n",masks=mask_str,color=colors.OKGREEN)

        if fraction_masked_channels > 0.5 and fraction_masked_channels <= 0.95:
            print_log(f"!!! 警告：{fraction_masked_channels * 100:.2f}% 的频带被掩蔽了！这似乎有点多 !!!",color=colors.WARNING)
            print("!!! 如果您认为太多，请尝试调整配置文件中的 RFIFIND 参数（例如增加 RFIFIND_FREQSIG）")

            time.sleep(1)

        if fraction_masked_channels > 0.95:
            print_log(f"!!! 错误：{fraction_masked_channels * 100:.2f}% 的频带被掩蔽了！这太多了 !!!",color=colors.ERROR)
            print("!!! 请调整配置文件中的 RFIFIND 参数，使被掩蔽的通道比例（可能）远小于 95%，然后重试。")
            exit()

        weights_file = config.list_Observations[i].mask.replace(".mask", ".weights")
        if os.path.exists(weights_file):
                array_weights = np.loadtxt(weights_file, unpack=True, usecols=(0, 1,), skiprows=1)
                config.ignorechan_list = ",".join([str(x) for x in np.where(array_weights[1] == 0)[0] ])
                config.nchan_ignored = len(config.ignorechan_list.split(","))
                print("\n\n找到 WEIGHTS 文件 '%s'。使用该文件忽略 %d 个通道，总共 %d 个通道（%.2f%%）" % (os.path.basename(weights_file), config.nchan_ignored, config.list_Observations[i].nchan, 100*config.nchan_ignored/np.float64(config.list_Observations[i].nchan)))
                print("被忽略的通道： %s" % (config.ignorechan_list))
                                

print_log("\n ====================STEP 2 - BIRDIES AND ZAPLIST   ====================== \n",color=colors.HEADER)

print("STEP_ZAPLIST = %s" % (config.flag_step_zaplist))

dir_birdies = os.path.join(config.root_workdir, "02_BIRDIES")
if config.flag_step_zaplist == 1:
        print_log("\n 02a) 使用掩模为每个文件创建一个 0-DM 质心时间序列。 \n",color=colors.HEADER)
        makedir(dir_birdies)
        for i in range(len(config.list_Observations)):
                time.sleep(0.1)
                print("\n正在运行 prepdata 为 \"%s\" 创建 0-DM 和质心时间序列..." % (config.list_Observations[i].file_nameonly), end=' ')
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
                          )
                print("完成！"); sys.stdout.flush()
                
        print_log("\n 02b) 对所有文件进行傅里叶变换。 \n",color=colors.HEADER)     

        config.list_0DM_datfiles = glob.glob("%s/*%s*.dat" % (dir_birdies, config.list_Observations[i].file_basename))   # 收集 02_BIRDIES_FOLDERS 中的 *.dat 文件
        for i in range(len(config.list_0DM_datfiles)):
                time.sleep(0.1)
                print("正在对 0-DM 质心时间序列 '%s' 运行 realfft..." % (os.path.basename(config.list_0DM_datfiles[i])), end=' ');sys.stdout.flush()
                LOG_basename = "02b_realfft_%s" % (os.path.basename(config.list_0DM_datfiles[i]))
                realfft(config.list_0DM_datfiles[i],
                        dir_birdies,
                        LOG_dir,
                        LOG_basename,
                        config.realfft_flags,
                        config.presto_env,
                        flag_LOG_append=0
                        )
                print("完成！");sys.stdout.flush()

        print_log("\n 02c) 去除红噪声。 \n",color=colors.HEADER)  

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
                         )

                print("完成！");sys.stdout.flush()

        print_log("\n 02d) 加速搜索和创建 zaplist。 \n",color=colors.HEADER)

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
                                                )
                print("完成！");sys.stdout.flush()

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

if config.if_ddplan == 1:
    dir_dedispersion = os.path.join(config.root_workdir, "03_ddsubbands")   
else:   
    dir_dedispersion = os.path.join(config.root_workdir, "03_subbands")

print_log("\n ==========STEP 3 - DEDISPERSION, DE-REDDENING AND PERIODICITY SEARCH========== \n",color=colors.HEADER)

LOG_basename = "03_prepsubband_and_search_FFT_%s" % (config.list_Observations[i].file_nameonly)
print("3) 去色散、去红噪声和周期性搜索：", end=' '); sys.stdout.flush()
makedir(dir_dedispersion)  # 创建去色散目录


if config.if_ddplan == 1:
    print_log("\n ====================DDplan去色散计划：  ====================== \n",color=colors.HEADER)
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
        print_log("提示：不会对完整长度的观测进行搜索！",color=colors.WARNING)
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
    print("\n**************************************************************")
    print("分段 %s 的 %s  -- 块 %s 的 %s" % (seg, sorted(config.dict_search_structure[obs].keys()), ck, sorted(config.dict_search_structure[obs][seg].keys())))
    print("**************************************************************")
    work_dir_chunk = os.path.join(work_dir_segment, ck)
    print("3) 去色散、去红化和周期性搜索：正在创建工作目录 '%s'..." % (work_dir_chunk), end=' '); sys.stdout.flush()
    makedir(work_dir_chunk)
    print("完成!"); sys.stdout.flush()

    zapfile = "%s/%s_DM00.00.zaplist" % (dir_birdies, config.list_Observations[i].file_basename)

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
                                                    dict_flag_steps)


# if config.flag_step_sifting == 1:
#         print()
#         print("##################################################################################################")
#         print("#                                  STEP 4 - CANDIDATE SIFTING ")
#         print("##################################################################################################")

#         dir_sifting = os.path.join(config.root_workdir, "04_SIFTING")
#         print("4) 候选体筛选：正在创建工作目录...", end=' '); sys.stdout.flush()
#         makedir(dir_sifting)
#         print("完成！")

#         dict_candidate_lists = {}

#         for i in range(len(config.list_Observations)):
#                 obs = config.list_Observations[i].file_basename
#                 print("Sifting candidates for observation %3d/%d '%s'." % (i+1, len(config.list_Observations), obs)) 
#                 for seg in sorted(config.dict_search_structure[obs].keys()):
#                         work_dir_segment = os.path.join(dir_sifting, config.list_Observations[i].file_basename, "%s" % seg)
#                         makedir(work_dir_segment)

#                         for ck in sorted(config.dict_search_structure[obs][seg].keys()):
#                                 work_dir_chunk = os.path.join(work_dir_segment, ck)
#                                 makedir(work_dir_chunk)
#                                 LOG_basename = "04_sifting_%s_%s_%s" % (obs, seg, ck)
#                                 work_dir_candidate_sifting = os.path.join(dir_sifting, obs, seg, ck)

#                                 print("4) CANDIDATE SIFTING: Creating working directory '%s'..." % (work_dir_candidate_sifting), end=' '); sys.stdout.flush()
#                                 makedir(work_dir_candidate_sifting)
#                                 print("done!")
#                                 print("4) CANDIDATE SIFTING: Sifting observation %d) \"%s\" / %s / %s..." % (i+1, obs, seg, ck), end=' ')
#                                 sys.stdout.flush()

#                                 config.dict_search_structure[obs][seg][ck]['candidates'] = sift_candidates(work_dir_chunk,
#                                                                                                             LOG_dir,
#                                                                                                             LOG_basename,
#                                                                                                             dir_dedispersion,
#                                                                                                             obs,
#                                                                                                             seg,
#                                                                                                             ck,
#                                                                                                             config.accelsearch_list_zmax,
#                                                                                                             config.jerksearch_zmax,
#                                                                                                             config.jerksearch_wmax,
#                                                                                                             config.sifting_flag_remove_duplicates,
#                                                                                                             config.sifting_flag_remove_dm_problems,
#                                                                                                             config.sifting_flag_remove_harmonics,
#                                                                                                             config.sifting_minimum_num_DMs,
#                                                                                                             config.sifting_minimum_DM,
#                                                                                                             config.period_to_search_min,
#                                                                                                             config.period_to_search_max
#                                 )

#         for i in range(len(config.list_Observations)):
#             # 构造候选体汇总文件的路径和文件名
#             candidates_summary_filename = "%s/%s_cands.summary" % (dir_sifting, config.list_Observations[i].file_basename)
#             candidates_summary_file = open(candidates_summary_filename, 'w')

#             # 初始化需要折叠的候选体总数
#             count_candidates_to_fold_all = 0
#             # 写入文件分隔符
#             candidates_summary_file.write("\n*****************************************************************")
#             # 写入当前观测文件中找到的候选体信息
#             candidates_summary_file.write("\n在 %s 中找到的候选体：\n\n" % (config.list_Observations[i].file_nameonly))
#             # 遍历所有段和块，统计候选体数量
#             for seg in sorted(config.dict_search_structure[obs].keys()):
#                 for ck in sorted(config.dict_search_structure[obs][seg].keys()):
#                     Ncands_seg_ck = len(config.dict_search_structure[obs][seg][ck]['candidates'])
#                     # 写入每个段和块的候选体数量
#                     candidates_summary_file.write("%20s  |  %10s  ---> %4d 候选体\n" % (seg, ck, Ncands_seg_ck))
#                     count_candidates_to_fold_all = count_candidates_to_fold_all + Ncands_seg_ck
#             # 写入总候选体数量
#             candidates_summary_file.write("\n总计 = %d 候选体\n" % (count_candidates_to_fold_all))
#             candidates_summary_file.write("*****************************************************************\n\n")

#             count_candidates_to_fold_redet = 0
#             count_candidates_to_fold_new = 0
#             list_all_cands = []
#             for seg in sorted(config.dict_search_structure[obs].keys()):
#                     for ck in sorted(config.dict_search_structure[obs][seg].keys()):
#                             config.dict_search_structure[obs][seg][ck]['candidates_redetections'] = []
#                             config.dict_search_structure[obs][seg][ck]['candidates_new'] = []

#                             for j in range(len(config.dict_search_structure[obs][seg][ck]['candidates'])):
#                                     candidate = config.dict_search_structure[obs][seg][ck]['candidates'][j]

#                                     flag_is_know, known_psrname, str_harmonic = check_if_cand_is_known(candidate, list_known_pulsars, numharm=16)

#                                     if flag_is_know == True:
#                                             config.dict_search_structure[obs][seg][ck]['candidates_redetections'].append(candidate)
#                                             count_candidates_to_fold_redet = count_candidates_to_fold_redet + 1
#                                     elif flag_is_know == False:
#                                             config.dict_search_structure[obs][seg][ck]['candidates_new'].append(candidate)
#                                             count_candidates_to_fold_new = count_candidates_to_fold_new + 1

#                                     dict_cand = {'cand': candidate, 'obs': obs, 'seg': seg, 'ck': ck, 'is_known': flag_is_know, 'known_psrname': known_psrname, 'str_harmonic': str_harmonic}
#                                     list_all_cands.append(dict_cand)
#             N_cands_all = len(list_all_cands)

#             for i_cand, dict_cand in zip(list(range(0, N_cands_all)), sorted(list_all_cands, key=lambda k: k['cand'].p, reverse=False)):
#                     if dict_cand['cand'].DM < 2:
#                             candidates_summary_file.write("Cand %4d/%d: %12.6f ms    |  DM: %7.2f pc cm-3    (%4s / %4s | sigma: %5.2f)  ---> Likely RFI\n" % (i_cand+1, N_cands_all, dict_cand['cand'].p * 1000., dict_cand['cand'].DM, dict_cand['seg'], dict_cand['ck'], dict_cand['cand'].sigma))
#                     else:
#                             if dict_cand['is_known'] == True:
#                                     candidates_summary_file.write("Cand %4d/%d:  %12.6f ms  |  DM: %7.2f pc cm-3    (%4s / %4s | sigma: %5.2f)  ---> Likely %s - %s\n" % (i_cand+1, N_cands_all, dict_cand['cand'].p * 1000., dict_cand['cand'].DM, dict_cand['seg'], dict_cand['ck'], dict_cand['cand'].sigma, dict_cand['known_psrname'], dict_cand['str_harmonic']))
#                             elif dict_cand['is_known'] == False:
#                                     candidates_summary_file.write("Cand %4d/%d:  %12.6f ms  |  DM: %7.2f pc cm-3    (%4s / %4s | sigma: %5.2f)\n" % (i_cand+1, N_cands_all, dict_cand['cand'].p * 1000., dict_cand['cand'].DM, dict_cand['seg'], dict_cand['ck'], dict_cand['cand'].sigma))

#             candidates_summary_file.close()

#             candidates_summary_file = open(candidates_summary_filename, 'r')
#             for line in candidates_summary_file:
#                     print(line, end=' ')
#             candidates_summary_file.close()


# if config.flag_step_folding == 1:
#         print()
#         print()
#         print("##################################################################################################")
#         print("#                                        STEP 5 - FOLDING ")
#         print("##################################################################################################")
#         print()

#         dir_folding = os.path.join(config.root_workdir, "05_FOLDING")
#         print("5) 折叠：正在创建工作目录...", end=' '); sys.stdout.flush()
#         if not os.path.exists(dir_folding):
#                 os.mkdir(dir_folding)
#         print("完成！")

#         for i in range(len(config.list_Observations)):
#                 obs = config.list_Observations[i].file_basename
#                 print("正在折叠观测 '%s'" % (obs))
#                 print()

#                 work_dir_candidate_folding = os.path.join(dir_folding, config.list_Observations[i].file_basename)
#                 print("5) 候选体折叠：正在创建工作目录 '%s'..." % (work_dir_candidate_folding), end=' '); sys.stdout.flush()
#                 if not os.path.exists(work_dir_candidate_folding):
#                         os.mkdir(work_dir_candidate_folding)
#                 print("完成！")

#                 file_script_fold_name = "script_fold.txt"
#                 file_script_fold_abspath = "%s/%s" % (work_dir_candidate_folding, file_script_fold_name)
#                 file_script_fold = open(file_script_fold_abspath, "w")
#                 file_script_fold.close()

#                 if config.flag_fold_known_pulsars == 1:
#                         key_cands_to_fold = 'candidates'

#                         print()
#                         print("5) 候选体折叠：我将折叠所有 %d 个候选体（包括 %s 个可能是重复检测的候选体）" % (N_cands_all, count_candidates_to_fold_redet))
#                         N_cands_to_fold = N_cands_all

#                 elif config.flag_fold_known_pulsars == 0:
#                         key_cands_to_fold = 'candidates_new'
#                         print()
#                         print("5) 候选体折叠：我将仅折叠 %d 个可能是新脉冲星的候选体（%s 个可能是重复检测的候选体将不被折叠）" % (count_candidates_to_fold_new, count_candidates_to_fold_redet))
#                         N_cands_to_fold = count_candidates_to_fold_new
#                 count_folded_ts = 1
#                 if config.flag_fold_timeseries == 1:

#                         LOG_basename = "05_folding_%s_timeseries" % (obs)
#                         print()
#                         print("正在折叠时序数据...")
#                         print()
#                         print("\033[1m >> 提示:\033[0m 使用 '\033[1mtail -f %s/LOG_%s.txt\033[0m' 查看折叠进度" % (LOG_dir, LOG_basename))
#                         print()
#                         for seg in sorted(config.dict_search_structure[obs].keys()):
#                                 for ck in sorted(config.dict_search_structure[obs][seg].keys()):
#                                         for j in range(len(config.dict_search_structure[obs][seg][ck][key_cands_to_fold])):
#                                                 candidate = config.dict_search_structure[obs][seg][ck][key_cands_to_fold][j]

#                                                 print("正在折叠候选体时序数据 %d/%d 的 %s: 段 %s / %s..." % (count_folded_ts, N_cands_to_fold, obs, seg, ck), end=' ')
#                                                 sys.stdout.flush()

#                                                 tstart_folding_cand_ts = time.time()
#                                                 file_to_fold = os.path.join(dir_dedispersion, obs, seg, ck, candidate.filename.split("_ACCEL")[0] + ".dat")
#                                                 flag_remove_dat_after_folding = 0
#                                                 if os.path.exists(file_to_fold):

#                                                         fold_candidate(work_dir_candidate_folding,
#                                                                 LOG_dir,
#                                                                 LOG_basename,
#                                                                 config.list_Observations[i],
#                                                                 dir_dedispersion,
#                                                                 obs,
#                                                                 seg,
#                                                                 ck,
#                                                                 candidate,
#                                                                 config.ignorechan_list,
#                                                                 config.prepfold_flags,
#                                                                 config.presto_env,
#                                                                 1,
#                                                                 "timeseries",
#                                                                config.num_simultaneous_prepfolds
#                                                         )

#                                                         tend_folding_cand_ts = time.time()
#                                                         time_taken_folding_cand_ts_s = tend_folding_cand_ts - tstart_folding_cand_ts
#                                                         print("done in %.2f s!" % (time_taken_folding_cand_ts_s))
#                                                         sys.stdout.flush()
#                                                         count_folded_ts = count_folded_ts + 1
#                                                 else:
#                                                         print("dat文件不存在！可能是因为你在配置文件中设置了FLAG_REMOVE_DATFILES_OF_SEGMENTS = 1。跳过...")
#                 count_folded_raw = 1
#                 if config.flag_fold_rawdata == 1:
#                         LOG_basename = "05_folding_%s_rawdata" % (obs)
#                         print()
#                         print("正在折叠原始数据 \033[1m >> 提示:\033[0m 使用 '\033[1mtail -f %s/LOG_%s.txt\033[0m' 查看折叠进度" % (LOG_dir, LOG_basename))
#                         for seg in sorted(list(config.dict_search_structure[obs].keys()), reverse=True):
#                                 for ck in sorted(config.dict_search_structure[obs][seg].keys()):
#                                         for j in range(len(config.dict_search_structure[obs][seg][ck][key_cands_to_fold])):
#                                                 candidate = config.dict_search_structure[obs][seg][ck][key_cands_to_fold][j]
#                                                 LOG_basename = "05_folding_%s_%s_%s_rawdata" % (obs, seg, ck)

#                                                 fold_candidate(work_dir_candidate_folding,
#                                                                 LOG_dir,
#                                                                 LOG_basename,
#                                                                 config.list_Observations[i],
#                                                                 dir_dedispersion,
#                                                                 obs,
#                                                                 seg,
#                                                                 ck,
#                                                                 candidate,
#                                                                 config.ignorechan_list,
#                                                                 config.prepfold_flags,
#                                                                 config.presto_env,
#                                                                 1,
#                                                                 "rawdata",
#                                                                config.num_simultaneous_prepfolds
#                                                 )

#                                                 count_folded_raw = count_folded_raw + 1

#                 os.chdir(work_dir_candidate_folding)
#                 cmd_pm_run_multithread = "%spm_run_multithread -cmdfile %s -ncpus %d" % (os.path.dirname(sys.argv[0]), file_script_fold_abspath, config.num_simultaneous_prepfolds)
#                 print()
#                 print()
#                 print("5) CANDIDATE FOLDING - Now running:")
#                 print("%s" % cmd_pm_run_multithread)
#                 run_cmd(cmd_pm_run_multithread)

# if config.flag_singlepulse_search == 1 and config.flag_step_singlepulse_search == 1:
#         print()
#         print()
#         print("##################################################################################################")
#         print("#                                        STEP 6 - SINGLE-PULSE SEARCH (PRESTO) ")
#         print("##################################################################################################")
#         print()

#         dir_singlepulse_search = os.path.join(config.root_workdir, "06_SINGLEPULSE")
#         if verbosity_level >= 1:
#                 print("6) 单脉冲搜索：正在创建工作目录...", end=' '); sys.stdout.flush()
#         if not os.path.exists(dir_singlepulse_search):
#                 os.mkdir(dir_singlepulse_search)

#         for i in range(len(config.list_Observations)):
#                 obs = config.list_Observations[i].file_basename
#                 time.sleep(1.0)
#                 work_dir_singlepulse_search_obs = os.path.join(dir_singlepulse_search, config.list_Observations[i].file_basename)
#                 if verbosity_level >= 2:
#                         print("6) 单脉冲搜索：正在创建工作目录 '%s'..." % (work_dir_singlepulse_search_obs), end=' '); sys.stdout.flush()
#                 if not os.path.exists(work_dir_singlepulse_search_obs):
#                         os.mkdir(work_dir_singlepulse_search_obs)
#                 if verbosity_level >= 2:
#                         print("完成！"); sys.stdout.flush()

#         if verbosity_level >= 1:
#                 print("完成！")


#         # Go into the 06_SINGLEPULSE directory
#         os.chdir(work_dir_singlepulse_search_obs)

#         # Create symbolic links to all the full-length *.dat and corresponding *.inf files
#         search_string_datfiles_full_length = "%s/03_DEDISPERSION/%s/full/ck00/*.dat" % (config.root_workdir, config.list_Observations[0].file_basename) #List of datfiles
#         search_string_inffiles_full_length = "%s/03_DEDISPERSION/%s/full/ck00/*.inf" % (config.root_workdir, config.list_Observations[0].file_basename) #List of inffiles
#         list_datfiles_full_length = glob.glob(search_string_datfiles_full_length)
#         list_inffiles_full_length = glob.glob(search_string_inffiles_full_length)

#         for f in list_datfiles_full_length + list_inffiles_full_length:
#                 symlink_filename = os.path.basename(f)
#                 if os.path.exists(symlink_filename) and os.path.islink(symlink_filename):
#                         print("Symlink %s already exists. Skipping..." % (symlink_filename))
#                 else:
#                         print("Making symbolic link of '%s'..." % (symlink_filename), end=''); sys.stdout.flush()
#                         os.symlink(f, symlink_filename)
#                         print("done!"); sys.stdout.flush()

#         LOG_singlepulse_search_basename = "06_singlepulse_search_%s" % (config.list_Observations[0].file_basename)
#         LOG_singlepulse_search_abspath  = "%s/LOG_%s.txt" % (LOG_dir, LOG_singlepulse_search_basename)
        
#         list_datfiles_to_singlepulse_search = glob.glob("%s/*.dat" % work_dir_singlepulse_search_obs)


#         singlepulse_search(work_dir_singlepulse_search_obs,
#                            LOG_dir,
#                            LOG_singlepulse_search_basename,
#                            list_datfiles_to_singlepulse_search,
#                            config.singlepulse_search_flags,
#                            config.num_simultaneous_singlepulse_searches,
#                            config.presto_env,
#                            verbosity_level,
#                            config.flag_step_singlepulse_search)

                
# if config.list_Observations[i].file_buffer_copy != "":
#         if config.flag_keep_data_in_buffer_dir == 1:
#                 print()
#                 print("Keeping a copy of '%s' from the buffer directory (%s)." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir))
#                 print("Remember to delete if you are not using it further.")
#         else:
#                 print("Removing copy of '%s' from the buffer directory (%s)..." % (config.list_Observations[i].file_nameonly, config.fast_buffer_dir), end=""), ; sys.stdout.flush()
#                 os.remove(config.list_Observations[i].file_buffer_copy)
#                 print("done!")

# print()