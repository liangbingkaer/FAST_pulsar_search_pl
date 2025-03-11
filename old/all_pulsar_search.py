# Written by penglong in 2023.11.11
#qq:2107053791

# 导入库
import os,sys
import numpy as np
from psr_fuc import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools

# 获取当前工作目录与核数
dir_a = os.getcwd()
num_processes = 20
cpu_num_str = str(num_processes) + '核'
print('全部核数：'+str(cpu_count())+'\n'+'使用核数：'+str(num_processes))
t_start = time.time()

# 参数设置：决定了折叠方案（非常重要）
rootname,filename = 'aqlx-1'    ,'*.fits'         # 决定折叠方案使用的源名称,源文件
open1 = 1                    # 1表示对fits'折叠，0表示对dat折叠
open2 = 1                    # 是否去除红噪声，1表示是，0表示否
open3 = 1                    # 1表示使用ddpan去色散方案，0表示使用自定义色散计划
open4 = 0                    # 是否进行单脉冲搜寻，1表示是，0表示否
open5 = 0                    # 是否进行质心修正
open6 = 0                    # 需和open5同时为true，需要配置pysolator，需双星完整参数星历表addmass.par
ra = '17:20:54.5063 '
dec = '-05:34:23.81556 '
# 忽略的通道(全流程使用)，使用请正确设置-ignorechan开头的字符串
ignorechan = '-ignorechan 680:810'  

# 去色散相关参数
maxDM = 500                 # ddpan的最大DM搜索范围
timebin = 0.1               # 建议0.1，无需修改
ddpl = [(20, 30, 0.1), (30, 34, 0.2)]             # 自定义去色散计划

#加速搜寻相关参数
zmax = 200                    # 加速a值
wmax = 0                    # 加速加速w值，不使用请设置为0  
Internal_memory = '-inmem '     #-inmem

#折叠相关参数
foldnum = 100                # 折图数量（按信噪比排序）
Nsub = 128                  # 子带数
foldway = '-nosearch'       # 折叠方式使用默认值
Nint = 128                  # 轮廓点数

#判断
fits_or_dats = "fits" if open1 == 1 else "dats"
ignorechan = ignorechan if ignorechan.startswith('-ignorechan') else ''
outmask = f'rfi{timebin}s'
if ignorechan != '':
    outmask = f'ig_rfi{timebin}s'
    pstart_str = 'IG' 
    dir_ig = os.path.join(dir_a, "IG")
    mkdir(dir_ig,1)
else:
    pstart_str='A'
maskname = f'{outmask}_rfifind.mask'

suba, bar = ('', ' ') if open5 == 1 else ('-nobary', ' -topo')
zmax_str = ' -zmax '+str(zmax)
wmax = ((wmax // 20) + 1) * 20 if wmax != 0 else 0
wmax_str = f' -wmax {wmax}' if wmax != 0 else '' 
candname = '_ACCEL_Cand_' if wmax == 0 else '_JERK_Cand_'

# 打印相关信息
print_program_message('start')
print_log(' ====================注意： ====================== \n')
print_log('源名为：' + rootname,masks=rootname)
open_id =str(open1) + str(open2) + str(open3) + str(open4) + str(open5) 
time_log( 'open=' +open_id,masks=open_id)
print_log('当前运行路径为：' + dir_a + '\n')
target_type = f'计划一共折叠{foldnum}张图'
print_log(f'对{fits_or_dats}进行 {foldway} 折叠: {target_type} zmax:{zmax} wmax:{wmax}',masks=[fits_or_dats,foldway,target_type,zmax,wmax])
print_log('不去除红噪声' if open2 != 1 else '去除红噪声',masks='不')
# DDplan参数设置
alldm_list = []
prepsubbandcmd_all = []
if open3 == 1:
    dir_i = os.path.join(dir_a, "ddifok")
    print_log(f'将使用DDplan. 其中maxDM = {maxDM}\n',masks =str(maxDM) )
else:
    dir_i = os.path.join(dir_a, "ifok")
    print_log('\n注意：将使用用户自己设置的dm值进行消色散，参数如下:\n')
    dm_times = 0
    for ddpl_value in ddpl:
        dm_times += 1
        loodm, highdm, ddm = ddpl_value
        ndms = int((highdm - loodm) // ddm)
        dm_list = [format(loodm + i * ddm, '.2f') for i in range(ndms)]
        alldm_list.extend(dm_list)
        maxDM = max(alldm_list)
        print_log(f'第{dm_times}次去色散参数为： 初始dm为: {loodm}  最大dm为: {highdm}  步长为: {ddm}  次数为：{ndms}\n')
        prepsubbandcmd=f'prepsubband {suba} {ignorechan} -nsub {Nsub} -lodm {loodm}  -dmstep {ddm}  -numdms {ndms}  -mask ../{maskname}  -o {rootname}  ../{filename} ' 
        prepsubbandcmd_all.append(prepsubbandcmd)
# 单脉冲搜寻
if open4 != 1:
    print_log('不进行单脉冲搜寻\n',masks='不')
else:
    dir_single = os.path.join(dir_a, 'all_single')
    if not os.path.exists(dir_single):
        os.mkdir(dir_single)
    print_log('顺便进行单脉冲搜寻\n')
   

#判断质心是否修正?

baryif = '_bary' if open5 == 1 else ''
print_log(f'进行质心修正\n 注意： ra = {ra}  dec = {dec} \n') if open5 == 1 else print_log('不进行质心修正',masks='不')
if ignorechan != '':
    print_log(f'\n注意：将全程忽略以下通道：{ignorechan}\n',masks=ignorechan)

# 线程和折点设置
print_log(f'线程为: {Nsub}    折点数: {Nint}\n')

#判断用户输入：
print_log("请运行前仔细检查参数，进行必要的修改：\n")
user_input = input( "不更改参数可通过移除对应ok.txt文件重新运行对应步骤\n"
                   "默认参数如上，注意检查\n"
                   "确认执行程序请输入 ***源名*** ：")
print_log('用户已输入： ' + user_input)
time.sleep(1)  # 等待1秒
process_user_input(user_input,rootname)

#确认文件名
pngdir = os.path.join(dir_a, "png")
dir_s_type = "ddsubbands" if open3 == 1 else "subbands"
dir_s = os.path.join(dir_a, dir_s_type)

redway = '_nored' if open2 != 1 else '_red'
mask_only = f"{dir_s_type[:2]}{str(maxDM)}_{fits_or_dats}_{foldway[1:]}_{str(timebin)}s_{redway}_{zmax}_{wmax}{baryif}"
png_dir = os.path.join(pngdir,mask_only)

os.makedirs(dir_s, exist_ok=True)
print_log(f"后续操作将在文件夹 {dir_s} 内进行")
os.makedirs(png_dir, exist_ok=True)
print_log(f"生成的图片将放入文件夹 {png_dir}")


dir_single = os.path.join(dir_a, "all_single")
png_single = os.path.join(pngdir, "single")

mkdir(dir_i,1)
mkdir(dir_single,open4)
mkdir(png_single,open4)

#需要参数：进程池数，总进程名，dm列表，dm转为为cmd的函数
def pool(num_processes,name,list,cmd_fuction):
    n = 0
    pbar = tqdm(total=len(list), desc=cpu_num_str+name, unit="dm")
    update = lambda *args: pbar.update()
    with Pool(num_processes) as p:
        results = []
        for dm in list:
            n += 1
            # 创建带有部分参数的函数
            print(f'多线程第{n}次运行')
            cmd_partial = functools.partial(cmd_fuction, dm,n)
            cmd, ifok = cmd_partial()  # 调用部分参数的函数
            pbar.set_description(f'{cpu_num_str}  {name} - Processing: {dm}')
            result = p.apply_async(child_task, args=(cmd,ifok,), callback=lambda _: pbar.update(), error_callback=print_error)
            results.append(result)
        p.close()
        p.join()   
    for result in results:
        result.get()
    pbar.close()

def child_task(cmd,ifok):
    start = time.time()
    run_cmd(cmd,ifok,dir=dir_s,start_time=start)
    

def print_error(value):
    print("error: ", value)

#正式rfi
print_log('\n ========================RFI  ========================== \n')
print_log(f'timebin值为: {timebin}')

rfi_json_path = os.path.join(dir_a, outmask+'.json')
mask_file = os.path.join(dir_a, maskname)

arficmd = f'rfifind {ignorechan} -time {timebin} -psrfits -noscales -nooffsets -o {outmask} {filename}'
rfi(arficmd, mask_file,rfi_json_path)


rfi_ps = maskname[:-5]+'.ps'
rfi_png = maskname[:-5]+'.png'
convert_ps_to_png(rfi_ps,rfi_png)

with open(rfi_json_path, 'r') as file:
    header = json.load(file)

print_log('\n ====================打印文件信息  ====================== \n')
print_log('\n'.join([f'{key}: {value}' for key, value in header.items()]))

total_time = header['Total time (s)']
total_points = header['Total points (N)']
tsamp = float(header['Sample time (s)'])
timeinf = f'Total time (s): {total_time}'
pointsinf = f'Total_points: {total_points}'
print_log(f'\n{timeinf}\n{pointsinf}\n')

#readfile:get BandWidth
print_log('''\n================readfile inf=================== \n''')
fitslist_names = [file for file in os.listdir(dir_a) if file.endswith('.fits')]
with open('fitslist.txt', 'w') as file:
    file.write(rootname+'\n')
    file.write('\n'.join(fitslist_names))
with open('fitslist.txt', 'r') as file:
    lines = file.readlines()
    line = lines[1]
    readfilecmd='readfile '+line
    output = subprocess.getoutput(readfilecmd)
    print(output)
    header1 = {}
    for line in output.split('\n'):
        items = line.split("=")
        if len(items) > 1:
            header1[items[0].strip()] = items[1].strip()
    BandWidth = header1.get('Total Bandwidth (MHz)')
    print_log('BandWidth:',BandWidth)

# ddplan
prep_cmd = os.path.join(dir_a,'prep.sh')
os.chdir(dir_a)
if open3 == 1:
    print_log('''\n============Generate Dedispersion Plan===============\n''') 
    ddplan_log = os.path.join(dir_a, 'ddplan.txt')
    Nchan = int(header['Num of channels'])
    fcenter = float(header['Center freq (MHz)'])
    Nsamp = int(header['Total points (N)'])
    print_log(f'#ddplan设置的最大的DM为 {maxDM}\n')
    ddplancmd = f'DDplan.py -d {maxDM} -n {Nchan} -b {BandWidth} -t {tsamp} -f {fcenter} -s {Nsub} -o DDplan.ps'
    ddplanout = subprocess.getoutput(ddplancmd)
    print_log(ddplanout)
    with open(ddplan_log, 'a') as logfile:
        logfile.write(f'#ddplan设置的最大的DM为 {maxDM}\n')
        logfile.write(ddplancmd+'\n')
        logfile.write(ddplanout+'\n')
    planlist = ddplanout.split('\n')
    planlist.reverse()

    ddplan = []
    for plan in planlist:
        if plan == '':
            continue
        elif plan.strip().startswith('Low DM'):
            break
        else:
            ddplan.append(plan)
    ddplan.reverse()
    for line in ddplan:
        dms = np.array([])
        ddpl = line.split()
        lowDM = float(ddpl[0])
        hiDM = float(ddpl[1])
        dDM = float(ddpl[2])
        DownSamp=1             #DownSamp = int(ddpl[3])
        NDMs = int(ddpl[6])
        calls = int(ddpl[7])
        Nout = Nsamp/DownSamp 
        Nout -= (Nout % 500)
        Nout = int(Nout)
        dmlist = np.array_split(np.arange(lowDM, hiDM, dDM), calls)
        dm_list = np.concatenate((dms,np.arange(lowDM,hiDM,dDM )), axis=0)
        alldm_list =np.concatenate((alldm_list,dm_list),axis=0)
        for dml in dmlist:
            lodm = dml[0]
            lodm = '{:.2f}'.format(lodm)
            prepsubbandcmd = f"prepsubband {suba} {ignorechan} -nsub {Nsub} -lodm {lodm} -dmstep {dDM} -numdms {NDMs} -numout {Nout} -downsamp {DownSamp} -mask ../{maskname} -o {rootname} ../{filename}"
            prepsubbandcmd_all.append(prepsubbandcmd)
        #print(prepsubbandcmd_all)
    #ps文件转换为png
    alldm_list = [f"{np.around(num, decimals=2):.2f}" for num in alldm_list]
    ddplan_ps_file = "DDplan.ps"
    ddplan_png_file =rootname+"_DDplan.png"
    convert_ps_to_png(ddplan_ps_file, ddplan_png_file)
    copy_files_with_name(dir_a, png_dir,ddplan_png_file)

#ddplan去色散
def ddprepsubband_cmd(cmd,n):
    ifok = os.path.join(dir_i,f'dd-{n}')
    return cmd ,ifok

if open3 == 1:
    ifok = os.path.join(dir_a, 'ok-1dd.txt') 
    aa =len(prepsubbandcmd_all)
    if not os.path.isfile(ifok):
        F_start_time = time.time()
        with open(prep_cmd, 'a') as logfile:
            for i in prepsubbandcmd_all:
                logfile.write(i+'\n')
        if aa < num_processes:
            a1,tall= 0,0
            for cmd in prepsubbandcmd_all:
                t1 = time.time()
                a1 += 1
                print_log(f'第{a1}次去色散',masks=str(a1))
                oklodm = os.path.join(dir_i,f'dd{a1}')
                run_cmd(cmd,oklodm,dir = dir_s, start_time=t1)
                t2 = time.time()
                dt = t2 -t1
                tall += dt
                display_progress_with_time(a1,aa,tall)
        else:
            pool(num_processes,'ddprepsubband',prepsubbandcmd_all,ddprepsubband_cmd)
        all_run_time(F_start_time,'ddplan去色散')
        os.system('touch '+ifok)
        shutil.copy2(ifok, png_dir)


os.chdir(dir_a)

#自定义去色散计划
def prepsubband_cmd(cmd,n):
    ifok = os.path.join(dir_i,f'prep{n}')
    return cmd ,ifok

if open3 != 1:
    ifok = os.path.join(dir_a, 'ok-1prepsubband.txt') 
    if not os.path.isfile(ifok):
        F_start_time = time.time()
        with open(prep_cmd, 'a') as logfile:
            for i in prepsubbandcmd_all:
                logfile.write(i+'\n')
        if len(prepsubbandcmd_all) < num_processes:
            times = 0
            for cmd in prepsubbandcmd_all:
                start_time = time.time()
                times += 1
                n = str(times)
                print_log(f'第{n}次去色散',masks=n)
                oklodm = dir_i+'/'+'prep'+str(n)+'.txt'
                run_cmd(cmd,oklodm,dir = dir_s, start_time=start_time)
                os.system('touch '+ifok)
                'touch '+ifok
        else:
            pool(num_processes,'prepsubband',prepsubbandcmd_all,prepsubband_cmd)
        all_run_time(F_start_time,'自定义去色散')
os.chdir(dir_a)



#prepdata质心修正
# 打开文件并读取内容
dir_data = os.path.join(dir_a,'data')
dir_orb = os.path.join(dir_a,'de_orb')
mkdir(dir_data,open5 )
mkdir(dir_orb ,open6 )

if open5 == 1:
    dir_i = os.path.join(dir_a, "bary_ifok")
    mkdir(dir_i,1)
    os.chdir(dir_s)
    rootname = 'bary_'+rootname
    dat_names = sorted([file for file in os.listdir(dir_s) if file.endswith('.dat')])
    dms = [filename.split('_')[1].replace('DM', '')[:-4] for filename in dat_names]
    inf_names = sorted([file for file in os.listdir(dir_s) if file.endswith('.inf') and not file.endswith('_red.inf')])
    print_log('''\n ==================== prepdata预质心修正  ====================== \n''')
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
    os.chdir(dir_a)
    print(f'ra={ra},dec={dec}')
    print_log('''\n ==================== ra,dec修正完毕  ====================== \n''')

def prepdata_cmd(dm,n):
    global rootname
    ifok = os.path.join(dir_i,f'data-{dm}')
    cmd = f'prepdata ../{dir_s_type}/{rootname[5:]}_DM{dm}.dat -o {rootname}_DM{dm} '
    return cmd,ifok

if open5 == 1:
    dir_s = dir_data
    ifok = os.path.join(dir_a, 'ok-1data.txt')
    if not os.path.isfile(ifok):
        pool(num_processes,'prepdata',dms,prepdata_cmd)
        os.system('touch '+ifok)
        shutil.copy2(ifok, png_dir)

#单脉冲搜寻
def single_pulse_cmd(dm,n):
    global rootname
    cmd = f'single_pulse_search.py -t 7 -b -m 200 -p {rootname}_DM{dm}.dat'
    ifok = os.path.join(dir_i,f'sing-{dm}')
    return cmd,ifok

whitelist =[]
if open4 == 1:
    F_start_time = time.time()
    ifok_a = os.path.join(dir_a, 'ok-2single.txt')    
    print_log('''\n ==================== 单脉冲搜寻  ====================== \n''')
    if not os.path.isfile(ifok_a):
        pool(num_processes,'single',alldm_list,single_pulse_cmd)
        files = os.listdir(dir_s)
        filtered_names = [file for file in files if any(str(point) in file and (file.endswith('.inf') or file.endswith('.singlepulse')) for point in alldm_list)]
        handle_files(dir_s,filtered_names,whitelist,'copy',dir_single)
        start_time = time.time()
        cmd = 'single_pulse_search.py -b -m 200 {rootname}_DM*.singlepulse'.format(rootname=rootname)
        run_cmd(cmd,ifok=None,dir=dir_single,start_time=start_time)
        singlepulsar_ps =rootname+ "_singlepulse.ps"
        singlepulsar_png ='A_'+rootname+"_singlepulse.png"
        convert_ps_to_png(singlepulsar_ps, singlepulsar_png,rotated=False)
        handle_files(dir_single, singlepulsar_png, whitelist, 'copy', png_single) 
        os.chdir(dir_a)
        os.system('touch '+ifok_a)
        shutil.copy2(ifok, png_dir)
        all_run_time(F_start_time,'单脉冲搜寻')
    else:
        print_log('maybe已经生成单脉冲图')

# 2.3 傅里叶变换
def fft_cmd(dm,n):
    global rootname
    line = f"{rootname}_DM{dm}.dat"
    cmd = f'realfft '+line
    ifok = os.path.join(dir_i,f'fft-{dm}')
    return cmd,ifok

ifok = os.path.join(dir_a, 'ok-3fft.txt')
if not os.path.isfile(ifok):
    pool(num_processes,'fft',alldm_list,fft_cmd)
    os.system('touch '+ifok)
    shutil.copy2(ifok, png_dir)
else:
    print_log('ddfft 已经成功运行过，跳过,想重新运行请 rm okfft.txt')
os.chdir(dir_a)

#判断是否去除红噪声
def rednoise_cmd(dm,n):
    global rootname
    line = f"{rootname}_DM{dm}.fft"
    cmd = f'rednoise '+line
    ifok = os.path.join(dir_i,f'red-{dm}')
    return cmd,ifok

if open2 != 1:
    print_log('不去除红噪声')
else:
    # 2.4 去除红噪声
    ifok = os.path.join(dir_a, 'ok-4red.txt')
    if not os.path.isfile(ifok):
        F_start_time = time.time()
        pool(num_processes,'rednoise',alldm_list,rednoise_cmd)
        os.system('touch '+ifok)
        shutil.copy2(ifok, png_dir)
        all_run_time(F_start_time,'去除红噪声')
    else:
        print_log('去除红噪声已经成功运行过，跳过')  
    os.chdir(dir_a)

# 加速搜寻
def accelsearch_cmd(line,n):
    global rootname
    global Internal_memory
    cmd = f'accelsearch -ncpus 4 -numharm 4 '+zmax_str+' '+wmax_str+' '+Internal_memory+line
    ifok = os.path.join(dir_i,f'search{zmax}-{line}')
    return cmd,ifok

print_log ('''\n================ddsearch subbands==================\n''')
ifok = os.path.join(dir_a, 'ok-5search.txt')
if not os.path.isfile(ifok):
    F_start_time = time.time()
    if open2 == 1:
        search_names = sorted([file for file in os.listdir(dir_s) if file.endswith('red.fft')])
    else :
        search_names = sorted([file for file in os.listdir(dir_s) if file.endswith('.fft') and not file.endswith('red.fft') ])
    pool(num_processes,'search',search_names,accelsearch_cmd)
    all_run_time(F_start_time,'去除红噪声')
else:
    print_log('搜寻已经成功运行过，跳过') 
os.chdir(dir_a)

#筛选：生成oksifting后将不会再进行筛选，防止数据被覆盖
folder_name = os.path.basename(png_dir)
folder_name = folder_name[:-10]
siftingname = 'cand_sifting_'+folder_name +'.txt'

oksift = os.path.join(dir_a,'ok-6sifting')
print_log ('''\n================Setp5:ddsifting candidates==================\n''')
if not os.path.isfile(oksift):
    os.chdir(dir_s)
    cands = ACCEL_sift(zmax,open2,wmax)
    candnumber = len(cands)
    time_log('待折叠候选体个数为：',len(cands))
    copy_files_with_name(dir_a , png_dir,'candidate_list_from_script')
    copy_files_with_name(dir_a, dir_s,'candidate_list_from_script' )
    cand_only = mask_only +'.txt'
    cand_file_only =os.path.join(dir_a,cand_only)

    with open(dir_a+'/candidate_list_from_script', "r") as f:
        lines = f.readlines()
        sifting = []
        for line in lines:
            if line.startswith("#"):
                print_log(line)
                sifting.append(line)
            if line.startswith(rootname):
                print_log(line)
                sifting.append(line) 
    with open(dir_a+'/cand_sifting.txt', "w") as f:
        f.write('#待折叠候选体个数为：'+str(candnumber)+'\n')
        for line in sifting:
            f.write(line)
    with open(cand_file_only, "w") as f:
        f.write('#待折叠候选体个数为：'+str(candnumber)+'\n')
        for line in sifting:
            f.write(line)
    copy_files_with_name(dir_a, png_dir,cand_only )
    os.system('touch '+oksift) 
    shutil.copy2(oksift,png_dir)
os.chdir(dir_a)

#按信噪比进行排序
# 文件路径
input_file_path = os.path.join(dir_a,'cand_sifting.txt')  # 请替换为实际的输入文件路径
if open2 == 1:
    SNR_file=os.path.join(dir_a,'cand_sift_SNR_red.txt') 
else:
    SNR_file=os.path.join(dir_a,'cand_sift_SNR.txt')    # 请替换为实际的输出文件路径

# 读取数据并按SNR排序
with open(input_file_path, 'r') as infile:
    # 读取所有行
    lines = infile.readlines()
    cand_n = len(lines)
    print('#待折叠候选体个数为：'+str(cand_n)+'\n')
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
print("排序后的数据已保存到", SNR_file)

# 3 fold
ACCEL_file= 'ACCEL_'+str(zmax)+'.cand'
if wmax != 0:
    ACCEL_file = 'ACCEL_'+str(zmax)+'_JERK_'+str(wmax)+'.cand'
os.chdir(dir_a)
start_time = time.time()
whitelist=[]
all_png_file =[]
dir_p = dir_a +'/png'
n = 0
with open(SNR_file, "r") as f:
    lines = f.readlines()
    a1 = 0
    tall = 0
    aa = len(lines)
    for line in lines:
        if line.startswith(rootname):
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
            outname =pstart_str+str(n)+'_'+rootname
            foldinfo = '#折叠第'+str(n)+'张图'
            print_log(cand_file,candnum,dm ,p_ms)
            datfile = rootname+'_DM'+str(dm)+'.dat'
            t1 = time.time()
            a1+=1
            print_log ('''\n================Setp5:folding candidates==================\n''')
            file_list = os.listdir(dir_s)
            filtered_f = [file for file in file_list if dm in file]
            filter_inf_files = [file for file in filtered_f if file.endswith(".inf")]
            filter_cand_files = [file for file in filtered_f if file.endswith(ACCEL_file)]
            #19c67_DM32.80_ACCEL_2_JERK_20.cand
            #print(filter_cand_files)

            png =outname+'_DM'+str(dm)+candname+ str(candnum).lstrip("0")+'.pfd.png'
            ps =outname+'_DM'+str(dm)+candname+ str(candnum).lstrip("0")+'.pfd.ps' 
            png_file =  os.path.join(png_dir,png)
            ps_file = os.path.join(dir_s,ps)
            all_png_file.append(png_file)

            t1 = time.time()       
            if n > foldnum:
                break
            os.chdir(dir_s)
            if open1 == 1:
                foldcmd = f"prepfold{bar} {ignorechan} -dm {dm} -n {Nint} -accelcand {candnum} -accelfile {cand_file}.cand -mask ../{maskname} ../{filename} -o {outname}_DM{dm} -noxwin {foldway}"    
            else:
                dir_p = png_dir
                foldcmd = f"prepfold{bar} {ignorechan} -dm {dm} -n {Nint} -accelcand {candnum} -accelfile {cand_file}.cand -mask {maskname} {datfile} -o {outname}_DM{dm} -noxwin {foldway}"            
            handle_files(dir_s, filter_inf_files, whitelist, 'copy',dir_p )
            handle_files(dir_s, filter_cand_files, whitelist, 'copy',dir_p )

            with open(prep_cmd, 'a') as file:
                if not os.path.isfile(png_file):
                    time_log(foldinfo +'\n')
                    file.write(foldinfo +'\n')
                    file.write(foldcmd+'\n')   
                    run_cmd(foldcmd,start_time=t1)
                if not os.path.isfile(png_file):
                    #print('未能成功生成png,尝试自行转换')
                    convert_ps_to_png(ps_file, png_file)
                    #convert_ps_to_png(ps, png)
                t2 = time.time()
                dt = t2 -t1
                tall += dt
                display_progress_with_time(min(n,foldnum),foldnum,tall)
                os.chdir(dir_a)
                #copy_files_with_name(dir_s, png_dir,png)           
            os.chdir(dir_a)

os.chdir(dir_a)
# files = [f for f in os.listdir(dir_a) if os.path.isfile(os.path.join(dir_a, f))]
# ok_files = [f for f in files if f.startswith("ok-")]
# handle_files(dir_s,ok_files,whitelist,'copy',png_dir)
end_time = time.time()
execution_time = end_time - start_time
execution_time_str = format_execution_time(execution_time)
time_log( "全部折叠运行时间为： " + execution_time_str + "\n")
time.sleep(2)   


t_end = time.time()
execution_time = t_end- t_start
execution_time_str = format_execution_time(execution_time)
time_log( "程序完整运行运行时间为： " + execution_time_str + "\n")


print('尝试打包文件')
email_content = '该程序运行成功\n'
email_content += f'源名：{rootname}\n'
email_content += f'png文件路径{png_dir}\n'
py_file = os.path.join(dir_a,'all_pulsar_search.py')
file_paths = all_png_file[:30]
file_paths.append(SNR_file)
file_paths.append(py_file)
send_email(email_content, file_paths)

print_program_message('end')
