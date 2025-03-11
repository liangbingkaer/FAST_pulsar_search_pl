# Written by penglong in 2023.11.11
# 添加内容：
# 增加ps文件转png   
# 可以自定义添加源名，使输出文件命名更加合理，使程序逻辑更加合理化   
# 将筛选结果保存   
# 可以查看程序各个模块运行时间，剩余运行时间，百分比提示   
# 保留所需文件，多余删除
# 更好的查看运行日志

import os,sys,glob,re
import time,datetime
import json
import shutil
import subprocess
from PIL import Image
from builtins import map
from datetime import timedelta
import presto.sifting as sifting
from operator import itemgetter, attrgetter

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import base64

cwd = os.getcwd()

#step2:sifting参数设置
def ACCEL_sift(zmax,open2,wmax):
    global cwd
    globaccel = "*_ACCEL_*%d" % zmax
    if wmax != 0:
        globaccel = "*_ACCEL_*%d_JERK_%d" % (zmax, wmax)
    if open2 == 1:
        globinf = "*DM*_red.inf"
    else:
        globinf = "*DM*.inf"
    min_num_DMs = 2
    # Lowest DM to consider as a "real" pulsar
    low_DM_cutoff = 1.0
    # Ignore candidates with a sigma (from incoherent power summation) less than this
    sifting.sigma_threshold = 4.0
    # Ignore candidates with a coherent power less than this
    sifting.c_pow_threshold = 50.0

    # If the birds file works well, the following shouldn't
    # be needed at all...  If they are, add tuples with the bad
    # values and their errors.
    #                (ms, err)
    sifting.known_birds_p = []
    #                (Hz, err)
    sifting.known_birds_f = []

    # The following are all defined in the sifting module.
    # But if we want to override them, uncomment and do it here.
    # You shouldn't need to adjust them for most searches, though.

    # How close a candidate has to be to another candidate to                
    # consider it the same candidate (in Fourier bins)
    sifting.r_err = 1.1
    # Shortest period candidates to consider (s)
    sifting.short_period = 0.0014
    # Longest period candidates to consider (s)
    sifting.long_period = 0.040
    # Ignore any candidates where at least one harmonic does exceed this power
    sifting.harm_pow_cutoff = 8.0
    print_log('min_num_DMs:  '+str(min_num_DMs) )
    print_log('low_DM_cutoff:  '+str(low_DM_cutoff) )
    print_log('sifting.sigma_threshold:  '+str(sifting.sigma_threshold))
    print_log('sifting.c_pow_threshold:  '+str(sifting.c_pow_threshold))
    print_log('sifting.r_err:  '+str(sifting.r_err))
    print_log('筛选的最短周期为： '+str(sifting.short_period ) )
    print_log('筛选的最长周期为： '+str(sifting.long_period))
    print_log('########记录完毕########')

    #--------------------------------------------------------------

    # Try to read the .inf files first, as _if_ they are present, all of
    # them should be there.  (if no candidates are found by accelsearch
    # we get no ACCEL files...
    inffiles = glob.glob(globinf)
    candfiles = glob.glob(globaccel)

    if open2 != 1:
        inffiles = sorted([file for file in inffiles if 'inf' in file and not 'red' in file])
        candfiles = sorted([file for file in candfiles if 'ACCEL' in file and not 'red' in file])
    # Check to see if this is from a short search
    if open2 ==1:
        if len(re.findall("_[0-9][0-9][0-9]M_" , inffiles[0])):
            dmstrs = [x.split("DM")[-1].split("_red_")[0] for x in candfiles]
        else:
            dmstrs = [x.split("DM")[-1].split("_red.inf")[0] for x in inffiles]
    if open2 !=1:
        if len(re.findall("_[0-9][0-9][0-9]M_" , inffiles[0])):
            dmstrs = [x.split("DM")[-1].split("_")[0] for x in candfiles]
        else:
            dmstrs = [x.split("DM")[-1].split(".inf")[0] for x in inffiles]

    dmstrs = [s[:-2] if s.endswith("_p") or s.endswith("_q") else s for s in dmstrs]
    dms = list(map(float, dmstrs))
    dms.sort()
    dmstrs = ["%.2f"%x for x in dms]

    # Read in all the candidates
    cands = sifting.read_candidates(candfiles)

    # Remove candidates that are duplicated in other ACCEL files
    if len(cands):
        cands = sifting.remove_duplicate_candidates(cands)

    # Remove candidates with DM problems
    if len(cands):
        cands = sifting.remove_DM_problems(cands, min_num_DMs, dmstrs, low_DM_cutoff)

    # Remove candidates that are harmonically related to each other
    # Note:  this includes only a small set of harmonics
    if len(cands):
        cands = sifting.remove_harmonics(cands)

    # Write candidates to STDOUT
    if len(cands):
        cands.sort(key=attrgetter('sigma'), reverse=True)
        sifting.write_candlist(cands,cwd+"/candidate_list_from_script")
    return cands

#目前时间
def get_current_time_to_minute():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    return current_time

#更好的打印
def print_log(*args, sep=' ', end='\n', file=None, flush=False, log_files=None,masks =None):
    global cwd
    if log_files is None:
        log_files = ['logall.txt']

    # 构建消息字符串
    message = sep.join(str(arg) for arg in args) + end
    # 写入日志文件
    log_contents = [message]
    for log_filename in log_files:
        log_path = os.path.join(cwd, log_filename)
        with open(log_path, 'a') as f:
            f.writelines(log_contents)

    # 打印到标准输出
    if masks is None:
        masks = []
        print(message, end='', file=sys.stdout, flush=flush)
    else:
        masks = [str(mask) for mask in masks]  # 将所有 mask 转换为字符串
        colored_output = ""
        in_mask = False
        for char in message:
            if any(char in mask for mask in masks):
                if not in_mask:
                    colored_output += "\033[31m"  # 31m 是红色
                    in_mask = True
            else:
                if in_mask:
                    colored_output += "\033[0m"  # 0m 是恢复默认颜色
                    in_mask = False
            colored_output += char
        if in_mask:
            colored_output += "\033[0m"  # 确保最后一组 mask 之后恢复默认颜色
        print(colored_output)

# 具有替代prin_log的效果，并且额外记录在储存时间的logruntime.txt
def time_log(info, log_files=None,masks =None):
    if log_files is None:
        if masks ==  None:
            print_log(info, log_files=['logruntime.txt', 'logall.txt'])
        else:
            print_log(info, log_files=['logruntime.txt', 'logall.txt'],masks = masks)

# 记录程序开始和结束
def print_program_message(phase):
    current_time = get_current_time_to_minute()
    if phase == 'start':
        message = '\n\n\n**************************************程序开始***********************************\n'
        message += '本次程序运行开始时间为：' + current_time
    elif phase == 'end':
        message = '本次程序运行结束时间为：' + current_time
        message += '\n**************************************退出程序***********************************\n\n\n'
        sys.exit(0)
    else:
        message = '未知的程序阶段'

    time_log(message)

def mkdir(dir,ifok):
    if ifok == 1:
        os.makedirs(dir, exist_ok=True)


#判断用户输入
def rm_file(file_names):
    for file_name in file_names:
        try:
            os.remove(file_name)
            print_log(f"文件 {file_name} 删除成功")
        except FileNotFoundError:
            print_log(f"文件 {file_name} 不存在")
        except PermissionError:
            print_log(f"没有权限删除文件 {file_name}")
        except Exception as e:
            print_log(f"删除文件 {file_name} 时出现错误: {str(e)}")

def process_user_input(user_input,rootname):
    user_input = user_input.lower()
    if user_input in ["yes", "y"]:
        print_log('程序执行')
    elif user_input == rootname:
        print_log(f'{rootname}正确，程序执行')
    else:
        print_log(f'请输入正确的源名{rootname}')
        print_program_message('end')

def all_run_time(start_time , info):
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = format_execution_time(execution_time)
    log_message = f"\n运行 {info} 的总时间为： {execution_time_str}\n"
    print(log_message) 
    time.sleep(2)

#计算运行时间
def format_execution_time(execution_time):
    if execution_time < 60:
        return f"{execution_time:.1f}秒"
    elif execution_time < 3600:
        minutes = execution_time // 60
        seconds = execution_time % 60
        return f"{minutes}分钟{seconds:.1f}秒"
    elif execution_time < 86400:
        hours = execution_time // 3600
        minutes = (execution_time % 3600) // 60
        return f"{hours}小时{minutes}分钟"
    else:
        days = execution_time // 86400
        hours = (execution_time % 86400) // 3600
        minutes = (execution_time % 3600) // 60
        return f"{days}天{hours}小时{minutes}分钟"

def time_consum(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_str = format_execution_time(execution_time)
    time_log("\n" + "运行时间为： " + execution_time_str + "\n")
    time_log('')
    time.sleep(2)

#更好的运行
def run_cmd(cmd, ifok=None, dir=None, start_time=None):
    """
    Execute a command and log it to a file.

    Args:
        cmd (str): The command to execute.
        ifok (str, optional): Path to a file. If it exists, the command is skipped. Defaults to None.
        dir (str, optional): dir where the command should be executed. Defaults to None.
        start_time (float, optional): Start time for measuring time consumption. Defaults to None.
    """
    global cwd
    
    if ifok is None or not os.path.isfile(ifok):
        if dir:
            os.chdir(dir)
        path =  os.getcwd()
        print(f'程序运行路径为:{path}')
        print_log(cmd, masks=cmd)  # Assuming print_log is defined elsewhere
        time.sleep(1)  # Assuming this sleep is intentional
        try:
            output = subprocess.getoutput(cmd)
            print(output)
            with open(os.path.join(cwd, 'cmd.sh'), 'a') as file:
                file.write(cmd + '\n')
        except Exception as e:
            print(f"Error executing command: {e}")
        finally:
            if ifok:
                if not ifok.endswith('png'):
                    os.system('touch ' + ifok)
            if start_time:
                time_log(cmd)
                time_consum(start_time)  # Assuming time_consum is defined elsewhere

    else:
        print(f'File {ifok} exists.\n Skipping command 跳过\n: {cmd}\n')


def rfi(rficmd,mask_file,rfi_json_path):
    start_time = time.time()
    time_log(f'执行命令： {rficmd}\n')
    if not os.path.isfile(mask_file):
        output = subprocess.getoutput(rficmd)
        print(output)
        time_consum(start_time)
        rfiheader = {}
        for line in output.split('\n'):
            items = line.split("=")
            if len(items) > 1:
                rfiheader[items[0].strip()] = items[1].strip()
        with open(rfi_json_path, 'w') as file:
            json.dump(rfiheader, file)
        total_time = rfiheader.get('Total time (s)')
        print_log('\n'+'Total time(s): ' + str(total_time)+'\n')
    else:
        print_log(f'存在{mask_file}  ，已经正式rfi过，跳过')

def convert_ps_to_png(input_file, output_file, rotated=True):
    if not os.path.isfile(input_file):
        print(f'error:{input_file} 不存在')
    if not os.path.isfile(output_file):
        try:
            # 使用Ghostscript命令转换.ps文件为.png
            subprocess.run(["gs", "-dQUIET", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png256", "-sOutputFile=" + output_file, "-r300", input_file], check=True)
            print(f"{input_file} 转换为 {output_file} 成功")

            if rotated:
                # 打开转换后的.png文件
                image = Image.open(output_file)
                # 旋转图像
                rotated_image = image.rotate(-90, expand=True)
                # 保存旋转后的图像
                rotated_image.save(output_file)
                print(f"{output_file} 旋转成功")

        except FileNotFoundError:
            print(f"文件 {input_file} 不存在")
        except subprocess.CalledProcessError as e:
            print(f"处理文件 {input_file} 时出现错误: {e}")
        except Exception as e:
            print(f"处理文件 {input_file} 时出现错误: {str(e)}")

def copy_files_with_name(source_folder, destination_folder, target_filename):
    try:
        # 遍历源文件夹中的文件
        for filename in os.listdir(source_folder):
            # 检查文件名是否匹配目标文件名
            if filename == target_filename:
                # 构建源文件路径和目标文件路径
                source_file = os.path.join(source_folder, filename)
                destination_file = os.path.join(destination_folder, filename)
                # 复制文件
                shutil.copy(source_file, destination_file)
                print_log(f"已复制文件：{filename}")
    except FileNotFoundError:
        print_log(f"文件夹不存在：{source_folder} 或 {destination_folder}")
    except Exception as e:
        print_log(f"处理文件时出现错误: {str(e)}")

def display_progress_with_time(current, total, tall):
    progress = current / total  # 计算当前进度百分比
    percentage = int(progress * 100)  # 转换为整数百分比
    # 估算剩余时间
    if current > 0:
        estimated_remaining_time = (total - current) * (tall / current)
    else:
        estimated_remaining_time = 0
    # 格式化时间
    remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))
    #remaining_time_str = str(timedelta(seconds=estimated_remaining_time))
    # 显示百分比进度、进度条、已经过的时间和估算的剩余时间
    print('总次数为：'+str(total)+'    正在进行第'+str(current)+'次循环'+"    Progress: "+str(percentage)+'%')
    print('预估剩余完成时间为：'+remaining_time_str)

def handle_files(directory, file_list, whitelist, action, to_dir=None):
    """
    Handle files in a directory based on the specified action.
    eg: handle_files("/path/to/source_directory", file_list_to_copy, ["do_not_copy file"], "copy", to_dir="/path/to/destination_directory_copy")

    Parameters:
    - directory: The directory path.
    - file_list: List of files to process.
    - whitelist: List of files to exclude from deletion, moving, or copying.
    - action: "delete" to delete files, "move" to move files, "copy" to copy files.
    - move_to_directory: The destination directory when action is "move". None by default.
    - copy_to_directory: The destination directory when action is "copy". None by default.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is in the specified file_list and is not in the whitelist
            if file in file_list and file not in whitelist:
                file_path = os.path.join(root, file)
                if action == "delete":
                    os.remove(file_path)
                    #print(f"Deleted file: {file_path}", end='\r')  # Use '\r' to overwrite the same line
                elif action == "move":
                    destination_path = os.path.join(to_dir, file)
                    shutil.move(file_path, destination_path)
                    #print(f"Moved file from {file_path} to {destination_path}", end='\r')
                elif action == "copy":
                    destination_path = os.path.join(to_dir, file)
                    if not os.path.exists(destination_path):  # Check if the file already exists
                        shutil.copy(file_path, destination_path)
                        #print(f"Copied file from {file_path} to {destination_path}", end='\r')
                    else:
                        pass
                        #print(f"File already exists at {destination_path}. Skipped.", end='\r')
    print(f'文件{action}操作成功')

def fix_inf(dir_s):
    os.chdir(dir_s)
    inf_names = sorted([file for file in os.listdir(dir_s) if file.endswith('.inf')])
    for inf in inf_names:
        with open(inf, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            if ' Dispersion measure (cm-3 pc) ' in lines[i]:
                match = re.search(r"DM(\d+\.\d{2})", inf)
                if match:
                    dm = match.group(1)
                    print(dm)
                else:
                    print("未找到匹配的模式")

                lines[i] = ' Dispersion measure (cm-3 pc)     =   '+str(dm)+'\n'

        with open(inf, 'w') as file:
            file.writelines(lines)
    os.chdir(cwd)

def send_email(content, file_paths=None):
    # Email configuration
    mail_host = "smtp.qq.com"  
    mail_pass = 'niwqoduzipjnchgb'  
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
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {file_path}')
            message.attach(part)

    try:
        # Connect to SMTP server
        smtpObj = smtplib.SMTP_SSL(mail_host, 465) 
        smtpObj.login(sender, mail_pass)  
        smtpObj.sendmail(sender, receivers, message.as_string())  
        smtpObj.quit()
        print('邮件发送成功！！')
    except smtplib.SMTPException as e:
        print(e.__traceback__.tb_lineno, e)
        print('邮件发送失败！！')
