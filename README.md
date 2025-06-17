# FAST_pulsar_search_pl

改编来自：https://github.com/alex88ridolfi/PULSAR_MINER

## 更新功能
* 更好的打印和 log,更多色彩
* 支持可选择全部 fits 的折叠
* 多线程并行
* 支持FAST数据质心修正
* 折叠不生成 png 或 png 空白时的额外处理
* 自定义邮箱发送折叠图
* 默认折叠dat文件，添加snr-dm辅助判断图,人为选择待折叠序列
* 命名逻辑优化


## 待补充内容
* 支持 add-search 合并 fits 
* 分段搜寻
* jinglepulsar
* 添加 pysolator 可选

## 使用
### 添加到环境变量
```python
export PATH=/home/peng/work/python-workspace/FAST_pulsar_search_pl:${PATH}
```

### step1
```python
search_prep.py
```
修改对应的.cfg配置文件参数

### step2
```python
FAST_pulsar_search_pl.py
```

### step3
```python
ts2raw.py
```
创建id_list.txt，添加序列后重新运行ts2raw.py

### step4
```python
pool_run_cmd.py -cmdfile /home/.../fold.sh -ncpus 4
```
