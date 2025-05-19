# FAST_pulsar_search_pl

改编来自：https://github.com/alex88ridolfi/PULSAR_MINER

## 更新功能
* 更好的打印和 log,更多色彩
* 支持可选择全部 fits 的折叠
* 多线程并行
* 支持FAST数据.质心修正
* 折叠不生成 png 或 png 空白时的额外处理
* 自定义邮箱发送折叠图

## 待补充内容
* 支持 add-search 合并 fits 后分段搜寻
* 添加 pysolator 可选

## 使用
### 添加到环境变量
```python
export PATH=/home/peng/work/python-workspace:${PATH}
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

### step4
```python
pm_run_multithread -cmdfile /home/.../fold.sh -ncpus 4
```
