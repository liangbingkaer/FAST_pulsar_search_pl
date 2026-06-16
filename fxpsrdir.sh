#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <x>"
  exit 1
fi

# 获取参数 'x'
x="$1"

# 创建一个计数器变量，用于追踪.fits文件数量
counter=0
xcounter=0

# 创建第一个文件夹
folder_name="f_$((counter + 1))"

# 遍历当前目录中所有的 .fits 文件
for file in $(ls -1 *.fits | sort); do
  if [[ -f "$file" ]]; then
    # 将 .fits 文件移动到相应的文件夹
    mkdir -p "$folder_name"
    mv "$file" "$folder_name/"
    
    # 增加.fits文件计数器
    xcounter=$((xcounter + 1))

    # 如果.fits文件计数器达到 'x'，则创建一个新的文件夹
    if [ "$xcounter" -eq "$x" ]; then
      xcounter=0
      counter=$((counter + 1))
      folder_name="f_$((counter + 1))"
    fi
  fi
done

# 在所有文件夹中复制 .py 文件和 .sh 文件
echo "将py和sh复制到f开头的文件夹"
for folder in */; do
    if [[ -d "$folder" ]]; then
        cp *.py "$folder/"
        cp *.sh "$folder/"
    fi
done

