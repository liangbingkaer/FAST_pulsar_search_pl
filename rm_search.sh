#!/bin/bash

# 删除文件夹函数
delete_folder() {
    if [ -d "$1" ]; then
        rm -rf "$1"
        echo "Folder '$1' deleted."
    else
        echo "Folder '$1' does not exist."
    fi
}

echo "WARNING!!! Will delete folders: 03_subbands, 03_barydata"
echo "Are you sure? (Y/N)"
read -r confirm1

echo "Please wait for 3 seconds to reconsider..."
sleep 3

echo "Are you still sure to delete folders: 03_subbands, 03_barydata? (Y/N)"
read -r confirm2

if [[ "$confirm1" = "Y" && "$confirm2" = "Y" ]]; then
    delete_folder "03_subbands"
    delete_folder "03_barydata"
else
    echo "Operation aborted. Exiting..."
    exit 1
fi

