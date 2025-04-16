#!/bin/bash

# 変換対象のディレクトリ（デフォルトはカレントディレクトリ）
TARGET_DIR=${1:-.}

# カレントディレクトリ内のファイルのみ処理（サブディレクトリは無視）
for file in "$TARGET_DIR"/*; do
    if [ -f "$file" ]; then
        dos2unix "$file"
    fi
done
