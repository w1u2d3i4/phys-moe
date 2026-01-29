#!/bin/bash
# 停止调参进程

LOG_DIR="tune_logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "日志目录不存在: $LOG_DIR"
    exit 1
fi

# 如果提供了PID文件，停止指定进程
if [ $# -ge 1 ]; then
    PID_FILE=$1
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        echo "停止进程 PID: $PID"
        kill $PID 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "进程已停止"
            rm -f $PID_FILE
        else
            echo "无法停止进程（可能已经结束）"
        fi
        exit 0
    else
        echo "PID文件不存在: $PID_FILE"
        exit 1
    fi
fi

# 查找所有调参进程
PIDS=$(ps aux | grep "tune_coremof_inor.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "没有运行中的调参进程"
    exit 0
fi

echo "找到以下调参进程:"
ps aux | grep "tune_coremof_inor.py" | grep -v grep

echo ""
read -p "是否要停止所有调参进程? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for PID in $PIDS; do
        echo "停止进程 PID: $PID"
        kill $PID 2>/dev/null
    done
    echo "所有进程已停止"
    
    # 清理PID文件
    if ls ${LOG_DIR}/*.pid 1> /dev/null 2>&1; then
        for PID_FILE in ${LOG_DIR}/*.pid; do
            PID=$(cat $PID_FILE 2>/dev/null)
            if [ ! -z "$PID" ] && ! ps -p $PID > /dev/null 2>&1; then
                rm -f $PID_FILE
            fi
        done
    fi
else
    echo "取消操作"
fi
