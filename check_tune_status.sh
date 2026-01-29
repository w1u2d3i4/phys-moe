#!/bin/bash
# 检查调参进程状态和日志

LOG_DIR="tune_logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "日志目录不存在: $LOG_DIR"
    exit 1
fi

echo "=========================================="
echo "调参进程状态"
echo "=========================================="

# 查找所有调参进程
PIDS=$(ps aux | grep "tune_coremof_inor.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "没有运行中的调参进程"
else
    echo "运行中的进程:"
    ps aux | grep "tune_coremof_inor.py" | grep -v grep
    echo ""
fi

echo "=========================================="
echo "日志文件列表"
echo "=========================================="

# 列出所有日志文件
if ls ${LOG_DIR}/tune_*.log 1> /dev/null 2>&1; then
    for LOG_FILE in ${LOG_DIR}/tune_*.log; do
        echo ""
        echo "文件: $LOG_FILE"
        echo "大小: $(du -h $LOG_FILE | cut -f1)"
        echo "最后更新: $(stat -c %y $LOG_FILE | cut -d. -f1)"
        
        # 查找对应的PID文件
        PID_FILE="${LOG_FILE%.log}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat $PID_FILE)
            if ps -p $PID > /dev/null 2>&1; then
                echo "状态: 运行中 (PID: $PID)"
            else
                echo "状态: 已结束 (PID: $PID)"
            fi
        fi
        
        # 显示最后几行
        echo "最后10行:"
        tail -n 10 $LOG_FILE | sed 's/^/  /'
    done
else
    echo "没有找到日志文件"
fi

echo ""
echo "=========================================="
echo "快速命令"
echo "=========================================="
echo "查看所有日志: tail -f ${LOG_DIR}/tune_*.log"
echo "查看最新日志: tail -f \$(ls -t ${LOG_DIR}/tune_*.log | head -1)"
