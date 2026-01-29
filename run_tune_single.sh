#!/bin/bash
# 运行单个数据集的自动调参（后台运行，输出到log文件）

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <dataset> [device]"
    echo "示例: $0 coremof19 0"
    echo "      $0 inor 1"
    exit 1
fi

DATASET=$1
DEVICE=${2:-0}

# 创建日志目录
LOG_DIR="tune_logs"
mkdir -p $LOG_DIR

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/tune_${DATASET}_${TIMESTAMP}.log"

echo "=========================================="
echo "开始调参: $DATASET"
echo "GPU设备: $DEVICE"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 使用nohup在后台运行，输出到log文件
nohup python tune_coremof_inor.py \
    --dataset $DATASET \
    --device $DEVICE \
    --epochs_step1 100 \
    --epochs_step2 100 \
    --epochs_step3 100 \
    > $LOG_FILE 2>&1 &

# 保存进程ID
PID=$!
echo "进程ID: $PID"
echo "PID已保存到: ${LOG_DIR}/tune_${DATASET}_${TIMESTAMP}.pid"
echo $PID > "${LOG_DIR}/tune_${DATASET}_${TIMESTAMP}.pid"

echo ""
echo "调参已在后台启动！"
echo "使用以下命令查看日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "使用以下命令停止训练:"
echo "  kill $PID"
echo "  或"
echo "  kill \$(cat ${LOG_DIR}/tune_${DATASET}_${TIMESTAMP}.pid)"
echo ""
echo "使用以下命令查看运行状态:"
echo "  ps aux | grep tune_coremof_inor"
echo "  ps -p $PID"
