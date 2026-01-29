#!/bin/bash
# 运行coremof和inor数据集的自动调参（后台运行，输出到log文件）

# 设置GPU设备
DEVICE=0

# 数据集列表
DATASETS=("coremof19" "inor")

# 创建日志目录
LOG_DIR="tune_logs"
mkdir -p $LOG_DIR

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 为每个数据集运行调参（后台运行）
for DATASET in "${DATASETS[@]}"; do
    LOG_FILE="${LOG_DIR}/tune_${DATASET}_${TIMESTAMP}.log"
    
    echo "=========================================="
    echo "开始调参: $DATASET"
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
    echo "使用 'tail -f $LOG_FILE' 查看实时日志"
    echo "使用 'kill $PID' 停止训练"
    echo ""
done

echo "所有数据集的调参已在后台启动！"
echo "使用 'ps aux | grep tune_coremof_inor' 查看运行状态"
echo "使用 'tail -f ${LOG_DIR}/tune_*.log' 查看日志"
