#!/bin/bash
# 快速测试调参脚本（使用100个epochs）

# 测试单个数据集的步骤1（col loss测试）
echo "快速测试：步骤1 - Col Loss测试（100 epochs）"
python tune_coremof_inor.py \
    --dataset coremof19 \
    --device 0 \
    --epochs_step1 100 \
    --skip_step2 \
    --skip_step3

echo ""
echo "测试完成！"
