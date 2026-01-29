# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /opt/data/private/condaenv/CrystalFusionNet

nvidia-smi > gpu.log

python train.py \
       --task ICL \
       --model ResNet1D_MoE \
       --data_path /opt/data/private/mp20/mp20_train \
       --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
       --device 0,1 \
       --seed 42 \
       --save_log True