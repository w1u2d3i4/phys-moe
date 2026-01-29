import torch
from datetime import datetime
from loguru import logger
from config.Arguments import Arguments
import os
import shutil
from pathlib import Path
from sg_matrix import read_rule_similarity_matrix, set_sg_to_sys_map

## local paths
root_dir = './' 
label_dir = 'Datasets/ltvr/cifar100'
data_dir = 'Datasets/ltvr/cifar100' 
utils_dir = './'

## server paths
save_dir = './Log/'
log_dir = './Log/'

def config(task:str, model:str, dataset:str, save_log:bool, data_path:str, rule_matrix_path:str)->Arguments:
    """
    Configure the arguments for the training
    task: task name
    model: model name
    dataset: dataset name
    opt: additional arguments
    """
    args = Arguments()
    args.task = task

    args.model = Arguments()
    args.train = Arguments()
    args.dataset = Arguments()
    args.scheduler = Arguments()
    args.rt_scheduler = Arguments()
    args.data_path = data_path
    args.rule_matrix_path = rule_matrix_path
    
    # training parameters
    args.train.batch_size = 128
    args.train.epochs = 200
    args.train.lr = 1e-4
    args.train.lr_rt = 1e-4
    args.train.rt_start = 180 
    args.train.optimizer = 'SGD'
    args.train.rt_optimizer = 'SGD'
    args.train.weight_decay = 5e-4
    args.train.num_workers = 0
    args.train.save_log = save_log
    args.train.save_dir = os.path.join(save_dir, task)
    args.train.log_dir = os.path.join(log_dir, task)
    args.train.resume =  None # 
    args.train.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.train.use_wandb = 0
    args.train.clip_grad = 5
    args.train.log_interval = 100
    args.train.save_epoch = [119,179]
    args.train.early_stop = 200
    args.train.core_metric = ['acc1','maximize']

    # learning rate scheduler
    
    args.scheduler.name = 'cosineannel'
    args.scheduler.warmup_epoch = 5
    args.scheduler.total_epoch = args.train.rt_start
    args.scheduler.eta_min = 0.0
    args.scheduler.base_lr=args.train.lr
    args.scheduler.warmup_lr=0.15

    args.rt_scheduler.name = 'cosineannel'
    args.rt_scheduler.warmup_epoch = 5
    args.rt_scheduler.total_epoch = args.train.epochs - args.train.rt_start
    args.rt_scheduler.eta_min = 0.0
    args.rt_scheduler.base_lr=args.train.lr
    args.rt_scheduler.warmup_lr=0.1

    ## parameters for dataset
    
    args.dataset.name = dataset
    args.dataset.label_dir = label_dir
    args.dataset.data_dir = data_dir
    args.dataset.loader = None
    args.dataset.imgsz = 32
    args.dataset.imb_factor = 0.01

    args.train.save_name = f'{dataset}-{int(1/args.dataset.imb_factor)}_{model}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

    ## parameters for model
    args.model.name = model
    #args.model.num_experts = 16
    args.model.num_experts = 8
    #args.model.num_classes = 100 if 'cifar100' in dataset.lower() else 10
    args.model.num_classes = 230
    args.model.dropout = 0.1
    args.model.sg_to_sys_map = set_sg_to_sys_map()
    args.model.p_dropout = 0.1
    #尾部专家
    #args.model.tail_expert_ids = [8,9,10,11,12,13,14,15]
    args.model.tail_expert_ids = [4,5,6,7]
    args.model.rule_matrix = read_rule_similarity_matrix(args.rule_matrix_path)

    args.core_params = Arguments()
    args.core_params.ce_weight = 1
    args.core_params.distill_weight = 1
    args.core_params.parallel_weight = 1
    args.core_params.diversity_epoch = 30
    args.core_params.diversity_weight = 1
    args.core_params.hnm_weight = 1
    args.core_params.hnm_start_epoch = 120
    args.core_params.hnm_k = 7
    args.core_params.hnm_momentum = 0.9
    args.core_params.hnm_temperature = 0.07

    if args.train.save_log:
        logger.add(f'{args.train.log_dir}/{args.train.save_name}/process.log',level="INFO",filter=lambda record: not record["extra"].get("params", False))
        logger.add(f'{args.train.log_dir}/{args.train.save_name}/params.log', level="INFO", filter=lambda record: record["extra"].get("params", False))
        try:
            backup_code(args.train.log_dir, args.train.save_name)
            logger.info("Code backup successful")
        except Exception as e:
            logger.error(f"Code backup failed: {e}")

    return args
    

def backup_code(self_log_dir: str, save_name: str):
    log_task_dir = os.path.join(self_log_dir, save_name)

    code_backup_path = os.path.join(log_task_dir, 'code')

    Path(code_backup_path).mkdir(parents=True, exist_ok=True)

    current_dir = os.getcwd()

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.abspath(item_path) == os.path.abspath(log_dir):
            continue

        dest_path = os.path.join(code_backup_path, item)

        if os.path.isfile(item_path):
            shutil.copy2(item_path, dest_path)
        elif os.path.isdir(item_path):
            shutil.copytree(item_path, dest_path, dirs_exist_ok=True)