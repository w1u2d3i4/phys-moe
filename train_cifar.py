"""
This is the main file for training CIFAR dataset.
This file is used to test this framework.
"""
from argparse import Namespace, ArgumentParser
from utils.utils import set_seed
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# need to be redefined for every new task
from config.config_cifar_base import config as base_config
from config.config_cifar_moe import config as config
import models.Resnet1D as models_1d
from datasets.OneD_Dataset import XrdData

#import models.Resnet as models_2d
#import datasets.CIFAR_LT as datasets_2d
import os

# =============================================================================

def get_model_and_dataset(args:Namespace)->tuple:
    """
    Get the model and dataset.
    This function need to be redefined for every new task.
    """
    """
    if 'OneD' in args.dataset.name:
        ds_module = datasets_1d
    else:
        ds_module = datasets_2d
    """
    #只需要传file_path，args.dataset应为XrdData
    #dataset = {'train': getattr(ds_module, args.dataset.name)(args, train=True),
    #           'val':getattr(ds_module, args.dataset.name.replace("IMBALANCE",""))(args, train=False)}
    file_paths = os.listdir(args.data_path)
    train_files = [os.path.join(args.data_path,f) for f in file_paths if f.startswith('train')]
    test_files = [os.path.join(args.data_path,f) for f in file_paths if f.startswith('test')]
    dataset = {'train': XrdData(train_files[0]),
               'val': XrdData(test_files[0])}

    args.label_dis = dataset['train'].get_cls_num_list()
    
    #if 'ResNet1D' in args.model.name:
    md_module = models_1d
    #else:
    #    md_module = models_2d

    model = getattr(md_module, args.model.name)(args)
    return model, dataset

def main(args:Namespace)->None:
    """
    Main function
    args: arguments
    """
    model, dataset = get_model_and_dataset(args)
    
    from Trainer.moe_trainer import MoE_Trainer
    trainer = MoE_Trainer(args, model, dataset, args.train.device)
    trainer.train()
    

# =============================================================================
# Normal Mode
# =============================================================================

if __name__ == '__main__':
    opt = ArgumentParser()
    opt.add_argument('--task', type=str, default='ICL', help='task name')
    opt.add_argument('--model', type=str, default='ResNet1D_MoE', help='model name')
    opt.add_argument('--dataset', type=str, default='mp20', help='dataset name')
    opt.add_argument('--data_path', type=str, default='/opt/data/private/mp20/mp20_train', required=True)
    opt.add_argument('--seed', type=int, default=42, help='random seed')
    opt.add_argument('--save_log', type=bool, default=True, help='save log')
    opt = opt.parse_args()
    set_seed(opt.seed)
    args = config(opt.task, opt.model, opt.dataset, opt.save_log, opt.data_path)
    args.base_args = base_config(opt.task, opt.model, opt.dataset, opt.save_log, opt.data_path)
    main(args)
