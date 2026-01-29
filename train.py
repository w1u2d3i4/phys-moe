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
from datasets.OneD_Dataset import XrdData

#import models.Resnet as models_2d
#import datasets.CIFAR_LT as datasets_2d
import os
import torch

# =============================================================================

def get_model_and_dataset(args:Namespace)->tuple:
    """
    Get the model and dataset.
    This function need to be redefined for every new task.
    """
    # 根据数据源类型选择数据集
    use_lmdb = getattr(args, 'use_lmdb', False)
    use_processed_npy = getattr(args, 'use_processed_npy', False)
    
    if use_lmdb:
        # 使用LMDB数据源（需要实时处理）
        from datasets.LMDB_Dataset import LMDBXrdData
        lmdb_path = getattr(args, 'lmdb_path', None)
        if lmdb_path is None:
            raise ValueError("使用LMDB数据源时，必须提供--lmdb_path参数")
        
        # 确定sg_count.csv路径
        sg_count_path = getattr(args.model, 'sg_count_path', None)
        if sg_count_path is None and hasattr(args, 'rule_matrix_path'):
            # 假设sg_count.csv在rule_matrix_path的同一目录
            rule_dir = os.path.dirname(args.rule_matrix_path)
            sg_count_path = os.path.join(rule_dir, 'sg_count.csv')
            if not os.path.exists(sg_count_path):
                sg_count_path = None
        
        dataset = {
            'train': LMDBXrdData(lmdb_path, train=True, sg_count_path=sg_count_path),
            'val': LMDBXrdData(lmdb_path, train=False, sg_count_path=sg_count_path)
        }
    elif use_processed_npy:
        # 使用已处理好的NPY数据（与MP20格式一致，直接读取）
        processed_data_dir = getattr(args, 'processed_data_dir', None)
        if processed_data_dir is None:
            raise ValueError("使用已处理的NPY数据时，必须提供--processed_data_dir参数")
        
        # 获取数据集名称（用于构建目录和文件名）
        dataset_name = getattr(args, 'dataset_name', 'ccdc_sg')
        
        # 构建文件路径（支持不同的数据集名称）
        train_dir = os.path.join(processed_data_dir, f'{dataset_name}_train')
        test_dir = os.path.join(processed_data_dir, f'{dataset_name}_test')
        
        train_file = os.path.join(train_dir, f'train_{dataset_name}.npy')
        val_file = os.path.join(train_dir, f'test_val_{dataset_name}.npy')
        test_file = os.path.join(test_dir, f'test_{dataset_name}.npy')
        
        # 检查文件是否存在
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"训练集文件不存在: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"验证集文件不存在: {val_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试集文件不存在: {test_file}")
        
        # 确定sg_count.csv路径（用于空间群分类）
        # 优先使用args.model.sg_count_path，如果没有则尝试自动查找
        dataset_name = getattr(args, 'dataset_name', 'ccdc_sg')
        sg_count_path = getattr(args.model, 'sg_count_path', None)
        if sg_count_path is None:
            # 尝试多个可能的路径
            possible_paths = []
            # 1. processed_data_dir下的{dataset_name}_sg_count.csv
            possible_paths.append(os.path.join(processed_data_dir, f'{dataset_name}_sg_count.csv'))
            # 2. ICL目录下的{dataset_name}_sg_count.csv或ccdc_sg_count.csv
            icl_dir = os.path.dirname(__file__)
            possible_paths.append(os.path.join(icl_dir, f'{dataset_name}_sg_count.csv'))
            possible_paths.append(os.path.join(icl_dir, 'ccdc_sg_count.csv'))
            # 3. rule_matrix_path同一目录下的sg_count.csv或ccdc_sg_count.csv
            if hasattr(args, 'rule_matrix_path'):
                rule_dir = os.path.dirname(args.rule_matrix_path)
                possible_paths.append(os.path.join(rule_dir, f'{dataset_name}_sg_count.csv'))
                possible_paths.append(os.path.join(rule_dir, 'ccdc_sg_count.csv'))
                possible_paths.append(os.path.join(rule_dir, 'sg_count.csv'))
            # 4. 当前目录下的{dataset_name}_sg_count.csv或ccdc_sg_count.csv
            possible_paths.append(f'{dataset_name}_sg_count.csv')
            possible_paths.append('ccdc_sg_count.csv')
            possible_paths.append('sg_count.csv')
            
            # 查找第一个存在的文件
            for path in possible_paths:
                if os.path.exists(path):
                    sg_count_path = path
                    break
        
        # 设置sg_count_path到args.model，供模型使用
        if sg_count_path and os.path.exists(sg_count_path):
            args.model.sg_count_path = sg_count_path
            print(f"使用空间群分类文件: {sg_count_path}")
        else:
            print(f"Warning: 未找到sg_count.csv或ccdc_sg_count.csv，将使用默认分类")
        
        # 使用XrdData读取（与MP20相同的方式）
        dataset = {
            'train': XrdData(train_file),
            'val': XrdData(val_file)
        }
    else:
        # 使用原始文件数据源（默认，MP20格式）
        file_paths = os.listdir(args.data_path)
        train_files = [os.path.join(args.data_path,f) for f in file_paths if f.startswith('train')]
        test_files = [os.path.join(args.data_path,f) for f in file_paths if f.startswith('test')]
        dataset = {'train': XrdData(train_files[0]),
                   'val': XrdData(test_files[0])}

    args.label_dis = dataset['train'].get_cls_num_list()
    
    # 根据 use_nocontrast 参数选择模型模块
    if hasattr(args, 'use_nocontrast') and args.use_nocontrast:
        import models.Resnet1D_nocontrast as models_1d
        # 如果使用 nocontrast 版本，模型名称需要调整
        if args.model.name == 'ResNet1D_MoE':
            model_name = 'ResNet1D_MoE_NoContrast'
        else:
            model_name = args.model.name
    else:
        import models.Resnet1D as models_1d
        model_name = args.model.name
    
    #if 'ResNet1D' in args.model.name:
    md_module = models_1d
    #else:
    #    md_module = models_2d

    model = getattr(md_module, model_name)(args)
    return model, dataset

def main(args:Namespace, device_list:list)->None:
    """
    Main function
    args: arguments
    """
    model, dataset = get_model_and_dataset(args)
    if len(device_list) >  1 :
        model = torch.nn.DataParallel(model,device_list).to(args.train.device)
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
    opt.add_argument('--data_path', type=str, default='/opt/data/private/mp20/mp20_train', 
                     help='data path for file-based dataset (required if not using --use_lmdb or --use_processed_npy)')
    opt.add_argument('--lmdb_path', type=str, default=None,
                     help='LMDB database path (required if using --use_lmdb)')
    opt.add_argument('--use_lmdb', action='store_true',
                     help='use LMDB data source (requires real-time processing)')
    opt.add_argument('--use_processed_npy', action='store_true',
                     help='use pre-processed NPY data (same format as MP20, no processing needed)')
    opt.add_argument('--processed_data_dir', type=str, default='/opt/data/private/xrd2c_data',
                     help='directory containing processed NPY data (should contain {dataset_name}_train/ and {dataset_name}_test/ subdirectories)')
    opt.add_argument('--dataset_name', type=str, default='ccdc_sg',
                     help='dataset name for processed NPY data (default: ccdc_sg, options: ccdc_sg, inor, coremof19)')
    opt.add_argument('--rule_matrix_path', type=str, default='/opt/data/private/ICL/rule_matrix.csv', required=True)
    opt.add_argument('--seed', type=int, default=42, help='random seed')
    opt.add_argument('--save_log', type=bool, default=True, help='save log')
    opt.add_argument("--device",type=str,default="0")
    opt.add_argument('--use_nocontrast', action='store_true', help='use model version without contrastive learning')
    opt.add_argument('--loss_scheme', type=str, default='default', 
                     choices=['default', 'scheme2', 'scheme3'], 
                     help='loss scheme: default=full loss, scheme2=remove L_reg and L_col, scheme3=remove L_reg, L_col and L_sg')
    opt.add_argument('--use_contrast_scheme2', action='store_true', 
                     help='use contrastive loss in scheme2 (only works when --loss_scheme scheme2)')
    opt.add_argument('--epochs', type=int, default=None,
                     help='total number of training epochs (default: 200 from config)')
    opt.add_argument('--sg_count_path', type=str, default=None,
                     help='path to sg_count.csv file for space group classification (default: auto-detect)')
    opt.add_argument('--lr_restart_enabled', action='store_true', default=True,
                     help='enable learning rate restart when performance stagnates (default: True)')
    opt.add_argument('--lr_restart_threshold', type=int, default=20,
                     help='number of epochs without improvement before LR restart (default: 20)')
    opt.add_argument('--lr_restart_factor', type=float, default=0.5,
                     help='LR restart factor (restart_lr = initial_lr * factor, default: 0.5)')
    opt.add_argument('--use_adamw', action='store_true',
                     help='use AdamW optimizer instead of SGD')
    opt.add_argument('--adjust_lr_strategy', action='store_true',
                     help='adjust LR strategy: delay early decay and increase target LR')
    opt.add_argument('--early_decay_ratio', type=float, default=None,
                     help='early decay ratio (default: 0.1, or 0.2 if --adjust_lr_strategy)')
    opt.add_argument('--early_decay_target_lr', type=float, default=None,
                     help='early decay target LR (default: 5e-4, or 1e-3 if --adjust_lr_strategy)')
    opt.add_argument('--max_grad_norm', type=float, default=1.5,
                     help='maximum gradient norm for clipping (default: 1.0, increased from 0.5 to allow more gradient flow)')
    opt.add_argument('--grad_warn_threshold', type=float, default=50.0,
                     help='gradient norm warning threshold (default: 100.0, set higher to reduce warnings)')
    opt.add_argument('--disable_grad_warn', action='store_true',
                     help='disable gradient norm warnings completely')
    opt.add_argument('--lambda_dol', type=float, default=0.5,
                     help='weight for MoE independent optimization loss (default: 0.5, increased from 0.1 since l_dol is now averaged over experts)')
    opt.add_argument('--lambda_col', type=float, default=0.3,
                     help='weight for collaborative learning loss (default: 0.3, used in scheme2)')
    opt.add_argument('--disable_col', action='store_true',
                     help='disable collaborative learning loss (L_col) in scheme2')
    opt.add_argument('--lambda_hier', type=float, default=1.5,
                     help='weight for hierarchical loss (l_sys + l_sg) (default: 1.5)')
    opt.add_argument('--lambda_scl', type=float, default=0.2,
                     help='weight for contrastive learning loss (default: 0.2, only used when --use_contrast_scheme2)')
    opt.add_argument('--frontend_type', type=str, default='resnet',
                     choices=['resnet', 'vit', 'resnet_transformer'],
                     help='frontend: resnet (ResTcn only), vit, or resnet_transformer (ResNet+4层Transformer)')
    opt.add_argument('--vit_patch_size', type=int, default=50,
                     help='ViT patch size (default: 50, only for --frontend_type vit)')
    opt.add_argument('--vit_embed_dim', type=int, default=256,
                     help='ViT embedding dimension (default: 256, only for vit)')
    opt.add_argument('--vit_depth', type=int, default=6,
                     help='ViT transformer depth (default: 6, only for vit)')
    opt.add_argument('--vit_num_heads', type=int, default=8,
                     help='ViT number of attention heads (default: 8, only for vit)')
    opt.add_argument('--vit_mlp_ratio', type=float, default=4.0,
                     help='ViT MLP ratio (default: 4.0, only for vit)')
    opt.add_argument('--vit_use_cls_token', action='store_true', default=True,
                     help='ViT use CLS token (default: True, only for vit)')
    # ResNet+Transformer 前端参数（仅当 --frontend_type resnet_transformer 时生效）
    opt.add_argument('--rt_embed_dim', type=int, default=256,
                     help='ResNet+Transformer embed_dim (default: 256)')
    opt.add_argument('--rt_num_heads', type=int, default=8,
                     help='ResNet+Transformer attention heads (default: 8)')
    opt.add_argument('--rt_num_layers', type=int, default=4,
                     help='ResNet+Transformer transformer layers (default: 4)')
    opt.add_argument('--rt_mlp_ratio', type=float, default=4.0,
                     help='ResNet+Transformer MLP ratio (default: 4.0)')
    opt.add_argument('--rt_dropout', type=float, default=None,
                     help='ResNet+Transformer dropout (default: same as p_dropout)')
    opt.add_argument('--rt_resnet_out_len', type=int, default=132,
                     help='ResNet 输出序列长度，约 8500/2^6 (default: 132)')
    opt = opt.parse_args()
    
    # 验证参数
    if opt.use_lmdb:
        if opt.lmdb_path is None:
            raise ValueError("使用--use_lmdb时必须提供--lmdb_path参数")
        if opt.use_processed_npy:
            raise ValueError("不能同时使用--use_lmdb和--use_processed_npy")
    elif opt.use_processed_npy:
        if opt.processed_data_dir is None:
            raise ValueError("使用--use_processed_npy时必须提供--processed_data_dir参数")
    else:
        if opt.data_path is None:
            raise ValueError("必须提供--data_path参数（或使用--use_lmdb/--use_processed_npy）")
    
    set_seed(opt.seed)
    
    # 解析device列表
    device_list = [int(i) for i in opt.device.split(',')]
    
    # 对于LMDB或已处理的NPY数据源，data_path可以设为None或使用默认值
    if opt.use_lmdb:
        data_path_for_config = opt.data_path or '/opt/data/private/mp20/mp20_train'
    elif opt.use_processed_npy:
        data_path_for_config = opt.data_path or '/opt/data/private/mp20/mp20_train'
    else:
        data_path_for_config = opt.data_path
    
    args = config(opt.task, opt.model, opt.dataset, opt.save_log, data_path_for_config, opt.rule_matrix_path)
    args.base_args = base_config(opt.task, opt.model, opt.dataset, opt.save_log, data_path_for_config)
    
    # 设置device（覆盖config中的默认设置）
    # config中默认设置为'cuda'（会使用cuda:0），这里根据命令行参数更新
    if len(device_list) == 1:
        # 单个GPU: 'cuda:0', 'cuda:1', etc.
        args.train.device = f'cuda:{device_list[0]}' if torch.cuda.is_available() else 'cpu'
    else:
        # 多个GPU: 使用第一个作为主设备
        args.train.device = f'cuda:{device_list[0]}' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {args.train.device} (device_list={device_list})")
    
    # 如果指定了epochs参数，覆盖config中的设置
    if opt.epochs is not None:
        args.train.epochs = opt.epochs
        # 同时更新scheduler的total_epoch（如果需要）
        if hasattr(args.scheduler, 'total_epoch'):
            # rt_start通常是epochs的90%，但这里保持原有逻辑
            args.scheduler.total_epoch = args.train.rt_start
        if hasattr(args.rt_scheduler, 'total_epoch'):
            args.rt_scheduler.total_epoch = args.train.epochs - args.train.rt_start
    
    # 传递数据源相关参数
    args.use_lmdb = opt.use_lmdb
    args.use_processed_npy = opt.use_processed_npy
    if opt.use_lmdb:
        args.lmdb_path = opt.lmdb_path
    if opt.use_processed_npy:
        args.processed_data_dir = opt.processed_data_dir
        args.dataset_name = opt.dataset_name
    # 将 use_nocontrast 参数传递给 args
    args.use_nocontrast = opt.use_nocontrast
    # 将 loss_scheme 参数传递给 args.model
    if opt.loss_scheme != 'default':
        args.model.loss_scheme = opt.loss_scheme
    elif opt.use_nocontrast:
        # 如果使用nocontrast，默认loss_scheme为nocontrast
        args.model.loss_scheme = 'nocontrast'
    else:
        # 默认使用完整版本
        args.model.loss_scheme = 'full'
    # 将 use_contrast_scheme2 参数传递给 args.model
    if opt.use_contrast_scheme2:
        args.model.use_contrast_scheme2 = True
    # 将 sg_count_path 参数传递给 args.model
    if opt.sg_count_path:
        args.model.sg_count_path = opt.sg_count_path
    
    # 学习率重启策略参数
    args.train.lr_restart_enabled = opt.lr_restart_enabled
    args.train.lr_restart_threshold = opt.lr_restart_threshold
    args.train.lr_restart_factor = opt.lr_restart_factor
    
    # 切换到AdamW优化器
    if opt.use_adamw:
        args.train.optimizer = 'AdamW'
        args.train.rt_optimizer = 'AdamW'
        # 设置学习率为5e-4（根据用户要求）
        args.train.lr = 5e-4
        args.train.lr_rt = 5e-4
        # AdamW的weight_decay通常更小
        args.train.weight_decay = 1e-4
        print(f"切换到AdamW优化器: lr={args.train.lr}, weight_decay={args.train.weight_decay}")
    
    # 调整学习率策略
    if opt.adjust_lr_strategy:
        # 延迟早期衰减：从10%改为20%
        args.train.early_decay_ratio = opt.early_decay_ratio if opt.early_decay_ratio is not None else 0.2
        # 提高目标学习率：从5e-4改为1e-3
        args.train.early_decay_target_lr = opt.early_decay_target_lr if opt.early_decay_target_lr is not None else 1e-3
        print(f"调整学习率策略: early_decay_ratio={args.train.early_decay_ratio}, "
              f"early_decay_target_lr={args.train.early_decay_target_lr}")
    elif opt.early_decay_ratio is not None or opt.early_decay_target_lr is not None:
        # 如果单独指定了参数，也应用
        if opt.early_decay_ratio is not None:
            args.train.early_decay_ratio = opt.early_decay_ratio
        if opt.early_decay_target_lr is not None:
            args.train.early_decay_target_lr = opt.early_decay_target_lr
        print(f"自定义学习率策略: early_decay_ratio={args.train.early_decay_ratio}, "
              f"early_decay_target_lr={args.train.early_decay_target_lr}")
    
    # 设置梯度裁剪阈值
    args.train.max_grad_norm = opt.max_grad_norm
    print(f"梯度裁剪阈值: max_grad_norm={args.train.max_grad_norm}")
    
    # 设置梯度警告相关参数
    args.train.grad_warn_threshold = opt.grad_warn_threshold
    args.train.grad_warn_enabled = not opt.disable_grad_warn
    if args.train.grad_warn_enabled:
        print(f"梯度警告阈值: grad_warn_threshold={args.train.grad_warn_threshold} (梯度范数超过此值才会输出警告)")
    else:
        print("梯度警告已禁用")
    
    # 设置损失权重（仅对scheme2有效）
    if opt.loss_scheme == 'scheme2':
        args.model.lambda_dol = opt.lambda_dol
        args.model.lambda_col = opt.lambda_col
        args.model.lambda_hier = opt.lambda_hier
        args.model.lambda_scl = opt.lambda_scl
        args.model.use_col = not opt.disable_col  # 如果指定--disable_col，则use_col=False
        col_status = "启用" if args.model.use_col else "禁用"
        print(f"损失权重配置 (Scheme2): lambda_dol={opt.lambda_dol}, lambda_col={opt.lambda_col} ({col_status}), lambda_hier={opt.lambda_hier}, lambda_scl={opt.lambda_scl}")
    
    # 设置前端网络类型及对应参数
    args.model.frontend_type = opt.frontend_type
    if opt.frontend_type == 'vit':
        args.model.vit_patch_size = opt.vit_patch_size
        args.model.vit_embed_dim = opt.vit_embed_dim
        args.model.vit_depth = opt.vit_depth
        args.model.vit_num_heads = opt.vit_num_heads
        args.model.vit_mlp_ratio = opt.vit_mlp_ratio
        args.model.vit_use_cls_token = opt.vit_use_cls_token
        print(f"前端网络: ViT")
        print(f"  patch_size={opt.vit_patch_size}, embed_dim={opt.vit_embed_dim}")
        print(f"  depth={opt.vit_depth}, num_heads={opt.vit_num_heads}, mlp_ratio={opt.vit_mlp_ratio}")
        print(f"  use_cls_token={opt.vit_use_cls_token}")
    elif opt.frontend_type == 'resnet_transformer':
        args.model.rt_embed_dim = opt.rt_embed_dim
        args.model.rt_num_heads = opt.rt_num_heads
        args.model.rt_num_layers = opt.rt_num_layers
        args.model.rt_mlp_ratio = opt.rt_mlp_ratio
        args.model.rt_dropout = opt.rt_dropout if opt.rt_dropout is not None else getattr(args.model, 'p_dropout', 0.1)
        args.model.rt_resnet_out_len = opt.rt_resnet_out_len
        print(f"前端网络: ResNet + 4 层 Transformer")
        print(f"  resnet_out_len={opt.rt_resnet_out_len}, embed_dim={opt.rt_embed_dim}")
        print(f"  num_layers={opt.rt_num_layers}, num_heads={opt.rt_num_heads}, mlp_ratio={opt.rt_mlp_ratio}")
    else:
        print(f"前端网络: ResTcn (仅 ResNet1D)")
    
    main(args, device_list)
