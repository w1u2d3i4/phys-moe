#!/usr/bin/env python3
"""
针对coremof和inor数据集的自动调参脚本

调参步骤：
1. 先测试是否启用col loss（测试启用和禁用两种情况）
2. 选择更好的一个
3. 然后调整loss权重
4. 调整lr下降速度（由快到慢）
"""

import re
import subprocess
import sys
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
from loguru import logger


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: int
    loss: float
    acc1: float
    acc2: float
    acc5: float
    sys_acc: float
    recall: float
    f1: float
    head_acc: float
    medium_acc: float
    tail_acc: float
    extreme_tail_acc: float
    best_metric: float
    best_sys_acc: float
    time_cost: float = 0.0
    
    @classmethod
    def from_log_line(cls, line: str) -> Optional['TrainingMetrics']:
        """从日志行解析指标"""
        pattern = r'Epoch\s+(\d+)\[([\d.]+)s\]:\s*({.*})'
        match = re.search(pattern, line)
        if not match:
            return None
        
        epoch = int(match.group(1))
        time_cost = float(match.group(2))
        metrics_str = match.group(3)
        
        try:
            import ast
            metrics_dict = ast.literal_eval(metrics_str)
            metrics = cls(
                epoch=epoch,
                time_cost=time_cost,
                loss=float(metrics_dict.get('ce', 0.0)),
                acc1=float(metrics_dict.get('acc1', 0.0)),
                acc2=float(metrics_dict.get('acc2', 0.0)),
                acc5=float(metrics_dict.get('acc5', 0.0)),
                sys_acc=float(metrics_dict.get('sys_acc', 0.0)),
                recall=float(metrics_dict.get('recall', 0.0)),
                f1=float(metrics_dict.get('f1', 0.0)),
                head_acc=float(metrics_dict.get('head_acc', 0.0)),
                medium_acc=float(metrics_dict.get('medium_acc', 0.0)),
                tail_acc=float(metrics_dict.get('tail_acc', 0.0)),
                extreme_tail_acc=float(metrics_dict.get('extreme_tail_acc', 0.0)),
                best_metric=float(metrics_dict.get('best metric', 0.0)),
                best_sys_acc=float(metrics_dict.get('best sys_acc', 0.0))
            )
            return metrics
        except Exception as e:
            return None


@dataclass
class ExperimentResult:
    """实验结果"""
    config_name: str
    use_col: bool
    lambda_dol: float
    lambda_col: float
    lambda_hier: float
    lambda_scl: float
    early_decay_ratio: float
    early_decay_target_lr: float
    final_best_metric: float
    final_tail_acc: float
    final_head_acc: float
    final_sys_acc: float
    
    def to_dict(self):
        return asdict(self)


class OutputMonitor:
    """终端输出监控器"""
    
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.metrics_history: List[TrainingMetrics] = []
        self.latest_metrics: Optional[TrainingMetrics] = None
        
    def monitor(self) -> List[TrainingMetrics]:
        """监控进程输出并解析指标"""
        metrics_list = []
        
        for line in iter(self.process.stdout.readline, b''):
            if not line:
                break
            
            line_str = line.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue
            
            print(line_str, flush=True)
            
            metrics = TrainingMetrics.from_log_line(line_str)
            if metrics:
                metrics_list.append(metrics)
                self.metrics_history.append(metrics)
                self.latest_metrics = metrics
            
            if self.process.poll() is not None:
                break
        
        return metrics_list
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        return self.latest_metrics


def run_training(base_cmd: List[str], config_name: str, epochs: int = 50) -> Optional[TrainingMetrics]:
    """运行训练并返回最终指标"""
    logger.info(f"\n{'='*80}")
    logger.info(f"开始训练: {config_name}")
    logger.info(f"{'='*80}")
    logger.info(f"命令: {' '.join(base_cmd)}")
    
    process = subprocess.Popen(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=False
    )
    
    monitor = OutputMonitor(process)
    monitor.monitor()
    
    return_code = process.wait()
    if return_code != 0:
        logger.error(f"训练失败，返回码: {return_code}")
        return None
    
    final_metrics = monitor.get_latest_metrics()
    if final_metrics:
        logger.info(f"\n训练完成: {config_name}")
        logger.info(f"  最终best_metric: {final_metrics.best_metric:.4f}")
        logger.info(f"  最终tail_acc: {final_metrics.tail_acc:.4f}")
        logger.info(f"  最终head_acc: {final_metrics.head_acc:.4f}")
        logger.info(f"  最终sys_acc: {final_metrics.sys_acc:.4f}")
    
    return final_metrics


def step1_test_col_loss(dataset_name: str, device: str, epochs: int = 100) -> bool:
    """
    步骤1: 测试是否启用col loss
    返回: True表示启用col loss更好，False表示禁用col loss更好
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"步骤1: 测试col loss（数据集: {dataset_name}）")
    logger.info(f"{'#'*80}")
    
    base_args = [
        'train.py',
        '--task', 'ICL',
        '--model', 'ResNet1D_MoE',
        '--dataset', 'mp20',
        '--use_processed_npy',
        '--processed_data_dir', '/opt/data/private/xrd2c_data',
        '--dataset_name', dataset_name,
        '--rule_matrix_path', '/opt/data/private/ICL/rule_matrix.csv',
        '--device', device,
        '--loss_scheme', 'scheme2',
        '--use_contrast_scheme2',
        '--use_adamw',
        '--lambda_dol', '0.5',
        '--lambda_col', '0.3',
        '--lambda_hier', '1.5',
        '--lambda_scl', '0.2',
        '--epochs', str(epochs),
        '--seed', '42',
        '--save_log', 'True'
    ]
    
    # 测试1: 启用col loss
    cmd_with_col = ['python'] + base_args
    metrics_with_col = run_training(cmd_with_col, f"{dataset_name}_with_col", epochs)
    
    # 测试2: 禁用col loss
    cmd_without_col = ['python'] + base_args + ['--disable_col']
    metrics_without_col = run_training(cmd_without_col, f"{dataset_name}_without_col", epochs)
    
    # 比较结果
    if metrics_with_col and metrics_without_col:
        score_with_col = metrics_with_col.best_metric
        score_without_col = metrics_without_col.best_metric
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Col Loss测试结果对比:")
        logger.info(f"  启用col loss: best_metric={score_with_col:.4f}, tail_acc={metrics_with_col.tail_acc:.4f}")
        logger.info(f"  禁用col loss: best_metric={score_without_col:.4f}, tail_acc={metrics_without_col.tail_acc:.4f}")
        
        # 优先考虑tail_acc，如果tail_acc相近则考虑best_metric
        if abs(metrics_with_col.tail_acc - metrics_without_col.tail_acc) > 0.05:
            # tail_acc差异明显，优先选择tail_acc高的
            use_col = metrics_with_col.tail_acc > metrics_without_col.tail_acc
            reason = f"tail_acc差异明显 ({metrics_with_col.tail_acc:.4f} vs {metrics_without_col.tail_acc:.4f})"
        else:
            # tail_acc相近，选择best_metric高的
            use_col = score_with_col > score_without_col
            reason = f"tail_acc相近，选择best_metric更高的 ({score_with_col:.4f} vs {score_without_col:.4f})"
        
        logger.info(f"  选择: {'启用' if use_col else '禁用'} col loss ({reason})")
        logger.info(f"{'='*80}\n")
        
        return use_col
    else:
        logger.warning("无法获取完整结果，默认启用col loss")
        return True


def step2_tune_loss_weights(dataset_name: str, device: str, use_col: bool, epochs: int = 100):
    """
    步骤2: 调整loss权重
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"步骤2: 调整loss权重（数据集: {dataset_name}, col_loss={'启用' if use_col else '禁用'}）")
    logger.info(f"{'#'*80}")
    
    # 定义搜索空间
    lambda_dol_space = [0.4, 0.5, 0.6]
    lambda_col_space = [0.2, 0.3, 0.4] if use_col else []
    lambda_hier_space = [1.2, 1.5, 1.8]
    lambda_scl_space = [0.15, 0.2, 0.25]
    
    best_result = None
    best_score = -float('inf')
    results = []
    
    # 网格搜索
    for lambda_dol in lambda_dol_space:
        for lambda_hier in lambda_hier_space:
            for lambda_scl in lambda_scl_space:
                if use_col:
                    for lambda_col in lambda_col_space:
                        config_name = f"{dataset_name}_dol{lambda_dol}_col{lambda_col}_hier{lambda_hier}_scl{lambda_scl}"
                        base_args = [
                            'train.py',
                            '--task', 'ICL',
                            '--model', 'ResNet1D_MoE',
                            '--dataset', 'mp20',
                            '--use_processed_npy',
                            '--processed_data_dir', '/opt/data/private/xrd2c_data',
                            '--dataset_name', dataset_name,
                            '--rule_matrix_path', '/opt/data/private/ICL/rule_matrix.csv',
                            '--device', device,
                            '--loss_scheme', 'scheme2',
                            '--use_contrast_scheme2',
                            '--use_adamw',
                            '--lambda_dol', str(lambda_dol),
                            '--lambda_col', str(lambda_col),
                            '--lambda_hier', str(lambda_hier),
                            '--lambda_scl', str(lambda_scl),
                            '--epochs', str(epochs),
                            '--seed', '42',
                            '--save_log', 'True'
                        ]
                        cmd = ['python'] + base_args
                        metrics = run_training(cmd, config_name, epochs)
                        
                        if metrics:
                            score = metrics.best_metric
                            result = ExperimentResult(
                                config_name=config_name,
                                use_col=True,
                                lambda_dol=lambda_dol,
                                lambda_col=lambda_col,
                                lambda_hier=lambda_hier,
                                lambda_scl=lambda_scl,
                                early_decay_ratio=0.1,  # 默认值
                                early_decay_target_lr=5e-4,  # 默认值
                                final_best_metric=score,
                                final_tail_acc=metrics.tail_acc,
                                final_head_acc=metrics.head_acc,
                                final_sys_acc=metrics.sys_acc
                            )
                            results.append(result)
                            if score > best_score:
                                best_score = score
                                best_result = result
                else:
                    config_name = f"{dataset_name}_dol{lambda_dol}_hier{lambda_hier}_scl{lambda_scl}_nocol"
                    base_args = [
                        'train.py',
                        '--task', 'ICL',
                        '--model', 'ResNet1D_MoE',
                        '--dataset', 'mp20',
                        '--use_processed_npy',
                        '--processed_data_dir', '/opt/data/private/xrd2c_data',
                        '--dataset_name', dataset_name,
                        '--rule_matrix_path', '/opt/data/private/ICL/rule_matrix.csv',
                        '--device', device,
                        '--loss_scheme', 'scheme2',
                        '--use_contrast_scheme2',
                        '--use_adamw',
                        '--disable_col',
                        '--lambda_dol', str(lambda_dol),
                        '--lambda_hier', str(lambda_hier),
                        '--lambda_scl', str(lambda_scl),
                        '--epochs', str(epochs),
                        '--seed', '42',
                        '--save_log', 'True'
                    ]
                    cmd = ['python'] + base_args
                    metrics = run_training(cmd, config_name, epochs)
                    
                    if metrics:
                        score = metrics.best_metric
                        result = ExperimentResult(
                            config_name=config_name,
                            use_col=False,
                            lambda_dol=lambda_dol,
                            lambda_col=0.0,
                            lambda_hier=lambda_hier,
                            lambda_scl=lambda_scl,
                            early_decay_ratio=0.1,
                            early_decay_target_lr=5e-4,
                            final_best_metric=score,
                            final_tail_acc=metrics.tail_acc,
                            final_head_acc=metrics.head_acc,
                            final_sys_acc=metrics.sys_acc
                        )
                        results.append(result)
                        if score > best_score:
                            best_score = score
                            best_result = result
    
    # 输出最佳结果
    if best_result:
        logger.info(f"\n{'='*80}")
        logger.info(f"步骤2最佳配置:")
        logger.info(f"  best_metric: {best_result.final_best_metric:.4f}")
        logger.info(f"  tail_acc: {best_result.final_tail_acc:.4f}")
        logger.info(f"  head_acc: {best_result.final_head_acc:.4f}")
        logger.info(f"  lambda_dol: {best_result.lambda_dol}")
        logger.info(f"  lambda_col: {best_result.lambda_col}")
        logger.info(f"  lambda_hier: {best_result.lambda_hier}")
        logger.info(f"  lambda_scl: {best_result.lambda_scl}")
        logger.info(f"{'='*80}\n")
    
    return best_result, results


def step3_tune_lr_decay(dataset_name: str, device: str, best_loss_config: ExperimentResult, epochs: int = 100):
    """
    步骤3: 调整lr下降速度（由快到慢）
    通过调整early_decay_ratio和early_decay_target_lr来实现
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"步骤3: 调整lr下降速度（数据集: {dataset_name}）")
    logger.info(f"{'#'*80}")
    
    # 定义lr策略搜索空间（由快到慢）
    # early_decay_ratio越小，衰减开始越早（下降越快）
    # early_decay_target_lr越小，目标lr越小（下降越快）
    lr_strategies = [
        # 快速下降
        {'early_decay_ratio': 0.05, 'early_decay_target_lr': 1e-4, 'name': 'fast'},
        # 中等速度下降
        {'early_decay_ratio': 0.1, 'early_decay_target_lr': 5e-4, 'name': 'medium'},
        # 慢速下降
        {'early_decay_ratio': 0.15, 'early_decay_target_lr': 8e-4, 'name': 'slow'},
        # 很慢下降
        {'early_decay_ratio': 0.2, 'early_decay_target_lr': 1e-3, 'name': 'very_slow'},
    ]
    
    best_result = None
    best_score = -float('inf')
    results = []
    
    for lr_strategy in lr_strategies:
        config_name = f"{dataset_name}_lr_{lr_strategy['name']}"
        
        base_args = [
            'train.py',
            '--task', 'ICL',
            '--model', 'ResNet1D_MoE',
            '--dataset', 'mp20',
            '--use_processed_npy',
            '--processed_data_dir', '/opt/data/private/xrd2c_data',
            '--dataset_name', dataset_name,
            '--rule_matrix_path', '/opt/data/private/ICL/rule_matrix.csv',
            '--device', device,
            '--loss_scheme', 'scheme2',
            '--use_contrast_scheme2',
            '--use_adamw',
            '--lambda_dol', str(best_loss_config.lambda_dol),
            '--lambda_hier', str(best_loss_config.lambda_hier),
            '--lambda_scl', str(best_loss_config.lambda_scl),
            '--epochs', str(epochs),
            '--seed', '42',
            '--save_log', 'True'
        ]
        
        if best_loss_config.use_col:
            base_args.extend(['--lambda_col', str(best_loss_config.lambda_col)])
        else:
            base_args.append('--disable_col')
        
        # 添加自定义lr策略参数
        # 使用adjust_lr_strategy标志，并覆盖默认值
        base_args.extend([
            '--adjust_lr_strategy',
            '--early_decay_ratio', str(lr_strategy['early_decay_ratio']),
            '--early_decay_target_lr', str(lr_strategy['early_decay_target_lr'])
        ])
        
        cmd = ['python'] + base_args
        metrics = run_training(cmd, config_name, epochs)
        
        if metrics:
            result = ExperimentResult(
                config_name=config_name,
                use_col=best_loss_config.use_col,
                lambda_dol=best_loss_config.lambda_dol,
                lambda_col=best_loss_config.lambda_col,
                lambda_hier=best_loss_config.lambda_hier,
                lambda_scl=best_loss_config.lambda_scl,
                early_decay_ratio=lr_strategy['early_decay_ratio'],
                early_decay_target_lr=lr_strategy['early_decay_target_lr'],
                final_best_metric=metrics.best_metric,
                final_tail_acc=metrics.tail_acc,
                final_head_acc=metrics.head_acc,
                final_sys_acc=metrics.sys_acc
            )
            results.append(result)
            if metrics.best_metric > best_score:
                best_score = metrics.best_metric
                best_result = result
    
    # 输出最佳结果
    if best_result:
        logger.info(f"\n{'='*80}")
        logger.info(f"步骤3最佳配置:")
        logger.info(f"  best_metric: {best_result.final_best_metric:.4f}")
        logger.info(f"  tail_acc: {best_result.final_tail_acc:.4f}")
        logger.info(f"  head_acc: {best_result.final_head_acc:.4f}")
        logger.info(f"  early_decay_ratio: {best_result.early_decay_ratio}")
        logger.info(f"  early_decay_target_lr: {best_result.early_decay_target_lr}")
        logger.info(f"{'='*80}\n")
    
    return best_result, results


def main():
    parser = argparse.ArgumentParser(description='针对coremof和inor数据集的自动调参')
    parser.add_argument('--dataset', type=str, choices=['coremof19', 'inor'], required=True,
                       help='数据集名称')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU设备')
    parser.add_argument('--epochs_step1', type=int, default=100,
                       help='步骤1的训练轮数（默认100）')
    parser.add_argument('--epochs_step2', type=int, default=100,
                       help='步骤2的训练轮数（默认100）')
    parser.add_argument('--epochs_step3', type=int, default=100,
                       help='步骤3的训练轮数（默认100）')
    parser.add_argument('--skip_step1', action='store_true',
                       help='跳过步骤1（直接使用指定配置）')
    parser.add_argument('--use_col', type=bool, default=None,
                       help='是否使用col loss（如果跳过步骤1则必须指定）')
    parser.add_argument('--skip_step2', action='store_true',
                       help='跳过步骤2')
    parser.add_argument('--skip_step3', action='store_true',
                       help='跳过步骤3')
    
    args = parser.parse_args()
    
    # 步骤1: 测试col loss
    if not args.skip_step1:
        use_col = step1_test_col_loss(args.dataset, args.device, args.epochs_step1)
    else:
        if args.use_col is None:
            logger.error("跳过步骤1时必须指定--use_col")
            return
        use_col = args.use_col
        logger.info(f"跳过步骤1，使用指定配置: use_col={use_col}")
    
    # 步骤2: 调整loss权重
    best_loss_config = None
    if not args.skip_step2:
        best_loss_config, loss_results = step2_tune_loss_weights(
            args.dataset, args.device, use_col, args.epochs_step2
        )
        if best_loss_config is None:
            logger.error("步骤2失败，无法继续")
            return
    else:
        logger.info("跳过步骤2，使用默认loss权重")
        best_loss_config = ExperimentResult(
            config_name=f"{args.dataset}_default",
            use_col=use_col,
            lambda_dol=0.5,
            lambda_col=0.3 if use_col else 0.0,
            lambda_hier=1.5,
            lambda_scl=0.2,
            early_decay_ratio=0.1,
            early_decay_target_lr=5e-4,
            final_best_metric=0.0,
            final_tail_acc=0.0,
            final_head_acc=0.0,
            final_sys_acc=0.0
        )
    
    # 步骤3: 调整lr下降速度
    if not args.skip_step3:
        best_lr_config, lr_results = step3_tune_lr_decay(
            args.dataset, args.device, best_loss_config, args.epochs_step3
        )
        
        # 输出最终最佳配置
        if best_lr_config:
            logger.info(f"\n{'#'*80}")
            logger.info(f"最终最佳配置（数据集: {args.dataset}）")
            logger.info(f"{'#'*80}")
            logger.info(f"  use_col: {best_lr_config.use_col}")
            logger.info(f"  lambda_dol: {best_lr_config.lambda_dol}")
            logger.info(f"  lambda_col: {best_lr_config.lambda_col}")
            logger.info(f"  lambda_hier: {best_lr_config.lambda_hier}")
            logger.info(f"  lambda_scl: {best_lr_config.lambda_scl}")
            logger.info(f"  early_decay_ratio: {best_lr_config.early_decay_ratio}")
            logger.info(f"  early_decay_target_lr: {best_lr_config.early_decay_target_lr}")
            logger.info(f"  最终best_metric: {best_lr_config.final_best_metric:.4f}")
            logger.info(f"  最终tail_acc: {best_lr_config.final_tail_acc:.4f}")
            logger.info(f"  最终head_acc: {best_lr_config.final_head_acc:.4f}")
            logger.info(f"  最终sys_acc: {best_lr_config.final_sys_acc:.4f}")
            logger.info(f"{'#'*80}\n")
            
            # 保存结果到JSON文件
            result_file = f"{args.dataset}_best_config.json"
            with open(result_file, 'w') as f:
                json.dump(best_lr_config.to_dict(), f, indent=2)
            logger.info(f"最佳配置已保存到: {result_file}")
    else:
        logger.info("跳过步骤3")


if __name__ == '__main__':
    main()
