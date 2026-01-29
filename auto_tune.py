#!/usr/bin/env python3
"""
基于终端输出监控的自动调参系统

该脚本可以：
1. 监控训练过程的终端输出
2. 解析关键指标（loss, accuracy等）
3. 根据指标变化自动调整超参数
4. 重新启动训练或动态调整参数
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
        # 匹配格式: Epoch 10[123.45s]: {'epoch': 10, 'ce': 0.1234, 'acc1': 0.5678, ...}
        pattern = r'Epoch\s+(\d+)\[([\d.]+)s\]:\s*({.*})'
        match = re.search(pattern, line)
        if not match:
            return None
        
        epoch = int(match.group(1))
        time_cost = float(match.group(2))
        metrics_str = match.group(3)
        
        try:
            # 解析字典字符串（使用ast.literal_eval更安全）
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
            logger.warning(f"解析指标失败: {e}, 行: {line}")
            return None


@dataclass
class HyperparameterConfig:
    """超参数配置"""
    lambda_dol: float = 0.5
    lambda_col: float = 0.3
    lambda_hier: float = 1.5
    lambda_scl: float = 0.2
    lr: float = 5e-4
    max_grad_norm: float = 1.0
    use_adamw: bool = True
    adjust_lr_strategy: bool = True
    
    def to_args(self) -> List[str]:
        """转换为命令行参数"""
        args = []
        args.extend(['--lambda_dol', str(self.lambda_dol)])
        args.extend(['--lambda_col', str(self.lambda_col)])
        args.extend(['--lambda_hier', str(self.lambda_hier)])
        args.extend(['--lambda_scl', str(self.lambda_scl)])
        args.extend(['--max_grad_norm', str(self.max_grad_norm)])
        if self.use_adamw:
            args.append('--use_adamw')
        if self.adjust_lr_strategy:
            args.append('--adjust_lr_strategy')
        return args


class OutputMonitor:
    """终端输出监控器"""
    
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.metrics_history: List[TrainingMetrics] = []
        self.latest_metrics: Optional[TrainingMetrics] = None
        
    def monitor(self, callback=None) -> List[TrainingMetrics]:
        """监控进程输出并解析指标"""
        metrics_list = []
        
        # 实时读取输出
        for line in iter(self.process.stdout.readline, b''):
            if not line:
                break
            
            line_str = line.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue
            
            # 打印原始输出
            print(line_str, flush=True)
            
            # 解析指标
            metrics = TrainingMetrics.from_log_line(line_str)
            if metrics:
                metrics_list.append(metrics)
                self.metrics_history.append(metrics)
                self.latest_metrics = metrics
                
                # 调用回调函数
                if callback:
                    callback(metrics)
            
            # 检查进程是否结束
            if self.process.poll() is not None:
                break
        
        return metrics_list
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """获取最新的指标"""
        return self.latest_metrics
    
    def get_metrics_history(self) -> List[TrainingMetrics]:
        """获取历史指标"""
        return self.metrics_history


class AutoTuner:
    """自动调参器"""
    
    def __init__(self, 
                 base_config: HyperparameterConfig,
                 search_space: Optional[Dict] = None,
                 min_epochs: int = 10,
                 patience: int = 5):
        """
        Args:
            base_config: 基础超参数配置
            search_space: 参数搜索空间
            min_epochs: 最小训练轮数（在此之后才开始调参）
            patience: 耐心值（多少轮没有改进后调整参数）
        """
        self.base_config = base_config
        self.current_config = base_config
        self.search_space = search_space or self._default_search_space()
        self.min_epochs = min_epochs
        self.patience = patience
        
        self.best_metric_value = -float('inf')
        self.stagnation_count = 0
        self.tune_history: List[Tuple[HyperparameterConfig, float]] = []
        
    def _default_search_space(self) -> Dict:
        """默认搜索空间"""
        return {
            'lambda_dol': [0.3, 0.4, 0.5, 0.6, 0.7],
            'lambda_col': [0.1, 0.2, 0.3, 0.4, 0.5],
            'lambda_hier': [1.0, 1.2, 1.5, 1.8, 2.0],
            'lambda_scl': [0.1, 0.15, 0.2, 0.25, 0.3],
            'max_grad_norm': [0.5, 1.0, 1.5, 2.0]
        }
    
    def should_tune(self, metrics: TrainingMetrics) -> bool:
        """判断是否需要调参"""
        if metrics.epoch < self.min_epochs:
            return False
        
        # 检查是否有改进
        current_metric = metrics.best_metric
        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            self.stagnation_count = 0
            return False
        else:
            self.stagnation_count += 1
            return self.stagnation_count >= self.patience
    
    def tune(self, metrics: TrainingMetrics, 
             problem_type: str = 'stagnation') -> HyperparameterConfig:
        """
        根据当前指标调整超参数
        
        Args:
            metrics: 当前训练指标
            problem_type: 问题类型 ('stagnation', 'gradient_explosion', 'poor_tail', 'poor_head')
        """
        new_config = HyperparameterConfig(**asdict(self.current_config))
        
        if problem_type == 'stagnation':
            # 性能停滞：尝试调整损失权重
            logger.info("检测到性能停滞，调整损失权重...")
            # 随机选择新的权重组合
            import random
            new_config.lambda_dol = random.choice(self.search_space['lambda_dol'])
            new_config.lambda_col = random.choice(self.search_space['lambda_col'])
            new_config.lambda_hier = random.choice(self.search_space['lambda_hier'])
            new_config.lambda_scl = random.choice(self.search_space['lambda_scl'])
            
        elif problem_type == 'poor_tail':
            # 尾部类性能差：增加协作学习和对比学习权重
            logger.info("检测到尾部类性能差，增加协作学习和对比学习权重...")
            new_config.lambda_col = min(new_config.lambda_col * 1.5, 0.5)
            new_config.lambda_scl = min(new_config.lambda_scl * 1.5, 0.4)
            
        elif problem_type == 'poor_head':
            # 头部类性能差：增加独立优化权重
            logger.info("检测到头部类性能差，增加独立优化权重...")
            new_config.lambda_dol = min(new_config.lambda_dol * 1.3, 0.8)
            new_config.lambda_col = max(new_config.lambda_col * 0.7, 0.1)
            
        elif problem_type == 'gradient_explosion':
            # 梯度爆炸：增加梯度裁剪阈值
            logger.info("检测到梯度爆炸，增加梯度裁剪阈值...")
            new_config.max_grad_norm = min(new_config.max_grad_norm * 1.5, 2.0)
        
        self.current_config = new_config
        self.stagnation_count = 0  # 重置停滞计数
        
        logger.info(f"新配置: lambda_dol={new_config.lambda_dol}, "
                   f"lambda_col={new_config.lambda_col}, "
                   f"lambda_hier={new_config.lambda_hier}, "
                   f"lambda_scl={new_config.lambda_scl}, "
                   f"max_grad_norm={new_config.max_grad_norm}")
        
        return new_config
    
    def analyze_problem(self, metrics: TrainingMetrics) -> str:
        """分析当前训练问题"""
        if metrics.tail_acc < 0.3 and metrics.head_acc > 0.7:
            return 'poor_tail'
        elif metrics.head_acc < 0.5 and metrics.tail_acc > 0.4:
            return 'poor_head'
        else:
            return 'stagnation'


def run_training_with_tuning(base_cmd: List[str],
                            base_config: HyperparameterConfig,
                            max_trials: int = 5,
                            min_epochs: int = 10,
                            patience: int = 5,
                            search_space: Optional[Dict] = None):
    """
    运行训练并进行自动调参
    
    Args:
        base_cmd: 基础训练命令（不包含超参数）
        base_config: 初始超参数配置
        max_trials: 最大调参次数
        min_epochs: 最小训练轮数
        patience: 耐心值
        search_space: 参数搜索空间
    """
    tuner = AutoTuner(base_config, search_space, min_epochs, patience)
    current_config = base_config
    
    for trial in range(max_trials):
        logger.info(f"\n{'='*60}")
        logger.info(f"开始第 {trial + 1}/{max_trials} 次训练")
        logger.info(f"{'='*60}")
        
        # 构建完整命令
        cmd = base_cmd + current_config.to_args()
        logger.info(f"训练命令: {' '.join(cmd)}")
        
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False
        )
        
        # 监控输出
        monitor = OutputMonitor(process)
        metrics_callback = None
        
        def on_metrics(metrics: TrainingMetrics):
            """指标更新回调"""
            if tuner.should_tune(metrics):
                problem_type = tuner.analyze_problem(metrics)
                new_config = tuner.tune(metrics, problem_type)
                logger.warning(f"Epoch {metrics.epoch}: 需要调整参数，但当前训练将继续...")
                # 注意：这里不能直接停止训练，需要等待当前训练完成
        
        monitor.monitor(callback=on_metrics)
        
        # 等待进程结束
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"训练失败，返回码: {return_code}")
            break
        
        # 获取最终指标
        final_metrics = monitor.get_latest_metrics()
        if final_metrics:
            final_score = final_metrics.best_metric
            logger.info(f"训练完成，最终best_metric: {final_score:.4f}")
            tuner.tune_history.append((current_config, final_score))
            
            # 如果性能很好，可以提前结束
            if final_score > 0.9:  # 根据实际情况调整阈值
                logger.info("性能已达到目标，提前结束调参")
                break
            
            # 准备下一次调参
            if trial < max_trials - 1:
                problem_type = tuner.analyze_problem(final_metrics)
                current_config = tuner.tune(final_metrics, problem_type)
        else:
            logger.warning("未能获取最终指标，使用默认配置继续")
    
    # 输出最佳配置
    if tuner.tune_history:
        best_config, best_score = max(tuner.tune_history, key=lambda x: x[1])
        logger.info(f"\n{'='*60}")
        logger.info(f"调参完成！最佳配置:")
        logger.info(f"  最佳得分: {best_score:.4f}")
        logger.info(f"  lambda_dol: {best_config.lambda_dol}")
        logger.info(f"  lambda_col: {best_config.lambda_col}")
        logger.info(f"  lambda_hier: {best_config.lambda_hier}")
        logger.info(f"  lambda_scl: {best_config.lambda_scl}")
        logger.info(f"  max_grad_norm: {best_config.max_grad_norm}")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='基于终端输出监控的自动调参系统')
    parser.add_argument('--base_cmd', type=str, required=True,
                       help='基础训练命令（Python脚本路径）')
    parser.add_argument('--base_args', type=str, nargs='*', default=[],
                       help='基础训练参数（不包含超参数）')
    parser.add_argument('--lambda_dol', type=float, default=0.5,
                       help='初始lambda_dol')
    parser.add_argument('--lambda_col', type=float, default=0.3,
                       help='初始lambda_col')
    parser.add_argument('--lambda_hier', type=float, default=1.5,
                       help='初始lambda_hier')
    parser.add_argument('--lambda_scl', type=float, default=0.2,
                       help='初始lambda_scl')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='初始max_grad_norm')
    parser.add_argument('--max_trials', type=int, default=5,
                       help='最大调参次数')
    parser.add_argument('--min_epochs', type=int, default=10,
                       help='最小训练轮数（之后才开始调参）')
    parser.add_argument('--patience', type=int, default=5,
                       help='耐心值（多少轮没有改进后调整参数）')
    parser.add_argument('--use_adamw', action='store_true', default=True,
                       help='使用AdamW优化器')
    parser.add_argument('--adjust_lr_strategy', action='store_true', default=True,
                       help='调整学习率策略')
    
    args = parser.parse_args()
    
    # 构建基础命令
    base_cmd = ['python', args.base_cmd] + args.base_args
    
    # 创建初始配置
    base_config = HyperparameterConfig(
        lambda_dol=args.lambda_dol,
        lambda_col=args.lambda_col,
        lambda_hier=args.lambda_hier,
        lambda_scl=args.lambda_scl,
        max_grad_norm=args.max_grad_norm,
        use_adamw=args.use_adamw,
        adjust_lr_strategy=args.adjust_lr_strategy
    )
    
    # 运行自动调参
    run_training_with_tuning(
        base_cmd=base_cmd,
        base_config=base_config,
        max_trials=args.max_trials,
        min_epochs=args.min_epochs,
        patience=args.patience
    )


if __name__ == '__main__':
    main()
