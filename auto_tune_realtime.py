#!/usr/bin/env python3
"""
实时监控终端输出并动态调整参数的自动调参系统

该脚本可以：
1. 实时监控训练过程的终端输出
2. 解析关键指标（loss, accuracy等）
3. 根据指标变化动态调整超参数（通过信号或文件）
4. 支持训练过程中的参数热更新
"""

import re
import subprocess
import sys
import os
import json
import time
import signal
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
from loguru import logger
import queue


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
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def save_to_file(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'HyperparameterConfig':
        """从文件加载"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class RealtimeMonitor:
    """实时监控器"""
    
    def __init__(self, process: subprocess.Popen, config_file: str):
        self.process = process
        self.config_file = config_file
        self.metrics_history: List[TrainingMetrics] = []
        self.latest_metrics: Optional[TrainingMetrics] = None
        self.metrics_queue = queue.Queue()
        self.running = True
        
    def monitor_output(self):
        """监控输出线程"""
        for line in iter(self.process.stdout.readline, b''):
            if not self.running:
                break
            
            line_str = line.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue
            
            # 打印原始输出
            print(line_str, flush=True)
            
            # 解析指标
            metrics = TrainingMetrics.from_log_line(line_str)
            if metrics:
                self.metrics_history.append(metrics)
                self.latest_metrics = metrics
                self.metrics_queue.put(metrics)
            
            # 检查进程是否结束
            if self.process.poll() is not None:
                break
        
        self.running = False
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """获取最新的指标（非阻塞）"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return self.latest_metrics
    
    def wait_for_metrics(self, timeout: float = None) -> Optional[TrainingMetrics]:
        """等待新的指标（阻塞）"""
        try:
            return self.metrics_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class AdaptiveTuner:
    """自适应调参器"""
    
    def __init__(self,
                 config_file: str,
                 base_config: HyperparameterConfig,
                 min_epochs: int = 5,
                 patience: int = 3):
        """
        Args:
            config_file: 配置文件路径（用于与训练进程通信）
            base_config: 基础配置
            min_epochs: 最小训练轮数
            patience: 耐心值
        """
        self.config_file = config_file
        self.current_config = base_config
        self.min_epochs = min_epochs
        self.patience = patience
        
        self.best_metric_value = -float('inf')
        self.stagnation_count = 0
        self.tune_count = 0
        
        # 保存初始配置
        self.current_config.save_to_file(config_file)
    
    def should_tune(self, metrics: TrainingMetrics) -> bool:
        """判断是否需要调参"""
        if metrics.epoch < self.min_epochs:
            return False
        
        current_metric = metrics.best_metric
        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            self.stagnation_count = 0
            return False
        else:
            self.stagnation_count += 1
            return self.stagnation_count >= self.patience
    
    def tune(self, metrics: TrainingMetrics) -> HyperparameterConfig:
        """根据指标调整参数"""
        self.tune_count += 1
        new_config = HyperparameterConfig(**asdict(self.current_config))
        
        # 分析问题类型
        problem_type = self.analyze_problem(metrics)
        
        if problem_type == 'poor_tail':
            # 尾部类性能差
            logger.info(f"Epoch {metrics.epoch}: 检测到尾部类性能差 (tail_acc={metrics.tail_acc:.4f})")
            new_config.lambda_col = min(new_config.lambda_col * 1.3, 0.5)
            new_config.lambda_scl = min(new_config.lambda_scl * 1.3, 0.4)
            new_config.lambda_hier = min(new_config.lambda_hier * 1.1, 2.0)
            
        elif problem_type == 'poor_head':
            # 头部类性能差
            logger.info(f"Epoch {metrics.epoch}: 检测到头部类性能差 (head_acc={metrics.head_acc:.4f})")
            new_config.lambda_dol = min(new_config.lambda_dol * 1.2, 0.8)
            new_config.lambda_col = max(new_config.lambda_col * 0.8, 0.1)
            
        elif problem_type == 'high_loss':
            # 损失过高
            logger.info(f"Epoch {metrics.epoch}: 检测到损失过高 (loss={metrics.loss:.4f})")
            new_config.max_grad_norm = min(new_config.max_grad_norm * 1.2, 2.0)
            new_config.lambda_dol = max(new_config.lambda_dol * 0.9, 0.3)
            
        else:
            # 性能停滞
            logger.info(f"Epoch {metrics.epoch}: 检测到性能停滞 (stagnation={self.stagnation_count} epochs)")
            # 尝试微调权重
            import random
            adjustments = {
                'lambda_dol': random.choice([0.9, 1.0, 1.1]),
                'lambda_col': random.choice([0.9, 1.0, 1.1]),
                'lambda_hier': random.choice([0.95, 1.0, 1.05]),
                'lambda_scl': random.choice([0.9, 1.0, 1.1])
            }
            new_config.lambda_dol = max(0.3, min(0.8, new_config.lambda_dol * adjustments['lambda_dol']))
            new_config.lambda_col = max(0.1, min(0.5, new_config.lambda_col * adjustments['lambda_col']))
            new_config.lambda_hier = max(1.0, min(2.0, new_config.lambda_hier * adjustments['lambda_hier']))
            new_config.lambda_scl = max(0.1, min(0.4, new_config.lambda_scl * adjustments['lambda_scl']))
        
        self.current_config = new_config
        self.stagnation_count = 0
        
        # 保存新配置到文件
        new_config.save_to_file(self.config_file)
        
        logger.info(f"  新配置已保存: lambda_dol={new_config.lambda_dol:.3f}, "
                   f"lambda_col={new_config.lambda_col:.3f}, "
                   f"lambda_hier={new_config.lambda_hier:.3f}, "
                   f"lambda_scl={new_config.lambda_scl:.3f}, "
                   f"max_grad_norm={new_config.max_grad_norm:.3f}")
        
        return new_config
    
    def analyze_problem(self, metrics: TrainingMetrics) -> str:
        """分析当前训练问题"""
        if metrics.tail_acc < 0.3 and metrics.head_acc > 0.6:
            return 'poor_tail'
        elif metrics.head_acc < 0.5 and metrics.tail_acc > 0.4:
            return 'poor_head'
        elif metrics.loss > 2.0:
            return 'high_loss'
        else:
            return 'stagnation'


def run_realtime_tuning(base_cmd: List[str],
                       base_config: HyperparameterConfig,
                       config_file: str = '/tmp/auto_tune_config.json',
                       min_epochs: int = 5,
                       patience: int = 3,
                       check_interval: float = 30.0):
    """
    运行实时调参
    
    Args:
        base_cmd: 基础训练命令
        base_config: 初始配置
        config_file: 配置文件路径（用于与训练进程通信）
        min_epochs: 最小训练轮数
        patience: 耐心值
        check_interval: 检查间隔（秒）
    """
    # 保存初始配置
    base_config.save_to_file(config_file)
    logger.info(f"初始配置已保存到: {config_file}")
    
    # 创建调参器
    tuner = AdaptiveTuner(config_file, base_config, min_epochs, patience)
    
    # 启动训练进程
    logger.info(f"启动训练进程: {' '.join(base_cmd)}")
    process = subprocess.Popen(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=False
    )
    
    # 创建监控器
    monitor = RealtimeMonitor(process, config_file)
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor.monitor_output, daemon=True)
    monitor_thread.start()
    
    # 主循环：定期检查指标并调参
    try:
        while process.poll() is None:
            # 等待新指标
            metrics = monitor.wait_for_metrics(timeout=check_interval)
            
            if metrics:
                logger.info(f"收到新指标: Epoch {metrics.epoch}, "
                           f"best_metric={metrics.best_metric:.4f}, "
                           f"tail_acc={metrics.tail_acc:.4f}, "
                           f"head_acc={metrics.head_acc:.4f}")
                
                # 判断是否需要调参
                if tuner.should_tune(metrics):
                    tuner.tune(metrics)
                    logger.info(f"参数已更新，训练进程将在下一个epoch读取新配置")
            
            # 检查进程状态
            if process.poll() is not None:
                break
        
        # 等待进程结束
        return_code = process.wait()
        logger.info(f"训练进程结束，返回码: {return_code}")
        
        # 输出最终指标
        final_metrics = monitor.latest_metrics
        if final_metrics:
            logger.info(f"\n最终结果:")
            logger.info(f"  Epoch: {final_metrics.epoch}")
            logger.info(f"  Best Metric: {final_metrics.best_metric:.4f}")
            logger.info(f"  Head Acc: {final_metrics.head_acc:.4f}")
            logger.info(f"  Medium Acc: {final_metrics.medium_acc:.4f}")
            logger.info(f"  Tail Acc: {final_metrics.tail_acc:.4f}")
            logger.info(f"  Extreme Tail Acc: {final_metrics.extreme_tail_acc:.4f}")
            logger.info(f"  总调参次数: {tuner.tune_count}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止监控...")
        process.terminate()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description='实时监控终端输出并自动调参')
    parser.add_argument('--base_cmd', type=str, required=True,
                       help='基础训练命令（Python脚本路径）')
    parser.add_argument('--base_args', type=str, nargs='*', default=[],
                       help='基础训练参数')
    parser.add_argument('--config_file', type=str, default='/tmp/auto_tune_config.json',
                       help='配置文件路径（用于与训练进程通信）')
    parser.add_argument('--lambda_dol', type=float, default=0.5)
    parser.add_argument('--lambda_col', type=float, default=0.3)
    parser.add_argument('--lambda_hier', type=float, default=1.5)
    parser.add_argument('--lambda_scl', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--min_epochs', type=int, default=5,
                       help='最小训练轮数')
    parser.add_argument('--patience', type=int, default=3,
                       help='耐心值')
    parser.add_argument('--check_interval', type=float, default=30.0,
                       help='检查间隔（秒）')
    
    args = parser.parse_args()
    
    # 构建基础命令
    base_cmd = ['python', args.base_cmd] + args.base_args
    
    # 创建初始配置
    base_config = HyperparameterConfig(
        lambda_dol=args.lambda_dol,
        lambda_col=args.lambda_col,
        lambda_hier=args.lambda_hier,
        lambda_scl=args.lambda_scl,
        max_grad_norm=args.max_grad_norm
    )
    
    # 运行实时调参
    run_realtime_tuning(
        base_cmd=base_cmd,
        base_config=base_config,
        config_file=args.config_file,
        min_epochs=args.min_epochs,
        patience=args.patience,
        check_interval=args.check_interval
    )


if __name__ == '__main__':
    main()
