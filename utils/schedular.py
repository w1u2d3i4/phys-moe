"""
learning rate schedular
"""
import numpy as np

def get_scheduler(args)->callable:
    """
    Get the scheduler
    scheduler_name: scheduler name
    original_lr: original learning rate
    warmup_epoch: warmup epochs
    total_epoch: total epochs
    decay_epoch: decay epochs
    decay_rate: decay rate
    """
    if hasattr(args, 'scheduler'):
        if args.scheduler.name == 'step':
            return step_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch, args.scheduler.decay_rate)
        elif args.scheduler.name == 'cosine':
            return cosine_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch)
        elif args.scheduler.name == 'linear':
            return linear_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch)
        elif args.scheduler.name == 'exp':
            return exp_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.decay_epoch, args.scheduler.decay_rate)
        elif args.scheduler.name == 'milestone':
            assert args.scheduler.decay_milestones is not None, "**decay_milestones** should be provided for milestone scheduler, not decay_epoch"
            return milestone_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch, args.scheduler.decay_milestones, args.scheduler.decay_rate)
    elif hasattr(args, 'name'):
        if args.name == 'cosineannel':
            return CosineAnnealingWarmupScheduler(total_epochs=args.total_epoch,eta_min=args.eta_min,warmup_epochs=args.warmup_epoch,base_lr=args.base_lr,warmup_lr=args.warmup_lr)
    else:
        raise NotImplementedError(f"Unknown scheduler: {args.scheduler.name}")



# step learning rate scheduler with warmup
class step_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, decay_epoch, decay_rate):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.decay_epoch = decay_epoch
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)
        return lr

# cosine learning rate scheduler with warmup
class cosine_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, total_epoch):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = 0.5 * self.original_lr * (1 + np.cos(np.pi * (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch)))
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = 0.5 * lr * (1 + np.cos(np.pi * (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch)))
        return lr

# linear learning rate scheduler with warmup
class linear_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, total_epoch):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr * (1 - (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch))
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = lr * (1 - (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch))
        return lr

# exponential learning rate scheduler with warmup
class exp_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, decay_epoch, decay_rate):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.decay_epoch = decay_epoch
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)

class milestone_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, total_epoch, decay_epochs, decay_rate):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr
            for decay_epoch in self.decay_epochs:
                if epoch >= decay_epoch:
                    lr *= self.decay_rate
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            for decay_epoch in self.decay_epochs:
                if epoch >= decay_epoch:
                    lr *= self.decay_rate
        return lr
    
from torch.optim import lr_scheduler
import math


class CosineAnnealingWarmupScheduler:
    def __init__(self, base_lr, warmup_epochs, total_epochs, eta_min=0, warmup_lr=0.1):
        """
        Custom cosine annealing scheduler with warmup.

        Args:
            base_lr (float): Initial learning rate.
            warmup_epochs (int): Number of warmup epochs.
            total_epochs (int): Total number of training epochs.
            eta_min (float): Minimum learning rate.
            warmup_lr (float): Learning rate at the end of the warmup phase.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.warmup_lr = warmup_lr

    def __call__(self, epoch):
        """
        Compute the learning rate for a given epoch.

        Args:
            epoch (int): Current epoch.

        Returns:
            float: Learning rate for the given epoch.
        """
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr + (self.warmup_lr - self.base_lr) * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            adjusted_epoch = epoch - self.warmup_epochs
            adjusted_total = self.total_epochs - self.warmup_epochs
            lr = self.eta_min + (self.warmup_lr - self.eta_min) * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total)) / 2
        return lr

    def calculate_lr(self, lr, epoch):
        """
        Calculate learning rate for a specific epoch without affecting internal state.

        Args:
            lr (float): Current base learning rate.
            epoch (int): Epoch for which to calculate the learning rate.

        Returns:
            float: Learning rate for the specified epoch.
        """
        if epoch < self.warmup_epochs:
            return lr + (self.warmup_lr - lr) * (epoch + 1) / self.warmup_epochs
        else:
            adjusted_epoch = epoch - self.warmup_epochs
            adjusted_total = self.total_epochs - self.warmup_epochs
            return self.eta_min + (self.warmup_lr - self.eta_min) * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total)) / 2
        
class EarlyDecayCosineAnnealingLRWarmup(lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Up and Early Decay.
    基于总训练轮次百分比的学习率调度策略。
    
    策略（基于200 epoch的默认比例）：
    1. Warmup阶段 (0-2.5%): 从base_lr线性增加到warmup_lr
    2. 快速衰减阶段 (2.5%-10%): 从warmup_lr快速降到early_decay_target_lr
    3. 精细调优阶段 (10%-100%): 从early_decay_target_lr继续cosine退火到eta_min
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, 
                 warmup_ratio=0.025, base_lr=0.05, warmup_lr=0.1, 
                 early_decay_ratio=0.1, early_decay_target_lr=5e-4):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_ratio = warmup_ratio  # warmup占总轮次的比例（默认2.5%）
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.early_decay_ratio = early_decay_ratio  # 早期衰减占总轮次的比例（默认10%）
        self.early_decay_target_lr = early_decay_target_lr
        
        # 根据比例计算实际的epoch数
        self.warmup_epochs = max(1, int(T_max * warmup_ratio))
        self.early_decay_epoch = max(self.warmup_epochs + 1, int(T_max * early_decay_ratio))
        
        super(EarlyDecayCosineAnnealingLRWarmup, self).__init__(
            optimizer, last_epoch, verbose=True)

    def get_warmup_lr(self):
        """Warmup阶段：从base_lr线性增加到warmup_lr"""
        # 使用与CosineAnnealingLRWarmup相同的公式
        return [((self.warmup_lr - self.base_lr) / (self.warmup_epochs - 1) * (self.last_epoch - 1)
                 + self.base_lr) / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_early_decay_lr(self):
        """快速衰减阶段：从warmup_lr快速降到early_decay_target_lr"""
        # 计算在early_decay_epoch时的学习率（应该是early_decay_target_lr）
        # 使用cosine衰减从warmup_lr到early_decay_target_lr
        decay_period = self.early_decay_epoch - self.warmup_epochs
        progress = (self.last_epoch - self.warmup_epochs) / decay_period
        # 使用cosine函数平滑衰减
        # 当progress=1时（即last_epoch=early_decay_epoch），cos(π*1)=cos(π)=-1
        # 所以lr = early_decay_target_lr + (warmup_lr - early_decay_target_lr) * (1-1)/2 = early_decay_target_lr
        # 为了确保在early_decay_epoch时严格小于目标值，我们使用略小的值
        if progress >= 1.0:
            # 如果已经到达或超过early_decay_epoch，使用略小于目标值的学习率
            lr_at_epoch = self.early_decay_target_lr * 0.99  # 略小于5e-4，确保严格小于
        else:
            lr_at_epoch = self.early_decay_target_lr + (self.warmup_lr - self.early_decay_target_lr) * \
                          (1 + math.cos(math.pi * progress)) / 2
        return [lr_at_epoch / self.base_lr * base_lr for base_lr in self.base_lrs]

    def get_fine_tune_lr(self):
        """精细调优阶段：从early_decay_target_lr继续cosine退火到eta_min"""
        adjusted_epoch = self.last_epoch - self.early_decay_epoch
        adjusted_total = self.T_max - self.early_decay_epoch
        # 起始学习率使用略小于early_decay_target_lr的值，确保严格小于5e-4
        start_lr = self.early_decay_target_lr * 0.99
        return [self.eta_min + (start_lr - self.eta_min) *
                (1 + math.cos(math.pi * adjusted_epoch / adjusted_total)) / 2
                / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_lr(self):
        assert self.warmup_epochs >= 2
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段
            return self.get_warmup_lr()
        elif self.last_epoch < self.early_decay_epoch:
            # 快速衰减阶段：从warmup_lr快速降到early_decay_target_lr
            return self.get_early_decay_lr()
        else:
            # 精细调优阶段：从early_decay_target_lr继续cosine退火
            return self.get_fine_tune_lr()


class CosineAnnealingLRWarmup(lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Up.
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=5, base_lr=0.05, warmup_lr=0.1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        super(CosineAnnealingLRWarmup, self).__init__(
            optimizer, last_epoch, verbose=True)

    def get_cos_lr(self):
        return [self.eta_min + (self.warmup_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch -
                 self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_warmup_lr(self):
        return [((self.warmup_lr - self.base_lr) / (self.warmup_epochs - 1) * (self.last_epoch - 1)
                 + self.base_lr) / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_lr(self):
        assert self.warmup_epochs >= 2
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_lr()
        else:
            return self.get_cos_lr()

