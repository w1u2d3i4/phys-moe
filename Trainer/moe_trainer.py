import torch
from loguru import logger
from Trainer.default_trainer import Default_Trainer,log_training_config,print_trainable_parameters
from utils.schedular import get_scheduler, CosineAnnealingLRWarmup, EarlyDecayCosineAnnealingLRWarmup
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast as autocast, GradScaler
from config.Arguments import Arguments
import nni

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class MoE_Trainer(Default_Trainer):
    def __init__(self,args,model,dataset,device):
        self.args = args   
        self.start_epoch = 0
        if hasattr(args,'base_args'):
            log_training_config(args,base_config=args.base_args)

        self.model = model
        print_trainable_parameters(self.model)
        self.dataset = dataset
        self.device = device
        self.model.to(self.device)
        self.gradients_experts = []

        if self.args.dataset.loader is not None:
            self.loader = {'train': self.dataset.train_loader, 'val': self.dataset.val_loader}
        else:
            weights_per_class = 1.0 /torch.tensor(args.label_dis) 
            weights_per_class /= weights_per_class.sum()
            
            targets_tensor = torch.tensor(self.dataset['train'].targets)
            if targets_tensor.dim() > 1:
                train_labels = torch.argmax(targets_tensor, dim=1)
            else:
                train_labels = targets_tensor
                
            sample_weights = weights_per_class[train_labels]
            
            self.sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(self.dataset['train']), replacement=True)

            self.loader = {'train': torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.args.train.batch_size, 
                                                                 num_workers=self.args.train.num_workers,
                                                                # sampler=self.sampler,
                                                                shuffle=True,
                                                                ),
                           'val': torch.utils.data.DataLoader(self.dataset['val'], batch_size=self.args.train.batch_size, 
                                                              shuffle=False, num_workers=self.args.train.num_workers),}
        

        self.scaler = GradScaler()
        
        # 根据优化器类型决定参数
        optimizer_name = self.args.train.optimizer
        optimizer_params = {
            'lr': self.args.train.lr,
            'weight_decay': self.args.train.weight_decay
        }
        # 只有SGD类优化器需要momentum参数
        if optimizer_name in ['SGD', 'RMSprop']:
            optimizer_params['momentum'] = 0.9
        # Adam类优化器可以设置betas
        elif optimizer_name in ['Adam', 'AdamW']:
            optimizer_params['betas'] = (0.9, 0.999)
        
        self.optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters(), **optimizer_params)
        
        # RT优化器
        rt_optimizer_name = self.args.train.rt_optimizer
        rt_optimizer_params = {
            'lr': self.args.train.lr_rt if hasattr(self.args.train,'lr_rt') else self.args.train.lr,
            'weight_decay': self.args.train.weight_decay
        }
        if rt_optimizer_name in ['SGD', 'RMSprop']:
            rt_optimizer_params['momentum'] = 0.9
        elif rt_optimizer_name in ['Adam', 'AdamW']:
            rt_optimizer_params['betas'] = (0.9, 0.999)
        
        self.rt_optimizer = getattr(torch.optim, rt_optimizer_name)(self.model.parameters(), **rt_optimizer_params)
        
        # 检查是否需要早期学习率衰减（基于总轮次的百分比）
        self.use_early_lr_decay = getattr(self.args.train, 'use_early_lr_decay', True)
        # 使用比例而不是固定epoch数（基于200 epoch的默认值：warmup=2.5%, early_decay=10%）
        self.warmup_ratio = getattr(self.args.train, 'warmup_ratio', 0.025)  # 2.5% = 5/200
        self.early_decay_ratio = getattr(self.args.train, 'early_decay_ratio', 0.1)  # 10% = 20/200
        self.early_decay_target_lr = getattr(self.args.train, 'early_decay_target_lr', 5e-4)
        
        # 如果启用早期衰减，记录日志
        if self.use_early_lr_decay:
            # 注意：T_max使用的是rt_start（第一阶段的总轮次），但比例是基于这个值计算的
            warmup_epochs = int(self.args.train.rt_start * self.warmup_ratio)
            early_decay_epoch = int(self.args.train.rt_start * self.early_decay_ratio)
            logger.info(f"使用早期学习率衰减策略（基于百分比）：")
            logger.info(f"  - 第一阶段总轮次: {self.args.train.rt_start} (rt_start)")
            logger.info(f"  - Warmup比例: {self.warmup_ratio*100:.1f}% (约{warmup_epochs} epochs)")
            logger.info(f"  - 早期衰减比例: {self.early_decay_ratio*100:.1f}% (约{early_decay_epoch} epochs)")
            logger.info(f"  - 目标学习率: {self.early_decay_target_lr:.6e}")
            logger.info(f"  - 确保Epoch >= {early_decay_epoch} 时，学习率 < {self.early_decay_target_lr:.6e}")
        
        if self.use_early_lr_decay:
            # 使用自定义的早期衰减调度器（基于百分比）
            self.scheduler = EarlyDecayCosineAnnealingLRWarmup(
                optimizer=self.optimizer,
                T_max=self.args.train.rt_start,
                eta_min=0.0,
                warmup_ratio=self.warmup_ratio,
                base_lr=self.args.train.lr,
                warmup_lr=0.15,
                early_decay_ratio=self.early_decay_ratio,
                early_decay_target_lr=self.early_decay_target_lr
            )
        else:
            # 使用原始调度器
            self.scheduler = CosineAnnealingLRWarmup(
                optimizer=self.optimizer,
                T_max=self.args.train.rt_start,
                eta_min=0.0,
                warmup_epochs=5,
                base_lr=self.args.train.lr,
                warmup_lr=0.15
            )
        
        self.rt_scheduler = CosineAnnealingLRWarmup(
            optimizer=self.rt_optimizer,
            T_max=self.args.train.epochs - self.args.train.rt_start,
            eta_min=0.0,
            warmup_epochs=5, #TODO：ori 5
            base_lr=self.args.train.lr_rt if hasattr(self.args.train,'lr_rt') else self.args.train.lr,
            warmup_lr=0.1
        )

        
        self.prepare_folder()
        if args.train.use_wandb:
            wandb.init(project=args.task, name=args.train.save_name)
            wandb.config.update(args)
            wandb.watch(self.model)
    
    def _get_model(self):
        """获取实际的模型对象，处理 DataParallel 包装的情况"""
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def train_one_epoch(self, epoch):
        self.model.train()
        if epoch >= self.args.train.rt_start:
            optimizer = self.rt_optimizer
        else:
            optimizer = self.optimizer
        with tqdm(self.loader['train'], desc=f"Epoch {epoch}", unit='batch') as pbar:
            for i, data in enumerate(pbar):

                optimizer.zero_grad()

                #with autocast():
                model = self._get_model()
                outputs = model.train_step(data, epoch >= self.args.train.rt_start,epoch)
                loss = outputs['loss']
                
                # 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Epoch {epoch} Batch {i}: Loss is NaN or Inf! Loss = {loss.item()}")
                    # 跳过这个batch
                    continue
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪，防止梯度爆炸
                # 使用scaler.unscale_来获取真实的梯度
                self.scaler.unscale_(optimizer)
                
                # 计算梯度范数（裁剪前，用于诊断）
                total_norm_before = 0.0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_before += param_norm.item() ** 2
                        param_count += 1
                total_norm_before = total_norm_before ** 0.5
                
                # 梯度裁剪：控制梯度爆炸
                # 默认阈值1.0，可根据训练情况调整（0.5-2.0都是合理范围）
                max_grad_norm = getattr(self.args.train, 'max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                # 计算裁剪后的梯度范数
                total_norm_after = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_after += param_norm.item() ** 2
                total_norm_after = total_norm_after ** 0.5
                
                # 只在检测到梯度异常时输出信息（使用裁剪前的梯度范数）
                if total_norm_before > 50.0:  # 降低警告阈值，更早发现问题
                    # 获取loss_dict用于输出
                    loss_dict = outputs.get('loss_dict', {})
                    loss_info = f"Loss = {loss.item():.6f}"
                    if loss_dict:
                        loss_breakdown = ", ".join([f"{k} = {v:.6f}" if isinstance(v, (int, float)) else f"{k} = {v}" 
                                                   for k, v in loss_dict.items()])
                        loss_info += f", Loss breakdown: [{loss_breakdown}]"
                    logger.warning(f"Epoch {epoch} Batch {i}: Gradient norm before clip = {total_norm_before:.6f}, "
                                 f"after clip = {total_norm_after:.6f} (max_norm={max_grad_norm}), "
                                 f"{loss_info}, Params with grad = {param_count}")
                elif total_norm_before < 1e-6:
                    # 获取loss_dict用于输出
                    loss_dict = outputs.get('loss_dict', {})
                    loss_info = f"Loss = {loss.item():.6f}"
                    if loss_dict:
                        loss_breakdown = ", ".join([f"{k} = {v:.6f}" if isinstance(v, (int, float)) else f"{k} = {v}" 
                                                   for k, v in loss_dict.items()])
                        loss_info += f", Loss breakdown: [{loss_breakdown}]"
                    logger.warning(f"Epoch {epoch} Batch {i}: Gradient norm too small: {total_norm_before:.6f} "
                                 f"(possible gradient vanishing), {loss_info}, Params with grad = {param_count}")
                self.scaler.step(optimizer)
                self.scaler.update()

                if hasattr(model, 'layer3s'):
                    self.gradients_experts = []
                    for expert in model.layer3s:
                        total_norm = 0.0
                        for p in expert.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        self.gradients_experts.append(torch.tensor(total_norm))

                if i % self.args.train.log_interval == 0:
                    pbar.set_postfix({'loss': loss.item()})
                    if self.args.train.use_wandb:
                        wandb.log({'batch': epoch * len(self.loader['train']) + i, 'training loss': loss.item()})
        if epoch >= self.args.train.rt_start:
            self.rt_scheduler.step()
        else:
            self.scheduler.step()
        return loss.item()
    
    def _eval(self,mode='val',epoch=0):
        """
        Evaluate the model
        """
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top2 = AverageMeter('Acc@2', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        sys_acc = AverageMeter('SysAcc', ':6.2f')
        recall = AverageMeter('Recall', ':6.2f')
        f1 = AverageMeter('F1', ':6.2f')
        head_acc = AverageMeter('HeadAcc', ':6.2f')
        medium_acc = AverageMeter('MediumAcc', ':6.2f')
        tail_acc = AverageMeter('TailAcc', ':6.2f')
        extreme_tail_acc = AverageMeter('ExtremeTailAcc', ':6.2f')
        exp_acc= []

        self.model.eval()
        if mode == 'val':
            loader = self.loader['val']
        elif mode == 'test':
            loader = self.loader['test']
        else:
            raise ValueError(f"Unknown mode: {mode}")
        with torch.no_grad():
            model = self._get_model()
            with tqdm(loader, desc=f"Evaluating {mode}", unit='batch') as pbar:
                for batch_idx, data in enumerate(pbar):
                    outputs = model.eval_step(data,rt=epoch>=self.args.train.rt_start)
                    losses.update(outputs['loss'].item(), data['intensity'].size(0))
                    top1.update(outputs['acc1'], data['intensity'].size(0))
                    top2.update(outputs['acc2'], data['intensity'].size(0))
                    top5.update(outputs['acc5'], data['intensity'].size(0))
                    sys_acc.update(outputs['sys_acc'], data['intensity'].size(0))
                    recall.update(outputs['recall'], data['intensity'].size(0))
                    f1.update(outputs['f1'], data['intensity'].size(0))
                    # 使用各类的样本数作为权重，而不是整个batch的样本数
                    head_count = outputs.get('head_count', 0)
                    medium_count = outputs.get('medium_count', 0)
                    tail_count = outputs.get('tail_count', 0)
                    extreme_tail_count = outputs.get('extreme_tail_count', 0)
                    
                    # 诊断：如果某些类的样本数为0，记录警告
                    if epoch < 5 and batch_idx == 0:
                        logger.info(f"Epoch {epoch} Batch 0: Sample counts - Head={head_count}, "
                                  f"Medium={medium_count}, Tail={tail_count}, ExtremeTail={extreme_tail_count}")
                    
                    if head_count > 0:
                        head_acc.update(outputs['head_acc'], head_count)
                    if medium_count > 0:
                        medium_acc.update(outputs['medium_acc'], medium_count)
                    if tail_count > 0:
                        tail_acc.update(outputs['tail_acc'], tail_count)
                    if extreme_tail_count > 0:
                        extreme_tail_acc.update(outputs.get('extreme_tail_acc', 0.0), extreme_tail_count)
                    exp_acc.append(outputs['exp_acc'])

                exp_acc = torch.stack(exp_acc)
                exp_acc = torch.mean(exp_acc,dim=0)
                exp_acc = exp_acc.cpu().numpy()
                exp_acc = exp_acc.tolist()
        if hasattr(self,'save_path'):
            import csv
            with open(f"{self.save_path}/{mode}_results.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(exp_acc)
            if not epoch>=self.args.train.rt_start:
                with open(f"{self.save_path}/{mode}_grad.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([grad.item() for grad in self.gradients_experts])
        model = self._get_model()
        model.on_eval_end()
        # 安全地提取值：如果avg是tensor则调用.item()，否则直接使用
        def safe_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        results = {
            'loss': {'ce':losses.avg}, 
            'metric': {
                'acc1': safe_item(top1.avg), 
                'acc2': safe_item(top2.avg),
                'acc5': safe_item(top5.avg), 
                'sys_acc': safe_item(sys_acc.avg),
                'recall': safe_item(recall.avg),
                'f1': safe_item(f1.avg),
                'head_acc': safe_item(head_acc.avg),
                'medium_acc': safe_item(medium_acc.avg),
                'tail_acc': safe_item(tail_acc.avg),
                'extreme_tail_acc': safe_item(extreme_tail_acc.avg)
            }
        }
        return results
    

    def train(self):
        
        """
        Train the model
        """
        import math
        import time
        self.best_val_metrics = -math.inf if self.args.train.core_metric[1] == 'maximize' else math.inf
        self.best_sys_acc = -math.inf  # 跟踪最佳晶系准确率
        self.best_epoch = 0
        # 保存best metric对应的所有指标
        self.best_metric_results = None
        # 用于学习率重启策略
        self.stagnation_epochs = 0  # 性能停滞的epoch数
        self.last_best_metric = -math.inf if self.args.train.core_metric[1] == 'maximize' else math.inf
        self.lr_restart_enabled = getattr(self.args.train, 'lr_restart_enabled', True)  # 是否启用学习率重启
        self.lr_restart_threshold = getattr(self.args.train, 'lr_restart_threshold', 10)  # 停滞多少epoch后重启
        self.lr_restart_factor = getattr(self.args.train, 'lr_restart_factor', 0.5)  # 重启时的学习率倍数
        self.load_checkpoint()
        for epoch in range(self.start_epoch,self.args.train.epochs):
            # 在rt_start时进行平滑过渡（方案1：平滑过渡）
            if epoch == self.args.train.rt_start:
                # 获取当前学习率（第一阶段scheduler的最后一个学习率）
                # 注意：需要在train_one_epoch之前获取，因为train_one_epoch会调用scheduler.step()
                # scheduler.step()在epoch结束时调用，所以epoch N的学习率是step()之后的值
                # 我们需要获取epoch (rt_start - 1) 结束时的学习率，即epoch rt_start开始时的学习率
                # 所以传入epoch - 1（即rt_start - 1）
                current_lr = self._calculate_current_lr(epoch - 1)
                
                # 如果scheduler有get_last_lr方法，直接使用（更准确）
                if hasattr(self.scheduler, 'get_last_lr'):
                    try:
                        scheduler_lr = self.scheduler.get_last_lr()[0]
                        # 如果scheduler返回的学习率合理，使用它
                        if scheduler_lr > 0:
                            current_lr = scheduler_lr
                    except:
                        pass  # 如果获取失败，使用计算值
                
                # 使用当前学习率的1/10，但至少为1e-5，作为rt_scheduler的起始学习率
                # 如果当前学习率已经很小（<1e-4），使用一个合理的最小值（1e-5），避免学习率为0
                if current_lr < 1e-6:
                    # 如果学习率已经接近0，使用一个合理的最小值
                    initial_rt_lr = 1e-5
                elif current_lr < 1e-4:
                    # 如果学习率很小但非零，保持当前学习率，避免突然增大
                    initial_rt_lr = current_lr
                else:
                    # 如果学习率较大，使用1/10，但至少1e-5
                    initial_rt_lr = max(current_lr * 0.1, 1e-5)
                
                logger.info(f"Epoch {epoch}: 切换到rt_scheduler（平滑过渡）")
                logger.info(f"  第一阶段最后学习率: {current_lr:.6e}")
                logger.info(f"  第二阶段起始学习率: {initial_rt_lr:.6e}")
                logger.info(f"  学习率变化比例: {initial_rt_lr/current_lr:.4f}x")
                
                # 更新rt_optimizer的学习率
                for param_group in self.rt_optimizer.param_groups:
                    param_group['lr'] = initial_rt_lr
                
                # 重新初始化rt_scheduler，使用平滑过渡的学习率
                # 注意：CosineAnnealingLRWarmup要求warmup_epochs >= 2
                # 我们设置warmup_epochs=2，但通过设置last_epoch跳过warmup阶段
                # warmup_lr等于base_lr，这样即使进入warmup阶段，学习率也不会改变
                self.rt_scheduler = CosineAnnealingLRWarmup(
                    optimizer=self.rt_optimizer,
                    T_max=self.args.train.epochs - self.args.train.rt_start,
                    eta_min=0.0,
                    warmup_epochs=2,  # 必须>=2，但通过last_epoch跳过
                    base_lr=initial_rt_lr,
                    warmup_lr=initial_rt_lr  # warmup_lr等于base_lr，这样warmup阶段不会改变学习率
                )
                # 手动设置last_epoch，跳过warmup阶段，直接进入cosine退火
                # 设置为warmup_epochs，这样get_lr()会直接进入cosine退火阶段
                self.rt_scheduler.last_epoch = 2  # 跳过warmup，直接进入cosine退火
            
            if epoch >= self.args.train.rt_start:
                for name, param in self.model.named_parameters():
                    if name[:14] != "rt_classifiers":
                        param.requires_grad = False
            
            self.adjust_learning_rate(epoch, self.args.train.lr)
            start_time = time.time()
            train_loss = self.train_one_epoch(epoch=epoch)
            
            val_result = self._eval(mode='val',epoch=epoch)
            end_time = time.time()
            val_dict = {'epoch':epoch,**val_result['loss'],**val_result['metric']}
            
            # 检查是否更新了best metric，如果是则保存所有指标
            current_metric = float(val_dict[self.args.train.core_metric[0]])
            is_best = False
            if self.args.train.core_metric[1] == 'maximize':
                is_best = current_metric > self.best_val_metrics
            else:
                is_best = current_metric < self.best_val_metrics
            
            if is_best:
                self.best_metric_results = val_dict.copy()
                self.stagnation_epochs = 0  # 重置停滞计数
            else:
                self.stagnation_epochs += 1
            
            # 学习率重启策略：当性能停滞时，重启学习率
            if self.lr_restart_enabled and epoch >= 20 and epoch < self.args.train.rt_start:
                if self.stagnation_epochs >= self.lr_restart_threshold:
                    # 获取当前学习率
                    current_lr = self.optimizer.param_groups[0]['lr']
                    # 重启学习率（使用初始学习率的一定比例）
                    restart_lr = self.args.train.lr * self.lr_restart_factor
                    # 确保重启后的学习率不会太小
                    restart_lr = max(restart_lr, 1e-5)
                    
                    if restart_lr > current_lr * 1.1:  # 只有当重启学习率明显大于当前学习率时才重启
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = restart_lr
                        
                        logger.info(f"Epoch {epoch}: 性能停滞{self.stagnation_epochs}个epoch，重启学习率")
                        logger.info(f"  当前学习率: {current_lr:.6e} -> 重启学习率: {restart_lr:.6e}")
                        logger.info(f"  当前best metric: {self.best_val_metrics:.4f}, 当前metric: {current_metric:.4f}")
                        
                        # 重置停滞计数，避免频繁重启
                        self.stagnation_epochs = 0
            
            if self._on_epoch_end(epoch, current_metric, self.args.train.core_metric[1]):
                break
            
            # 更新最佳晶系准确率
            current_sys_acc = float(val_dict['sys_acc'])
            if current_sys_acc > self.best_sys_acc:
                self.best_sys_acc = current_sys_acc
            
            val_dict['best metric'] = self.best_val_metrics
            val_dict['best sys_acc'] = self.best_sys_acc
            
            # 每10轮输出一次best metric对应的所有指标
            if epoch % 10 == 0 and self.best_metric_results is not None:
                best_results_str = {}
                for key, value in self.best_metric_results.items():
                    if isinstance(value, float):
                        best_results_str[key] = f"{value:.4f}"
                    else:
                        best_results_str[key] = value
                logger.info(f"Best Metric Results at Epoch {self.best_epoch}: {best_results_str}")
            if self.args.train.use_wandb:
                wandb.log(val_dict)
            if hasattr(self.args.train,'use_nni') and self.args.train.use_nni:
                nni.report_intermediate_result({'default':float(val_dict[self.args.train.core_metric[0]]),
                                                'cur_best':val_dict['best metric'],
                                                })
            for key,value in val_dict.items():
                if isinstance(value,float):
                    val_dict[key] = f"{value:.4f}"
            logger.info(f"Epoch {epoch}[{end_time-start_time:.2f}s]: {val_dict}")
            
        if hasattr(self.loader,'test'):
            self._eval(mode='test')
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best Metric: {self.best_val_metrics}")
        logger.info(f"Best Sys Acc: {self.best_sys_acc:.4f}")
        
        # 输出best metric对应的所有指标（包括三类精度）
        if self.best_metric_results is not None:
            logger.info(f"=== Best Metric Results (all metrics) at Epoch {self.best_epoch} ===")
            logger.info(f"Best Metric ({self.args.train.core_metric[0]}): {self.best_val_metrics:.4f}")
            logger.info(f"Loss: {self.best_metric_results.get('ce', 'N/A'):.4f}")
            logger.info(f"Acc1: {self.best_metric_results.get('acc1', 'N/A'):.4f}")
            logger.info(f"Acc2: {self.best_metric_results.get('acc2', 'N/A'):.4f}")
            logger.info(f"Acc5: {self.best_metric_results.get('acc5', 'N/A'):.4f}")
            logger.info(f"Sys Acc: {self.best_metric_results.get('sys_acc', 'N/A'):.4f}")
            logger.info(f"Recall: {self.best_metric_results.get('recall', 'N/A'):.4f}")
            logger.info(f"F1: {self.best_metric_results.get('f1', 'N/A'):.4f}")
            logger.info(f"Head Acc: {self.best_metric_results.get('head_acc', 'N/A'):.4f}")
            logger.info(f"Medium Acc: {self.best_metric_results.get('medium_acc', 'N/A'):.4f}")
            logger.info(f"Tail Acc: {self.best_metric_results.get('tail_acc', 'N/A'):.4f}")
            logger.info(f"Extreme Tail Acc: {self.best_metric_results.get('extreme_tail_acc', 'N/A'):.4f}")
            logger.info(f"================================================================")
        
        return_dict = {
            'best_epoch': self.best_epoch, 
            'best_metric': self.best_val_metrics, 
            'best_sys_acc': self.best_sys_acc,
            'best_metric_results': self.best_metric_results
        }
        if hasattr(self.args.train,'use_nni') and self.args.train.use_nni:
            nni.report_final_result({'default':self.best_val_metrics})
        return return_dict
    
    def adjust_learning_rate(self, epoch, lr):
        pass
    
    def _calculate_current_lr(self, epoch):
        """
        计算指定epoch的学习率（用于平滑过渡）
        这个方法模拟scheduler的计算逻辑，但不实际调用step()
        注意：这里需要模拟CosineAnnealingLRWarmup的实际计算逻辑
        """
        if epoch < self.args.train.rt_start:
            # 使用第一阶段scheduler的逻辑
            # 从代码看，warmup_lr=0.15是绝对值，不是比例
            base_lr = self.args.train.lr  # 1e-4
            warmup_lr = 0.15  # 绝对值0.15
            warmup_epochs = 5
            T_max = self.args.train.rt_start  # 180
            eta_min = 0.0
            
            if epoch < warmup_epochs:
                # warmup阶段：从base_lr线性增加到warmup_lr
                # 从get_warmup_lr看：((warmup_lr - base_lr) / (warmup_epochs - 1) * (last_epoch - 1) + base_lr)
                # last_epoch在epoch N结束时是N，所以计算epoch N的学习率时，last_epoch = N
                # 但我们需要计算epoch结束时的lr，所以用epoch（因为step()在epoch结束时调用）
                lr = ((warmup_lr - base_lr) / (warmup_epochs - 1) * (epoch - 1) + base_lr)
            else:
                # cosine退火阶段
                import math
                adjusted_epoch = epoch - warmup_epochs
                adjusted_total = T_max - warmup_epochs
                # 公式：eta_min + (warmup_lr - eta_min) * (1 + cos(...)) / 2 / base_lr * base_lr
                # 简化：eta_min + (warmup_lr - eta_min) * (1 + cos(...)) / 2
                lr = eta_min + (warmup_lr - eta_min) * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total)) / 2
            return lr
        else:
            # 第二阶段，返回rt_scheduler的学习率
            if hasattr(self, 'rt_scheduler') and hasattr(self.rt_scheduler, 'get_last_lr'):
                return self.rt_scheduler.get_last_lr()[0]
            return self.args.train.lr_rt if hasattr(self.args.train, 'lr_rt') else self.args.train.lr

    def save_checkpoint(self,epoch,filename='checkpoint.pt'):
        """
        Save the model checkpoint
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rt_optimizer': self.rt_optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'rt_scheduler': self.rt_scheduler.state_dict(),
            'epoch': epoch,
            'best_epoch': self.best_epoch,
            'best_sys_acc': self.best_sys_acc,
        }
        import os
        torch.save(checkpoint, os.path.join(self.save_path, filename))

    def load_checkpoint(self):
        """
        Load the model checkpoint
        """
        import math
        if self.args.train.resume is not None:
            checkpoint = torch.load(self.args.train.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.rt_optimizer.load_state_dict(checkpoint['rt_optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.rt_scheduler.load_state_dict(checkpoint['rt_scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['best_epoch']
            pre_val_results = self._eval(mode='val',epoch=checkpoint['epoch'])
            pre_val_results = {**pre_val_results['loss'], **pre_val_results['metric']}
            self.best_val_metrics = pre_val_results[self.args.train.core_metric[0]]
            # 从checkpoint恢复best_sys_acc，如果不存在则从当前评估结果获取
            if 'best_sys_acc' in checkpoint:
                self.best_sys_acc = checkpoint['best_sys_acc']
            else:
                self.best_sys_acc = pre_val_results.get('sys_acc', -math.inf)
            logger.info(f"Checkpoint loaded at epoch {self.start_epoch}, val metrics {self.args.train.core_metric[0]}:{self.best_val_metrics:.4f}, sys_acc:{self.best_sys_acc:.4f}")
        else:
            logger.info("No checkpoint loaded")

    def load_checkpoint_mid(self,path):
        """
        Load the model checkpoint
        """
        if path is not None:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.rt_optimizer.load_state_dict(checkpoint['rt_optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.rt_scheduler.load_state_dict(checkpoint['rt_scheduler'])

    def test(self):
        """
        Test the model
        """
        self.load_checkpoint()
        """
        Evaluate the model
        """
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        self.model.eval()
        
        loader = self.loader['val']
        mode = 'test'
        with torch.no_grad():
            model = self._get_model()
            with tqdm(loader, desc=f"Evaluating {mode}", unit='batch') as pbar:
                for _,data in enumerate(pbar):
                    outputs = model.eval_step(data,rt=1)
                    losses.update(outputs['loss'].item(), data['intensity'].size(0))
                    top1.update(outputs['acc1'], data['intensity'].size(0))
                    top5.update(outputs['acc5'], data['intensity'].size(0))


        model.on_eval_end()
        results = {'loss': {'ce':losses.avg}, 'metric': {'acc1': top1.avg.item(), 'acc5': top5.avg.item()}}