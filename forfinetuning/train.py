import os
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Temporary folder')
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
args = parser.parse_args()

config = Config()

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = config.mixed_precision
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=False),
            utils.GradScalerKwargs(backoff_factor=0.5)],
    )
    args.dist = False

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    if args.use_accelerate:
        device = accelerator.local_process_index
    else:
        device = config.device

if config.rand_seed:
    set_seed(config.rand_seed + device)

epoch_st = 1
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
# if args.use_accelerate and accelerator.mixed_precision != 'no':
#     config.compile = False
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)

from dataset import custom_collate_fn

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=is_train, sampler=None, drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    val_loader = prepare_dataloader(
        MyData(datasets=config.validation_set, data_size=config.size, is_train=False),
        batch_size=1, to_be_distributed=False, is_train=False
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    print(len(val_loader), "batches of validation data.")
    return train_loader, val_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    elif config.model == 'BiRefNetC2F':
        model = BiRefNetC2F(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    
    # Setting optimizer
    if config.optimizer == 'AdamW':
        # optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = data_loaders
        if args.use_accelerate:
            self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()
        
        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)

        # print(f"[DEBUG tra] scaled_preds type: {type(scaled_preds)}")
        # print(f"[DEBUG tra] scaled_preds content: {scaled_preds}")
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels)
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        if epoch > args.epochs + config.finetune_last_epochs:
            if config.task in ['Matting', 'custom']:
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9

        for batch_idx, batch in enumerate(self.train_loader):
            # with nullcontext if not args.use_accelerate or accelerator.gradient_accumulation_steps <= 1 else accelerator.accumulate(self.model):
            self._train_batch(batch)
            # Logger
            if (epoch < 2 and batch_idx < 100 and batch_idx % 20 == 0) or batch_idx % max(100, len(self.train_loader) / 100 // 100 * 100) == 0:
                info_progress = f'Epoch[{epoch}/{args.epochs}] Iter[{batch_idx}/{len(self.train_loader)}].'
                info_loss = 'Training Losses:'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += f' {loss_name}: {loss_value:.5g} |'
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = f'@==Final== Epoch[{epoch}/{epoch}]  Training Loss: {self.loss_log.avg:.5g}  '
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg
    
    @torch.no_grad()
    def validate_epoch(self, epoch=None):
        self.model.eval()
        self.loss_dict = {}
        val_loss_log = AverageMeter()

        for batch_idx, batch in enumerate(self.val_loader):
            if args.use_accelerate:
                inputs = batch[0].to(device)
                gts = batch[1].to(device)
            else:
                inputs = batch[0].to(device)
                gts = batch[1].to(device)

            # forward pass
            outputs = self.model(inputs)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                scaled_preds, _ = outputs
            else:
                scaled_preds = outputs

            # print(f"[DEBUG] scaled_preds type: {type(scaled_preds)}")
            # print(f"[DEBUG] scaled_preds content: {scaled_preds}")

            # loss 계산
            loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
            self.loss_dict.update(loss_dict_pix)
            self.loss_dict['loss_pix'] = loss_pix.item()

            val_loss_log.update(loss_pix.item(), inputs.size(0))

        # 최종 로그
        if epoch is not None:
            logger.info(f'@==Final== Epoch[{epoch}] Validation Loss: {val_loss_log.avg:.5g}')

        return val_loss_log.avg



# def main():

#     trainer = Trainer(
#         data_loaders=init_data_loaders(to_be_distributed),
#         model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
#     )

#     for epoch in range(epoch_st, args.epochs+1):
#         train_loss = trainer.train_epoch(epoch)
#         # Save checkpoint
#         if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
#             if args.use_accelerate:
#                 state_dict = trainer.model.state_dict()
#             else:
#                 state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
#             torch.save(state_dict, os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)))
#     if to_be_distributed:
#         destroy_process_group()
def main():
    train_loader, val_loader = init_data_loaders(to_be_distributed)
    trainer = Trainer(
        data_loaders=train_loader,
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )
    trainer.val_loader = val_loader  # 추가

    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = -1

    for epoch in range(epoch_st, args.epochs+1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate_epoch()

        logger.info(f"Validation Loss: {val_loss:.5f}")

        # Best 모델 갱신 (파일로 저장하지 않음)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = (
                trainer.model.state_dict()
                if args.use_accelerate
                else (trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict())
            )
            best_epoch = epoch
            logger.info(f"==> New best model found at epoch {epoch} (val loss: {val_loss:.5f})")

        # 주기적 저장 (이전과 동일)
        # if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
        #     torch.save(
        #         trainer.model.state_dict(),
        #         os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth')
        #     )

    # 마지막에 best model 저장
    if best_state_dict is not None:
        best_path = os.path.join(args.ckpt_dir, f'best_model_epoch_{best_epoch}.pth')
        torch.save(best_state_dict, best_path)
        logger.info(f"==> Final best model saved to {best_path}")

    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()

#         # Best 모델 저장
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(trainer.model.state_dict(), os.path.join(args.ckpt_dir, f'best_model_epoch_{epoch}.pth'))
#             logger.info("==> Best model saved!")

#         # 주기적으로 저장
#         if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
#             torch.save(trainer.model.state_dict(), os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth'))

#     if to_be_distributed:
#         destroy_process_group()


# if __name__ == '__main__':
#     main()
    