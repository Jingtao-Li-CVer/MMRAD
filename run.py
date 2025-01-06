import argparse
import logging
import os
from typing import Dict, List, Tuple
import yaml
from tqdm import tqdm
from scipy import integrate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import datasets
import models
import numpy as np
from auroc import compute_auroc
import torch.nn.init as init
import utils
from sklearn.metrics import roc_auc_score, roc_curve
from statistics import mean
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist
from numpy import ndarray as NDArray
from random_seed import setup_seed

setup_seed(66666)

import torch

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29568'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
device = "cuda:0"


def make_data_loader(spec, local_rank, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders(local_rank):
    train_loader = make_data_loader(config.get('train_dataset'), local_rank, tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), local_rank, tag='val')
    return train_loader, val_loader



def test(model, loader, save_path) -> None:
    model.eval()
    model.training_mode = False

    artifacts_fp: Dict[str, List[NDArray]] = {
        "img": [],
        "reconst": [],
        "gt": [],
        "amap": [],
    }

    artifacts_oct: Dict[str, List[NDArray]] = {
        "img": [],
        "reconst": [],
        "gt": [],
        "amap": [],
    }

    artifacts_detection: Dict[str, List[NDArray]] = {
        "img": [],
        "reconst": [],
        "gt": [],
        "amap": [],
    }
    

    for batch in tqdm(loader):
        for k, v in batch.items():
            if k != 'file_name':
                batch[k] = v.to(device)

        with torch.no_grad():
            model.oct_input = batch['oct_inp']
            model.fp_input = batch['fp_inp']
            anomaly_map_fp, anomaly_map_oct, anomaly_score_fusion = model.forward()
            anomaly_map_fp = torch.sigmoid(anomaly_map_fp)
            anomaly_map_oct = torch.sigmoid(anomaly_map_oct)
            anomaly_score_fusion = torch.sigmoid(anomaly_score_fusion)
            
        fp_img = torch.nn.functional.interpolate(batch['fp_inp'], scale_factor=0.25)
        fp_gt = torch.nn.functional.interpolate(batch['fp_gt'], scale_factor=0.25, mode='nearest')
        anomaly_map_fp = torch.nn.functional.interpolate(anomaly_map_fp, scale_factor=0.25, mode='nearest')

        oct_img = torch.nn.functional.interpolate(batch['oct_inp'], scale_factor=0.25)
        oct_gt = torch.nn.functional.interpolate(batch['oct_gt'], scale_factor=0.25, mode='nearest')
        anomaly_map_oct = torch.nn.functional.interpolate(anomaly_map_oct, scale_factor=0.25, mode='nearest')

        artifacts_fp["amap"].extend(anomaly_map_fp.detach().cpu().numpy())
        artifacts_fp["img"].extend(fp_img.permute(0, 2, 3, 1).detach().cpu().numpy())
        artifacts_fp["reconst"].extend(fp_img.permute(0, 2, 3, 1).detach().cpu().numpy())
        artifacts_fp["gt"].extend(fp_gt.detach().cpu().numpy())

        artifacts_oct["amap"].extend(anomaly_map_oct.detach().cpu().numpy())
        artifacts_oct["img"].extend(oct_img.permute(0, 2, 3, 1).detach().cpu().numpy())
        artifacts_oct["reconst"].extend(oct_img.permute(0, 2, 3, 1).detach().cpu().numpy())
        artifacts_oct["gt"].extend(oct_gt.detach().cpu().numpy())

        artifacts_detection["amap"].extend(anomaly_score_fusion.detach().cpu().numpy())
        artifacts_detection["img"].extend(oct_img.permute(0, 2, 3, 1).detach().cpu().numpy())
        artifacts_detection["reconst"].extend(oct_img.permute(0, 2, 3, 1).detach().cpu().numpy())
        artifacts_detection["gt"].extend((batch['image_level_detection_gt']).detach().cpu().numpy())

    fp_amp = np.array(artifacts_fp["amap"])
    artifacts_fp["amap"] = (fp_amp - fp_amp.min())/(fp_amp.max() - fp_amp.min())

    fp_amp = np.array(artifacts_oct["amap"])
    artifacts_oct["amap"] = (fp_amp - fp_amp.min())/(fp_amp.max() - fp_amp.min())

    fp_amp = np.array(artifacts_detection["amap"])
    artifacts_detection["amap"] = (fp_amp - fp_amp.min())/(fp_amp.max() - fp_amp.min())

    try:
        # log('\n')
        auroc_fp = compute_auroc(0, np.array(artifacts_fp["amap"]), np.array(artifacts_fp["gt"]), save_path, image_level=False, log_info=log)
        # log('\n')
        # writer.flush()
        auroc_oct = compute_auroc(0, np.array(artifacts_oct["amap"]), np.array(artifacts_oct["gt"]), save_path, image_level=False, log_info=log)
        # log('\n')
        # writer.flush()
        auroc_detection = compute_auroc(0, np.array(artifacts_detection["amap"]), np.array(artifacts_detection["gt"]), save_path, image_level=True, log_info=log)

    except ValueError:
        print('ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.')
        auroc_detection = 0.0
        auroc_oct = 0.0
        auroc_fp = 0.0
        pass

    # savefig_v1(0, artifacts_fp["img"],   artifacts_fp["reconst"], artifacts_fp["gt"], artifacts_fp["amap"], './evalute_pixel_level',\
    #       np.array([0.38812142, 0.28388125, 0.15787678]), np.array([0.23149024, 0.16714649, 0.10265642]), 255, tag='FP')

    # savefig_v1(0, artifacts_oct["img"],   artifacts_oct["reconst"], artifacts_oct["gt"], artifacts_oct["amap"], './evalute_pixel_level',\
    #         np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]), 255, tag='OCT')

    return auroc_fp + auroc_oct + auroc_detection
    # return auroc_detection


def prepare_training(local_rank):
    if config.get('resume') is not None:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler



def train(local_rank, train_loader, model):
    global device
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    # device = torch.device(device)
    loss_list = []

    for batch in tqdm(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        model.set_input(batch['fp_inp'], batch['fp_gt'], batch['oct_inp'], batch['oct_gt'], batch['image_level_detection_gt'])

        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss)
        loss_list.extend(batch_loss)
        loss_len = len(loss_list)
        if loss_len % 10 == 0:
            print("training loss: " + str(torch.tensor(loss_list[-10:]).mean()))
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def random_initialize_transformer_model(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            init.normal_(param, mean=0, std=0.01)

def main(local_rank, config_, save_path, args, ):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders(local_rank)
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training(local_rank)
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module
    random_initialize_transformer_model(model)

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "LowLevelPromptGenerator" not in name:
            para.requires_grad_(False)
        if 'low_level_extractor' in name:
            para.requires_grad_(False)

    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        model.training_mode = True

        train_loss_G = train(local_rank,train_loader, model)
        torch.cuda.empty_cache()
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if local_rank == 0 and (epoch_val is not None) and (epoch % epoch_val == 0):
            scoreDF_sum = test(model, val_loader, save_path)

            if local_rank == 0:
                if (scoreDF_sum) > max_val_v:
                    max_val_v = scoreDF_sum
                    save(config, model, save_path, 'best' + str(scoreDF_sum))

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


def cleanup():
    dist.destroy_process_group()


def start(local_rank, world_size):
    dist.init_process_group('nccl', rank=local_rank, world_size=world_size)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/medical-sam-vit-b.yaml")
    parser.add_argument('--name', default='test_name')
    parser.add_argument('--tag', default='test_tag')
    parser.add_argument("--local_rank", type=int, default=3, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./output', save_name)

    main(local_rank, config, save_path, args=args)
    cleanup()

def run(start, word_size):
    mp.spawn(start, args=(word_size, ), nprocs=word_size, join=True)

if __name__ == "__main__":
    run(start, 1)