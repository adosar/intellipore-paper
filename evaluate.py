import os
import json
import yaml
import random
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError
from jsonargparse import ArgumentParser
from lightning.pytorch import Trainer, seed_everything
from aidsorb.data import get_names, PCDDataset as VoxelsDataset
from aidsorb.litmodules import PCDLit as VoxelsLit
from torchvision.transforms.v2 import Compose, RandomChoice
from retnext.transforms import (
        AddChannelDim, RandomRotate90, RandomFlip,
        RandomReflect, ClipScaleVoxels,
        )
from retnext.modules import IntelliPore

from litmodules import MultiTaskLit


def main_cli():
    parser = ArgumentParser(prog='Finetune or train IntelliPore from scratch.')
    parser.add_class_arguments(Trainer, 'trainer')
    parser.add_argument('--voxels_path', type=str)
    parser.add_argument('--labels_path', type=str)
    parser.add_argument('--index_col', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--train_sizes', type=list)
    #parser.add_argument('--n_runs', type=int)
    parser.add_argument('--n_runs', type=list)
    parser.add_argument('--n_frozen_layers', type=int, default=6)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--ft_backbone_lr', type=float, default=1e-3)
    parser.add_argument('--ft_head_lr', type=float, default=1e-2)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--bias_init', type=bool, default=True)
    parser.add_argument('-c', '--config', action='config')

    return parser


def get_dataloaders(
        *,
        train_size,
        voxels_path,
        labels_path,
        index_col,
        target,
        train_batch_size,
        ):

    train_names, val_names, test_names = [
            list(get_names(os.path.join(Path(voxels_path).parent, f'{stg}.json')))
            for stg in ['train', 'validation', 'test']
            ]

    df_labels = pd.read_csv(
            labels_path,
            index_col=index_col,
            usecols=[index_col, target],
            )

    # Some tasks from SpbNet benchmarks contain NaNs.
    train_names, val_names, test_names = [
            list(df_labels.loc[stg_names].dropna().index)
            for stg_names in [train_names, val_names, test_names]
            ]

    # Use same preprocessing and augmentation as the pretrained models.
    transform_train = Compose([
        AddChannelDim(), ClipScaleVoxels(),
        RandomChoice([
            torch.nn.Identity(),
            RandomRotate90(),
            RandomFlip(),
            RandomReflect()
            ])
        ])
    transform_eval = Compose([AddChannelDim(), ClipScaleVoxels()])

    config_dataloaders = dict(num_workers=8, persistent_workers=True)

    if train_size is None:  # Use all training data as in SpbNet.
        train_size = len(train_names)

    print('Number of train data wo NaNs: ', train_size)

    train_set = VoxelsDataset(
            random.sample(train_names, k=train_size),  # Random subset of train_size.
            path_to_X=voxels_path,
            path_to_Y=labels_path,
            index_col=index_col,
            labels=[target],
            transform_x=transform_train,
    )

    val_set = VoxelsDataset(
            val_names,
            path_to_X=voxels_path,
            path_to_Y=labels_path,
            index_col=index_col,
            labels=[target],
            transform_x=transform_eval,
    )

    test_set = VoxelsDataset(
            test_names,
            path_to_X=voxels_path,
            path_to_Y=labels_path,
            index_col=index_col,
            labels=[target],
            transform_x=transform_eval,
    )

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True, **config_dataloaders)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, **config_dataloaders)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, **config_dataloaders)

    return train_loader, val_loader, test_loader


def custom_optimizer(self):
    return torch.optim.Adam([
        {'params': self.model.backbone.parameters(), 'lr': bb_lr},
        {'params': self.model.head.parameters(), 'lr': head_lr},
        ])


if __name__ == '__main__':
    parser = main_cli()
    cfg = parser.parse_args()

    results = {s: {} for s in cfg.train_sizes}

    # Add target and ckpt name to dirname.
    cfg.trainer.default_root_dir += f'from_{cfg.ckpt_path.replace("/", "_")}'
    cfg.trainer.default_root_dir += f'-to_{cfg.target.replace("/", "_")}/'

    del cfg.config  # Not JSON serializable.

    default_root_dir = cfg.trainer.default_root_dir

    # Seeds random module as well.
    seed_everything(42, workers=True)

    if len(cfg.train_sizes) != len(cfg.n_runs):
        raise ValueError("'train_sizes' and 'n_runs' must have the same size")

    print(f'\033[31;1mTraining from: {cfg.ckpt_path}\033[0m')

    for size, n_runs in zip(cfg.train_sizes, cfg.n_runs):
        print(f'\033[32;1mTraining set size: {size}\033[0m')

        cfg.trainer.default_root_dir = default_root_dir + f'{size}/'  # New dir for each size.

        for run in range(n_runs):
            print(f'\033[34;1mRun number: {run}\033[0m')

            results[size][run] = {}

            # New objects for each experiment.
            init = parser.instantiate_classes(cfg)

            criterion = torch.nn.MSELoss()
            metric = MetricCollection([R2Score(), MeanAbsoluteError()])

            if cfg.ckpt_path == 'scratch':
                litmodel = VoxelsLit(
                        model=IntelliPore(n_outputs=1),  # Randomly initialized model.
                        criterion=criterion,
                        metric=metric,
                        )
            else:
                # Model with pretrained backbone.
                backbone = MultiTaskLit.load_from_checkpoint(cfg.ckpt_path).backbone
                backbone[:cfg.n_frozen_layers].requires_grad_(False)
                backbone[:cfg.n_frozen_layers].eval()

                model = IntelliPore(n_outputs=1)
                model.backbone = backbone

                # Use different learning rates for backbone and head.
                bb_lr = cfg.ft_backbone_lr
                head_lr = cfg.ft_head_lr

                # Monkey patch optimizer.
                VoxelsLit.configure_optimizers = custom_optimizer

                litmodel = VoxelsLit(model=model, criterion=criterion, metric=metric)

            train_loader, val_loader, test_loader = get_dataloaders(
                    train_size=size,
                    voxels_path=cfg.voxels_path,
                    labels_path=cfg.labels_path,
                    index_col=cfg.index_col,
                    target=cfg.target,
                    train_batch_size=cfg.train_batch_size,
                    )

            # Initialize bias of last layer with y_mean to ease optimization.
            if cfg.bias_init is True:
                train_names = list(train_loader.dataset.pcd_names)
                y_train_mean = train_loader.dataset.Y.loc[train_names].mean().item()
                torch.nn.init.constant_(litmodel.model.head.bias, y_train_mean)

            trainer = init.trainer
            trainer.fit(litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)

            results[size][run] = trainer.test(litmodel, dataloaders=test_loader, ckpt_path='best')[0]

            del train_loader, val_loader, test_loader

        with open(cfg.trainer.default_root_dir + 'config.yaml', 'w') as fhand:
            yaml.dump(cfg.as_dict(), fhand)

    with open(default_root_dir + 'results.json', 'w') as fhand:
        json.dump(results, fhand, indent=4)
