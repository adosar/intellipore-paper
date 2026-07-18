import os
import json
from pathlib import Path
from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Adam
from torch.nn import Linear, ModuleDict, MSELoss, Sequential, Module
from torch import nn
from torch.utils.data import DataLoader
import lightning as L
from torchmetrics.functional import r2_score
from aidsorb.data import get_names, PCDDataset


class MultiTaskLit(L.LightningModule):
    def __init__(
            self,
            backbone: Module,
            data_config: dict[str, dict],
            data_weights: dict[str, float],
            train_batch_size: int,
            adam_lr: float = 1e-3,
            ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.criterion = MSELoss()
        self.backbone = backbone.backbone

        self.train_bs = train_batch_size  # Assumes `drop_last` is enabled

        self.heads = ModuleDict({
            db: nn.Linear(128, len(data_config[db]['labels']))
            for db in data_config.keys()
            })

    def forward(self, x: Tensor, db: str) -> Tensor:
        return self.heads[db](self.backbone(x))

    def compute_loss(self, batch: Tensor) -> Tensor:
        # Process all inputs in single forward pass (BN layers)
        z_all = self.backbone(torch.cat([b for b, _ in batch.values()]))
        z_dict = {
                db: z_all[i * self.train_bs : (i+1) * self.train_bs]
                for i, db in enumerate(batch.keys())
                }

        # Calculate total db-weighted loss
        return torch.sum(torch.stack([
            self.criterion(input=self.heads[db](z_dict[db]), target=batch[db][1])
            * self.hparams.data_weights[db]
            for db in batch.keys()
            ]))

    def training_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch)

        self.log('train_mse', total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        db = list(self.hparams.data_config.keys())[dataloader_idx]

        x, y = batch
        preds = self(x, db)

        self.val_preds[dataloader_idx].append(preds)
        self.val_targets[dataloader_idx].append(y)

    def on_validation_epoch_start(self):
        self.val_preds = {
                loader_idx: []
                for loader_idx in range(len(self.hparams.data_config.keys()))
                }
        self.val_targets = {
                loader_idx: []
                for loader_idx in range(len(self.hparams.data_config.keys()))
                }

    def on_validation_epoch_end(self):
        all_metrics = []
        for loader_idx, db in enumerate(self.hparams.data_config.keys()):
            preds = torch.cat(self.val_preds[loader_idx])
            targets = torch.cat(self.val_targets[loader_idx])
            metric = r2_score(preds=preds, target=targets).item()
            all_metrics.append(metric)
            self.log(f'val_r2/{db}', metric, prog_bar=True)

        avg_metric = sum(all_metrics) / len(all_metrics)
        self.log('val_r2', avg_metric, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        db = list(self.hparams.data_config.keys())[dataloader_idx]

        x, y = batch
        preds = self(x, db)

        self.test_preds[dataloader_idx].append(preds)
        self.test_targets[dataloader_idx].append(y)

    def on_test_epoch_start(self):
        self.test_preds = {
                loader_idx: []
                for loader_idx in range(len(self.hparams.data_config.keys()))
                }
        self.test_targets = {
                loader_idx: []
                for loader_idx in range(len(self.hparams.data_config.keys()))
                }

    def on_test_epoch_end(self):
        tasks_scores = {}
        for loader_idx, db in enumerate(self.hparams.data_config.keys()):
            preds = torch.cat(self.test_preds[loader_idx])
            targets = torch.cat(self.test_targets[loader_idx])

            scores = r2_score(preds=preds, target=targets, multioutput='raw_values')

            labels = self.hparams.data_config[db]['labels']
            tasks_scores[db] = {lbl: s.item() for lbl, s in zip(labels, scores)}

        print(json.dumps(tasks_scores, indent=4))
        with open(f'{self.trainer.default_root_dir}/test_results.json', 'w') as fhand:
            json.dump(tasks_scores, fhand, indent=4)

        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.adam_lr)


class MultiTaskDM(L.LightningDataModule):
    def __init__(
            self,
            data_config: dict[str, dict],
            transform_train: Callable,
            transform_eval: Callable,
            collate_fn: Callable | None = None,
            train_batch_size: int = 32,
            eval_batch_size: int = 256,
            ) -> None:

        super().__init__()
        self.save_hyperparameters()

        # https://github.com/Lightning-AI/pytorch-lightning/issues/20726#issuecomment-2840880104
        self.transform_train = transform_train
        self.transform_eval = transform_eval
        self.collate_fn = collate_fn

    def setup(self, stage: str | None = None):
        if stage == 'fit':
            self._setup_datasets('train')
            self._setup_datasets('validation')

        elif stage == 'validate':
            self._setup_datasets('validation')

        elif stage == 'test':
            self._setup_datasets('test')

        elif stage is None:
            self._setup_datasets('train')
            self._setup_datasets('validation')
            self._setup_datasets('test')

    def _setup_datasets(self, mode: str) -> None:
        if mode == 'train':
            transform_x = self.transform_train
        else:
            transform_x = self.transform_eval

        datasets = {
                db: PCDDataset(
                    **cfg,
                    pcd_names=get_names(os.path.join(Path(cfg['path_to_X']).parent, f'{mode}.json')),
                    transform_x=transform_x
                    )
                for db, cfg in self.hparams.data_config.items()
                }

        setattr(self, f'{mode}_datasets', datasets)

    def train_dataloader(self):
        dataloaders = {
                db: DataLoader(
                    self.train_datasets[db],
                    shuffle=True,
                    drop_last=True,
                    batch_size=self.hparams.train_batch_size,
                    num_workers=12,
                    persistent_workers=True,
                    collate_fn=self.collate_fn,
                    )
                for db in self.hparams.data_config.keys()
                }

        return dataloaders

    def val_dataloader(self):
        dataloaders = {
                db: DataLoader(
                    self.validation_datasets[db],
                    shuffle=False,
                    drop_last=False,
                    batch_size=self.hparams.eval_batch_size,
                    num_workers=8,
                    persistent_workers=True,
                    collate_fn=self.collate_fn,
                    )
                for db in self.hparams.data_config.keys()
                }

        return dataloaders

    def test_dataloader(self):
        dataloaders = {
                db: DataLoader(
                    self.test_datasets[db],
                    shuffle=False,
                    drop_last=False,
                    batch_size=self.hparams.eval_batch_size,
                    num_workers=8,
                    persistent_workers=True,
                    collate_fn=self.collate_fn,
                    )
                for db in self.hparams.data_config.keys()
                }

        return dataloaders
