from importlib.abc import Loader
import os

import pandas as pd
import seaborn as sn
import yaml

import torch
from IPython.core.display import display
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from preprocessing.dataset import train_transforms, test_transforms
from models.model import LitResnet

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path_dataset", type=str, default=os.environ.get("PATH_DATASETS", "."), help='Path to dataset')
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

config_file = open('configs/cifar.yaml', mode='r')
cfg = yaml.load(config_file, Loader=yaml.Fullloader)

seed_everything(cfg.exp_params.seed)
PATH_DATASETS = args.path_dataset
BATCH_SIZE = args.batch_size
NUM_WORKERS = int(os.cpu_count() / 2)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

model = LitResnet(lr=cfg.exp_params.LR, momentum=cfg.model_params.momentum, weight_decay=cfg.model_params.weight_decay)

trainer = Trainer(
    max_epochs=cfg.exp_params.max_epoch,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)


metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
sn.relplot(data=metrics, kind="line")