from app.ldm.cifar10_ldm import CIFAR10LDM
from app.lm.resnet_cls_lm import ResNetClsLM
from app.utils import get_kwargs, get_logger, get_lm, load_weights
from pytorch_lightning import seed_everything
from app.trainer import Trainer


kwargs = get_kwargs(CIFAR10LDM, ResNetClsLM, Trainer)
seed_everything(kwargs['seed'], workers=True)
lm, ckpt_path = get_lm(ResNetClsLM, **kwargs)
ldm = CIFAR10LDM(**kwargs)
load_weights(lm, **kwargs)
logger = get_logger(lm, **kwargs)
trainer = Trainer(monitor_='val_acc', logger_=logger, **kwargs)
trainer.fit(lm, datamodule=ldm, ckpt_path=ckpt_path)