from app.lm.resnet_cls_lm import ResNetClsLM
from app.utils import get_kwargs, get_logger, get_lm, load_weights
from app.ldm.cifar100_ldm import CIFAR100LDM
from pytorch_lightning import seed_everything
from app.trainer import Trainer


kwargs = get_kwargs(CIFAR100LDM, ResNetClsLM, Trainer)
seed_everything(kwargs['seed'], workers=True)
lm, ckpt_path = get_lm(ResNetClsLM, **kwargs)
ldm = CIFAR100LDM(**kwargs)
load_weights(lm, **kwargs)
trainer = Trainer(**kwargs)
trainer.tune(lm, datamodule=ldm, **kwargs)