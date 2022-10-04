from app.lm.supernet_lm import SuperNetLM
from app.utils import get_kwargs, get_lm, \
  load_subnet_weights
from app.ldm.cifar100_ldm import CIFAR100LDM
from app.lm.subnet_cls_lm import SubnetClsLM
from pytorch_lightning import seed_everything
from app.trainer import Trainer


kwargs = get_kwargs(CIFAR100LDM, SubnetClsLM, Trainer)
seed_everything(kwargs['seed'], workers=True)
lm, ckpt_path = get_lm(SubnetClsLM, **kwargs)
ldm = CIFAR100LDM(**kwargs)
load_subnet_weights(SuperNetLM, lm, ldm, **kwargs)
trainer = Trainer(**kwargs)
trainer.tune(lm, datamodule=ldm)