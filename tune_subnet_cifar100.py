from app.arg_wrapper import ArgWrapper
from app.ldm.cifar100_ldm import CIFAR100LDM
from app.lm.subnet_cls_lm import SubnetClsLM
from app.trainer_wrapper import TrainerWrapper


kwargs = ArgWrapper(CIFAR100LDM, SubnetClsLM, TrainerWrapper)\
  .parse_kwargs()
  
ldm = CIFAR100LDM(**kwargs)
lm = SubnetClsLM.create(ldm, **kwargs)

TrainerWrapper(**kwargs).tune(lm, datamodule=ldm)