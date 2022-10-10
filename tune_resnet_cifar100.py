from app.arg_wrapper import ArgWrapper
from app.lm.resnet_cls_lm import ResNetClsLM
from app.ldm.cifar100_ldm import CIFAR100LDM
from app.trainer_wrapper import TrainerWrapper


kwargs = ArgWrapper(CIFAR100LDM, ResNetClsLM, TrainerWrapper)\
  .parse_kwargs()
  
ldm = CIFAR100LDM(**kwargs)
lm = ResNetClsLM.create(ldm, **kwargs)

TrainerWrapper(**kwargs).tune(lm, ldm)