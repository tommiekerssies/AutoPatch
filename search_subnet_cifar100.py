from app.arg_wrapper import ArgWrapper
from app.ldm.cifar100_ldm import CIFAR100LDM
from app.lm.supernet_lm import SuperNetLM
from app.trainer_wrapper import TrainerWrapper


kwargs = ArgWrapper(CIFAR100LDM, SuperNetLM, TrainerWrapper)\
  .parse_kwargs()

ldm = CIFAR100LDM(**kwargs)
lm = SuperNetLM.create(ldm, **kwargs)

TrainerWrapper(**kwargs).search(lm, ldm, **kwargs)
