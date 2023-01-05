from app.arg_wrapper import ArgWrapper
from app.lightning_module.supernet import SuperNet
from app.lightning_data_module.mvtec import MVTec
from app.trainer_wrapper import TrainerWrapper


kwargs = ArgWrapper(MVTec, SuperNet, TrainerWrapper).parse_kwargs()
ldm = MVTec(**kwargs)
lm = SuperNet.create(**kwargs)
TrainerWrapper(**kwargs).search(lm, ldm)
