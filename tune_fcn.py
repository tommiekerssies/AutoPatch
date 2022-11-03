from app.arg_wrapper import ArgWrapper
from app.lightning_module.multi_label_sem_seg.fcn import FCN
from app.lightning_data_module.aoi import AOI
from app.trainer_wrapper import TrainerWrapper


kwargs = ArgWrapper(AOI, FCN, TrainerWrapper).parse_kwargs()
ldm = AOI(**kwargs)
lm = FCN.create(**kwargs)
TrainerWrapper(**kwargs).tune(lm, ldm, **kwargs)
