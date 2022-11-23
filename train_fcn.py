from pytorch_lightning import seed_everything
from app.arg_wrapper import ArgWrapper
from app.lightning_module.fcn import FCN
from app.lightning_data_module.aoi import AOI
from app.trainer_wrapper import TrainerWrapper


seed_everything(0, workers=True)
kwargs = ArgWrapper(AOI, FCN, TrainerWrapper).parse_kwargs()
ldm = AOI(**kwargs)
lm = FCN.create(**kwargs)
TrainerWrapper(**kwargs).fit(lm, ldm)
