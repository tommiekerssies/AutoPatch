from app.arg_wrapper import ArgWrapper
from app.lightning_module.aoi import AOI as AOI_LM
from app.lightning_data_module.aoi import AOI as AOI_LDM
from app.trainer_wrapper import TrainerWrapper


kwargs = ArgWrapper(AOI_LDM, AOI_LM, TrainerWrapper).parse_kwargs()
ldm = AOI_LDM(**kwargs)
lm = AOI_LM.create(ldm=ldm, **kwargs)
TrainerWrapper(**kwargs).fit(lm, ldm)
