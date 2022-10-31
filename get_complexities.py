from app.arg_wrapper import ArgWrapper
from app.lightning_module.supernet import SuperNet


# TODO: don't require specific architecture args to be passed to AOI_LM
kwargs = ArgWrapper(SuperNet).parse_kwargs()
lm = SuperNet.create(**kwargs)
lm.get_complexities()
