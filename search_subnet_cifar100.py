from app.ldm.cifar100_ldm import CIFAR100LDM
from app.lm.supernet_lm import SuperNetLM
from pytorch_lightning import seed_everything
from app.searcher import Searcher
from app.trainer import Trainer
from app.utils import get_kwargs, get_lm, get_logger, load_weights


kwargs = get_kwargs(CIFAR100LDM, SuperNetLM, Trainer, Searcher)
seed_everything(kwargs['seed'], workers=True)
lm, _ = get_lm(SuperNetLM, **kwargs)
load_weights(lm, **kwargs)
ldm = CIFAR100LDM(**kwargs)
logger = get_logger(lm, **kwargs)
trainer = Trainer(logger_=logger, **kwargs)
searcher = Searcher(trainer, logger, **kwargs)
searcher.search(lm, ldm, **kwargs)
