from lib.gaia.base_rule import SAMPLE_RULES
import lib.gaia.eval_rule as eval_rule
from mmcv.utils import build_from_cfg
from lib.gaia.model_space_manager import ModelSpaceManager
from os.path import join


class Searcher:
  def __init__(self, trainer, logger_, work_dir, model_space_file,
               min_gflops, max_gflops, **kwargs):
    self.trainer = trainer
    self.logger = logger_
    self.work_dir = work_dir
    self.model_space_file = model_space_file
    self.min_gflops = min_gflops
    self.max_gflops = max_gflops
  
  def search(self, lm, ldm):
    model_space = ModelSpaceManager.load(join(self.work_dir, self.model_space_file))
    model_sampling_rule = build_from_cfg(dict(type='eval', func_str=
      f'lambda x: x[\'overhead.flops\'] >= {self.min_gflops * 1e9} ' +
      f'and x[\'overhead.flops\'] < {self.max_gflops * 1e9}'
    ), SAMPLE_RULES)

    subnet_candidates = model_space.ms_manager.apply_rule(model_sampling_rule)
    subnet_candidates = subnet_candidates.sample(frac=1).ms_manager.pack()

    best_distance = None
    for subnet in subnet_candidates:
      subnet['arch'].pop('encoder_k')
      lm.model.manipulate_arch(subnet['arch'])
      self.trainer.predict(lm, ldm)
      distance = lm.distance.compute()
      
      self.logger.experiment.log(dict(subnet=subnet, distance=distance))
      if best_distance is None or distance < best_distance:
        self.logger.experiment.log(dict(best_subnet=subnet, best_distance=distance))
        best_distance = distance
  
  @staticmethod  
  def add_argparse_args(parent_parser):
    parser = parent_parser.add_argument_group("Searcher")
    parser.add_argument("--model_space_file", type=str)
    parser.add_argument("--min_gflops", type=float)
    parser.add_argument("--max_gflops", type=float)
    return parent_parser