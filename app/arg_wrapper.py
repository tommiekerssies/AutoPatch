from argparse import ArgumentParser

class ArgWrapper(ArgumentParser):
  def __init__(self, *classes):
    super().__init__()
    self._add_global_args()
    for cls in classes:
      cls.add_argparse_args(self)
  
  def _add_global_args(self):
    parser = self.add_argument_group("Global")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume_run_id", type=str)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--weights_file", type=str)
    parser.add_argument("--prefix_old", type=str)
    parser.add_argument("--prefix_new", type=str)
    parser.add_argument("--project_name", type=str)

  def parse_kwargs(self) -> dict:
    return vars(self.parse_args())