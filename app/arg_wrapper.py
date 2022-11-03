from argparse import ArgumentParser


class ArgWrapper(ArgumentParser):
    def __init__(self, *classes):
        super().__init__()
        self._add_global_args()
        for cls in classes:
            cls.add_argparse_args(self)

    def _add_global_args(self):
        self.add_argument("--seed", type=int, default=0)
        self.add_argument("--work_dir", type=str, default="/dataB1/tommie_kerssies")

    def parse_kwargs(self) -> dict:
        return vars(self.parse_args())
