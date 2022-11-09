from argparse import Namespace
import inspect
from os import environ
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from app.callback.scheduled_stop import ScheduledStopCallback
from app.lightning_module.supernet import SuperNet
from lib.gaia.base_rule import SAMPLE_RULES
import lib.gaia.eval_rule as eval_rule
from mmcv.utils import build_from_cfg
from lib.gaia.model_space_manager import ModelSpaceManager
from os.path import join
from pytorch_lightning.loggers.wandb import WandbLogger


class TrainerWrapper(Trainer):
    @staticmethod
    def add_argparse_args(parser):
        Trainer.add_argparse_args(parser)
        parser.add_argument("--resume_run_id", type=str)
        parser.add_argument("--stop_time", type=str)
        parser.add_argument("--patience", type=int)
        parser.add_argument("--project_name", type=str, default="fine-tune_aoi")
        parser.add_argument("--max_gflops", type=float)
        parser.add_argument("--supernet_run_id", type=str)
        parser.add_argument("--model_space_file", type=str)

    def __init__(
        self,
        seed,
        resume_run_id,
        supernet_run_id,
        project_name,
        work_dir,
        patience,
        stop_time,
        max_epochs,
        monitor,
        monitor_mode,
        model_space_file,
        max_gflops,
        **kwargs,
    ):
        self.resume_run_id = resume_run_id
        self.supernet_run_id = supernet_run_id
        self.work_dir = work_dir
        self.model_space_file = model_space_file
        self.max_gflops = max_gflops

        callbacks = [
            ModelCheckpoint(
                monitor=monitor,
                mode=monitor_mode,
                save_last=True,
                filename=f"{{epoch}}-{{{monitor}}}",
                verbose=True,
            ),
            ScheduledStopCallback(stop_time),
        ]

        if patience:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode=monitor_mode,
                    patience=patience,
                    verbose=True,
                )
            )

        # we only want to pass in valid Trainer args,
        # the rest may be user specific
        valid_kwargs = inspect.signature(Trainer.__init__).parameters
        self.trainer_kwargs = {
            name: kwargs[name] for name in valid_kwargs if name in kwargs
        }

        logger = None
        if project_name:
            logger = WandbLogger(
                id=resume_run_id,
                project=project_name,
                save_dir=work_dir,
                resume="must" if resume_run_id else "never",
                settings=wandb.Settings(code_dir="."),
            )

        self.trainer_kwargs |= dict(
            precision=16,
            strategy="ddp_find_unused_parameters_false",
            accelerator="auto",
            log_every_n_steps=1,
            deterministic="warn",
            callbacks=callbacks,
            max_epochs=max_epochs or -1,
            logger=logger,
        )

        super().__init__(**self.trainer_kwargs)

    def fit(self, lm, ldm):
        if self.supernet_run_id and not self.resume_run_id:
            self.load_weights_from_supernet(lm, ldm)

        if self.logger:
            self.logger.watch(lm)  # type: ignore

        super().fit(lm, datamodule=ldm, ckpt_path=lm.ckpt_path)

    def search(self, ldm):
        # TODO: move this search code closer to supernet, or to a seperate class.
        # TODO: allow resuming supernet searching.
        if self.supernet_run_id is None:
            raise ValueError("supernet_run_id must be provided for searching.")

        # create the supernet we want to extract a subnet from
        supernet_lm = SuperNet.create(
            resume_run_id=self.supernet_run_id, work_dir=self.work_dir
        )

        model_space = ModelSpaceManager.load(join(self.work_dir, self.model_space_file))
        model_sampling_rule = build_from_cfg(
            dict(
                type="eval",
                func_str=f"lambda x: x['overhead.flops'] < {self.max_gflops * 1e9}",
            ),
            SAMPLE_RULES,
        )

        subnet_candidates = model_space.ms_manager.apply_rule(model_sampling_rule)
        subnet_candidates = subnet_candidates.sample(frac=1).ms_manager.pack()

        best_distance = None
        for subnet in subnet_candidates:
            if "encoder_k" in subnet["arch"]:
                subnet["arch"].pop("encoder_k")
            supernet_lm.model.manipulate_arch(subnet["arch"])
            self.predict(supernet_lm, ldm)
            distance = supernet_lm.distance.compute()

            self.logger.experiment.log(dict(subnet=subnet, distance=distance))  # type: ignore
            if best_distance is None or distance < best_distance:
                self.logger.experiment.log(  # type: ignore
                    dict(best_subnet=subnet, best_distance=distance)
                )
                best_distance = distance

    def tune(self, lm, ldm, lr, batch_size, **kwargs):
        auto_lr_find = not lr
        auto_scale_batch_size = None if batch_size else "binsearch"

        # Set lr and bs to default values to prevent errors
        if auto_lr_find:
            lm.hparams.lr = 1e-3
        if auto_scale_batch_size:
            ldm.hparams.batch_size = 2

        # Don't perform lr warm-up
        lm.hparams.warmup_steps = 0

        tune_trainer = Trainer.from_argparse_args(
            Namespace(**self.trainer_kwargs),
            strategy=None,
            devices=1,
            num_nodes=1,
            auto_lr_find=auto_lr_find,
            auto_scale_batch_size=auto_scale_batch_size,
        )

        tune_trainer.tune(lm, datamodule=ldm)

    def load_weights_from_supernet(self, lm, ldm):
        # to prevent error when starting training later set this env
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # create the supernet we want to extract a subnet from
        supernet_lm = SuperNet.create(
            resume_run_id=self.supernet_run_id, work_dir=self.work_dir
        )

        # Doing a forward pass with supernet in deploy mode should
        # remove unused layers and channels from its state dict,
        # resulting in only the subnet weights.
        supernet_lm.model.deploy()
        supernet_lm.model.manipulate_arch(
            dict(
                encoder_q=dict(
                    stem=dict(width=lm.model_cfg["backbone"]["stem_width"]),
                    body=dict(
                        width=lm.model_cfg["backbone"]["body_width"],
                        depth=lm.model_cfg["backbone"]["body_depth"],
                    ),
                )
            )
        )
        Trainer(strategy="ddp", accelerator="auto", fast_dev_run=True).predict(
            supernet_lm, ldm
        )

        return lm.load_state_dict_verbose(supernet_lm.state_dict())
