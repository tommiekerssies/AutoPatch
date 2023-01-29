import inspect
from random import shuffle
from numpy import argmax, divide, zeros_like
from pandas import json_normalize, read_csv, merge
from pytorch_lightning import Trainer
from sklearn.metrics import f1_score, precision_recall_curve
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from lib.patchcore.patchcore import PatchCore
from os.path import exists
from pathlib import Path
from torch import cat, flatten, max, min, squeeze


class TrainerWrapper(Trainer):
    @staticmethod
    def add_argparse_args(parser):
        Trainer.add_argparse_args(parser)
        parser.add_argument("--resume_run_id", type=str)
        parser.add_argument("--supernet_run_id", type=str)
        parser.add_argument("--project_name", type=str)
        parser.add_argument("--run_name", type=str)

    def __init__(
        self,
        resume_run_id,
        supernet_run_id,
        work_dir,
        project_name,
        run_name,
        out_layers,
        img_size,
        num_workers,
        category,
        **kwargs,
    ):
        self.resume_run_id = resume_run_id
        self.supernet_run_id = supernet_run_id
        self.work_dir = work_dir
        self.project_name = project_name
        self.out_layers = out_layers
        self.img_size = img_size
        self.num_workers = num_workers
        self.category = category

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
                name=run_name,
                save_dir=work_dir,
                resume="must" if resume_run_id else "never",
                settings=wandb.Settings(code_dir="."),
            )

        self.trainer_kwargs |= dict(
            strategy="ddp_find_unused_parameters_false",
            log_every_n_steps=1,
            deterministic=True,
            logger=logger,
            detect_anomaly=True,
            max_epochs=-1,
        )

        super().__init__(**self.trainer_kwargs)

    def search(self, lm, ldm):
        pixel_labels_val = squeeze(
            flatten(
                cat([batch["mask"] for batch in ldm.setup().predict_dataloader()[1]]),
                start_dim=-2,
            )
        )
        pixel_labels_test = squeeze(
            flatten(
                cat([batch["mask"] for batch in ldm.setup().predict_dataloader()[2]]),
                start_dim=-2,
            )
        )

        output_path = Path(self.work_dir, f"{self.category}.csv")

        def already_evaluated(arch):
            if exists(output_path):
                arch_df = json_normalize({"arch": arch}).astype(str)
                output_df = read_csv(output_path).astype(str)
                matches = merge(arch_df, output_df, on=arch_df.columns.tolist())

                if len(matches) > 1:
                    raise ValueError(f"Multiple results found for architecture: {arch}")
                elif len(matches) > 0:
                    print(f"Already evaluated architecture: {arch}")
                    return True

            return False

        subnets = lm.get_subnets()
        shuffle(subnets)

        for subnet in subnets:
            if already_evaluated(subnet["arch"]):
                continue

            lm.model.manipulate_arch(subnet["arch"])

            (
                subnet["flops"],
                subnet["params"],
                subnet["flops_hr"],
                subnet["params_hr"],
            ) = lm.get_backbone_complexity()

            outs_train, outs_val, outs_test = self.predict(lm, ldm)

            pretrain_embed_dimension = (
                subnet["arch"]["encoder_q"]["body"]["width"][self.out_layers[-1]]
                * lm.block.expansion
            )
            patchcore = PatchCore(
                device=self.device_ids[0],
                n_layers=len(self.out_layers),
                input_shape=(3, self.img_size, self.img_size),
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=pretrain_embed_dimension,
            )

            print("Fitting PatchCore...")
            patchcore.fit(outs_train)

            print("Predicting PatchCore...")
            unscaled_pixel_scores_val = flatten(
                patchcore.predict(outs_val), start_dim=-2
            )
            unscaled_pixel_scores_test = flatten(
                patchcore.predict(outs_test), start_dim=-2
            )

            print("Computing metrics...")
            score_max = max(unscaled_pixel_scores_val)
            score_min = min(unscaled_pixel_scores_val)

            pixel_scores_val = (unscaled_pixel_scores_val - score_min) / (
                score_max - score_min
            )
            pixel_scores_test = (unscaled_pixel_scores_test - score_min) / (
                score_max - score_min
            )

            precision_curve, recall_curve, thresholds = precision_recall_curve(
                flatten(pixel_labels_val).int(), flatten(pixel_scores_val).cpu()
            )
            f1_scores = divide(
                2 * precision_curve * recall_curve,
                precision_curve + recall_curve,
                out=zeros_like(precision_curve),
                where=(precision_curve + recall_curve) != 0,
            )
            i_f1_max = argmax(f1_scores)

            subnet["pixel_f1_val"] = f1_score(
                flatten(pixel_labels_val).bool(),
                flatten(pixel_scores_val.cpu() > thresholds[i_f1_max]),
            )
            subnet["pixel_f1_test"] = f1_score(
                flatten(pixel_labels_test).bool(),
                flatten(pixel_scores_test.cpu() > thresholds[i_f1_max]),
            )

            subnet["img_f1_val"] = f1_score(
                max(pixel_labels_val, dim=-1).bool(),
                max(pixel_scores_val, dim=-1).cpu() > thresholds[i_f1_max],
            )
            subnet["img_f1_test"] = f1_score(
                max(pixel_labels_test, dim=-1).bool(),
                max(pixel_scores_test, dim=-1).cpu() > thresholds[i_f1_max],
            )

            if already_evaluated(subnet["arch"]):
                continue

            json_normalize(subnet).to_csv(
                output_path,
                mode="a",
                header=not exists(output_path),
                index=False,
            )
