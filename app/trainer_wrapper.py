import inspect
from random import shuffle
from numpy import array, concatenate, count_nonzero
from pandas import Series, json_normalize, read_csv, merge
from pytorch_lightning import Trainer
from sklearn.metrics import jaccard_score
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from lib.patchcore.metrics import (
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from lib.patchcore.patchcore import PatchCore
from os.path import exists
from pathlib import Path


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
        masks = concatenate(
            [batch["mask"] for batch in ldm.setup().predict_dataloader()[2]]
        )
        masks_holdout = concatenate(
            [batch["mask"] for batch in ldm.predict_dataloader()[3]]
        )
        anomaly_labels = array([count_nonzero(x) > 0 for x in masks])
        anomaly_labels_holdout = array([count_nonzero(x) > 0 for x in masks_holdout])

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
            arch = subnet["arch"]

            if already_evaluated(arch):
                continue

            lm.model.manipulate_arch(arch)

            (
                subnet["flops"],
                subnet["params"],
                subnet["flops_hr"],
                subnet["params_hr"],
            ) = lm.get_backbone_complexity()

            outs_train, outs_train_holdout, outs_test, outs_test_holdout = self.predict(
                lm, ldm
            )

            subnet["distance"] = float(lm.distance.compute())

            pretrain_embed_dimension = (
                arch["encoder_q"]["body"]["width"][self.out_layers[-1]]
                * lm.block.expansion
            )
            patchcore = PatchCore(
                device=self.device_ids[0],
                n_layers=len(self.out_layers),
                input_shape=(3, self.img_size, self.img_size),
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=pretrain_embed_dimension,
            )

            print("Fitting PatchCore")
            patchcore.fit(outs_train)

            print("Predicting PatchCore train holdout")
            scores_train_holdout, _ = patchcore.predict(outs_train_holdout)
            min_score = min(scores_train_holdout)
            max_score = max(scores_train_holdout)

            print("Predicting PatchCore test")
            scores, segmentations = patchcore.predict(outs_test)

            print("Predicting PatchCore test holdout")
            scores_holdout, segmentations_holdout = patchcore.predict(outs_test_holdout)

            print("Computing metrics")
            subnet["test"] = {
                "thresholded_metrics": self._get_thresholded_metrics(
                    scores,
                    segmentations,
                    masks,
                    anomaly_labels,
                    min_score,
                    max_score,
                ),
                "imagewise_retrieval_metrics": compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                ),
                "pixelwise_retrieval_metrics": compute_pixelwise_retrieval_metrics(
                    segmentations, masks
                ),
            }

            subnet["test_holdout"] = {
                "thresholded_metrics": self._get_thresholded_metrics(
                    scores_holdout,
                    segmentations_holdout,
                    masks_holdout,
                    anomaly_labels_holdout,
                    min_score,
                    max_score,
                ),
                "imagewise_retrieval_metrics": compute_imagewise_retrieval_metrics(
                    scores_holdout, anomaly_labels_holdout
                ),
                "pixelwise_retrieval_metrics": compute_pixelwise_retrieval_metrics(
                    segmentations_holdout, masks_holdout
                ),
            }

            if already_evaluated(arch):
                continue

            json_normalize(subnet).to_csv(
                output_path,
                mode="a",
                header=not exists(output_path),
                index=False,
            )

    def _get_thresholded_metrics(
        self,
        scores,
        segmentations,
        masks,
        anomaly_labels,
        min_score,
        max_score,
        threshold=1.0,
    ):
        binary_preds = ((scores - min_score) / (max_score - min_score)) > threshold
        binary_segmentations = (
            (segmentations - min_score) / (max_score - min_score)
        ) > threshold

        IoU = []
        for i in range(len(binary_segmentations)):
            pred = binary_segmentations[i].flatten()
            gt = array(masks[i]).astype(bool).flatten()
            if gt.any() or pred.any():
                IoU.append(jaccard_score(gt, pred))

        tp = sum((binary_preds == True) & (anomaly_labels == True))
        tn = sum((binary_preds == False) & (anomaly_labels == False))
        fp = sum((binary_preds == True) & (anomaly_labels == False))
        fn = sum((binary_preds == False) & (anomaly_labels == True))
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        bal_acc = (tpr + tnr) / 2
        mIoU = sum(IoU) / len(IoU)

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp),
            "tpr_recall": tpr,
            "tnr": tnr,
            "bal_acc": bal_acc,
            "mIoU": mIoU,
            "performance": (bal_acc + mIoU) / 2,
        }
