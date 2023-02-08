from argparse import ArgumentParser
from inspect import signature
from logging import WARNING, INFO, basicConfig, getLogger, info
from statistics import mean
from optuna import create_study
from torch import set_float32_matmul_precision, sum
from feature_extractor import FeatureExtractor
from metrics import lookup_precision_recall
from mvtec import MVTecDataModule
from patchcore import PatchCore
from pytorch_lightning import Trainer, seed_everything
from ofa.model_zoo import ofa_net
from optuna.samplers import NSGAIISampler
from datetime import datetime


def main(args, trainer_kwargs):
    seed_everything(args.seed, workers=True)
    set_float32_matmul_precision("medium")
    getLogger("pytorch_lightning").setLevel(WARNING)
    basicConfig(level=INFO)

    source_datamodules = [
        MVTecDataModule(
            args.dataset_dir,
            class_,
            args.max_img_size,
            args.train_batch_size,
            args.test_batch_size,
        )
        for class_ in args.sources
    ]
    target_datamodule = MVTecDataModule(
        args.dataset_dir,
        args.target,
        args.max_img_size,
        args.train_batch_size,
        args.test_batch_size,
    )

    trainer_kwargs |= dict(
        num_sanity_val_steps=0,
        logger=False,
        deterministic="warn",
        detect_anomaly=True,
        max_epochs=1,
    )

    def objective(trial):
        supernet_name = trial.suggest_categorical(
            "supernet_name",
            [
                "ofa_proxyless_d234_e346_k357_w1.3",
                "ofa_mbv3_d234_e346_k357_w1.0",
                "ofa_mbv3_d234_e346_k357_w1.2",
            ],
        )
        supernet = ofa_net(supernet_name, pretrained=True)
        supernet.set_active_subnet(
            ks=trial.suggest_int("subnet_kernel_size", 3, 7, step=2),
            e=trial.suggest_categorical("subnet_expansion_ratio", [3, 4, 6]),
            d=trial.suggest_int("subnet_depth", 2, 4, step=1),
        )

        extraction_layers = []
        for i_stage, block_indices in enumerate(supernet.block_group_info):
            extract_from_stage = trial.suggest_categorical(
                f"{supernet_name}_stage_{i_stage}", [True, False]
            )
            if extract_from_stage:
                depth = supernet.runtime_depth[i_stage]
                i_extract_block = trial.suggest_int(
                    f"{supernet_name}_stage_{i_stage}_block",
                    block_indices[0],
                    block_indices[depth - 1],
                    step=1,
                )
                extraction_layers.append(f"blocks.{i_extract_block}")

        if not extraction_layers:
            raise RuntimeError("No extraction layers selected")

        backbone = FeatureExtractor(supernet, extraction_layers)
        k_nn = trial.suggest_int("k_nn", 1, 4, step=1)
        patch_stride = trial.suggest_int("patch_stride", 1, 4, step=1)
        patch_kernel_size = trial.suggest_int("patch_kernel_size", 1, 8, step=1)
        patch_channels = trial.suggest_int("patch_channels", 8, 640, step=8)
        img_size = trial.suggest_int("img_size", 128, args.max_img_size, step=32)
        threshold = trial.suggest_float("threshold", 0.0, 1.0)

        min_pred_list = []
        max_pred_list = []
        pc_seg_list = []
        rc_seg_list = []
        pc_clf_list = []
        rc_clf_list = []
        thresholds_seg_list = []
        thresholds_clf_list = []
        latency_list = []
        for datamodule in source_datamodules:
            info(f"Fitting on source {datamodule.class_}...")
            patchcore = PatchCore(
                backbone,
                k_nn,
                patch_stride,
                patch_kernel_size,
                patch_channels,
                img_size,
            )
            Trainer(**trainer_kwargs).fit(patchcore, datamodule=datamodule)

            min_pred_list.append(patchcore.min_pred.compute().item())
            max_pred_list.append(patchcore.max_pred.compute().item())
            seg_pc, seg_rc, seg_thresholds = patchcore.seg_pr_curve.compute()
            clf_pc, clf_rc, clf_thresholds = patchcore.clf_pr_curve.compute()
            pc_seg_list.append(seg_pc)
            rc_seg_list.append(seg_rc)
            pc_clf_list.append(clf_pc)
            rc_clf_list.append(clf_rc)
            thresholds_seg_list.append(seg_thresholds)
            thresholds_clf_list.append(clf_thresholds)
            latency_list.append(patchcore.latency.compute().item())

        # Scale the thresholds, as we didn't scale the predictions in the loop (because we didn't know the min and max yet).
        min_pred = min(min_pred_list)
        max_pred = max(max_pred_list)
        for thresholds in thresholds_seg_list + thresholds_clf_list:
            thresholds -= min_pred
            thresholds /= max_pred - min_pred

        -sum((recall[1:] - recall[:-1]) * precision[:-1])

        pr_seg_tuples = [
            lookup_precision_recall(seg_pc, seg_rc, seg_thresholds, threshold)
            for seg_pc, seg_rc, seg_thresholds in zip(
                pc_seg_list, rc_seg_list, thresholds_seg_list
            )
        ]
        pr_clf_tuples = [
            lookup_precision_recall(clf_pc, clf_rc, clf_thresholds, threshold)
            for clf_pc, clf_rc, clf_thresholds in zip(
                pc_clf_list, rc_clf_list, thresholds_clf_list
            )
        ]

        info(f"Evaluating on target {args.target}...")
        target_patchcore = PatchCore(
            backbone,
            k_nn,
            patch_stride,
            patch_kernel_size,
            patch_channels,
            img_size,
            min_pred,
            max_pred,
        )
        Trainer(**trainer_kwargs).fit(target_patchcore, datamodule=target_datamodule)

        target_seg_precision, target_seg_recall = lookup_precision_recall(
            *target_patchcore.seg_pr_curve.compute(), threshold
        )
        target_clf_precision, target_clf_recall = lookup_precision_recall(
            *target_patchcore.clf_pr_curve.compute(), threshold
        )
        trial.set_user_attr("target_seg_precision", target_seg_precision)
        trial.set_user_attr("target_seg_recall", target_seg_recall)
        trial.set_user_attr("target_clf_precision", target_clf_precision)
        trial.set_user_attr("target_clf_recall", target_clf_recall)
        trial.set_user_attr("target_latency", target_patchcore.latency.compute().item())

        return (
            mean([precision_seg for precision_seg, _ in pr_seg_tuples]),
            mean([recall_seg for _, recall_seg in pr_seg_tuples]),
            mean([precision_clf for precision_clf, _ in pr_clf_tuples]),
            mean([recall_clf for _, recall_clf in pr_clf_tuples]),
            mean(latency_list),
        )

    study = create_study(
        study_name=args.study_name
        or datetime.now().strftime(f"{args.target}_%Y-%m-%d_%H:%M:%S"),
        load_if_exists=True,
        directions=["maximize", "maximize", "maximize", "maximize", "minimize"],
        storage=args.db_url if args.log else None,
        sampler=NSGAIISampler(seed=args.seed),
    )
    study.optimize(
        objective, n_trials=args.n_trials, n_jobs=args.n_jobs, catch=RuntimeError
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--study_name", type=str)
    parser.add_argument("--n_trials", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=391)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--max_img_size", type=int, default=320)
    parser.add_argument("--log", action="store_true")
    parser.add_argument(
        "--dataset_dir", type=str, default="/dataB1/tommie_kerssies/MVTec"
    )
    parser.add_argument(
        "--sources", nargs="+", default=["carpet", "grid", "leather", "wood"]
    )
    parser.add_argument("--target", type=str, default="tile")
    parser.add_argument(
        "--db_url",
        type=str,
        default="postgresql://tommie_kerssies:tommie_kerssies@10.78.50.251",
    )

    Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    trainer_kwargs = {
        name: vars(args)[name]
        for name in signature(Trainer.__init__).parameters
        if name in vars(args)
    }

    main(parser.parse_args(), trainer_kwargs)
