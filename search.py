from argparse import ArgumentParser
from inspect import signature
from optuna import create_study
from feature_extractor import FeatureExtractor
from mvtec import MVTecDataModule
from patchcore import PatchCore
from pytorch_lightning import Trainer, seed_everything
from ofa.model_zoo import ofa_net
from optuna.samplers import NSGAIISampler
from datetime import datetime


def main(args, trainer_kwargs):
    seed_everything(args.seed, workers=True)

    data_module = MVTecDataModule(
        work_dir=args.work_dir,
        dataset_dir=args.dataset_dir,
        max_img_size=args.max_img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )

    trainer_kwargs |= dict(
        num_sanity_val_steps=0,
        logger=False,
        deterministic="warn",
        detect_anomaly=True,
        max_epochs=args.epochs,
    )

    def objective(trial):
        trainer = Trainer(**trainer_kwargs)

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

        patchcore = PatchCore(
            backbone=FeatureExtractor(supernet, extraction_layers),
            k_nn=trial.suggest_int("k_nn", 1, 4, step=1),
            patch_stride=trial.suggest_int("patch_stride", 1, 4, step=1),
            patch_kernel_size=trial.suggest_int("patch_kernel_size", 1, 6, step=1),
            patch_channels=trial.suggest_int("patch_channels", 8, 256, step=8),
            img_size=trial.suggest_int("img_size", 128, args.max_img_size, step=32),
        )

        trainer.fit(patchcore, data_module)
        val_latency = patchcore.latency.compute().item()
        val_seg_f1 = patchcore.seg_f1.compute().item()
        val_clf_f1 = patchcore.clf_f1.compute().item()
        seg_threshold = patchcore.seg_f1.threshold.item()
        clf_threshold = patchcore.clf_f1.threshold.item()

        trainer.test(patchcore, data_module)
        test_latency = patchcore.latency.compute().item()
        test_seg_f1 = patchcore.seg_f1.compute().item()
        test_clf_f1 = patchcore.clf_f1.compute().item()

        trial.set_user_attr("seg_threshold", seg_threshold)
        trial.set_user_attr("clf_threshold", clf_threshold)
        trial.set_user_attr("test_latency", test_latency)
        trial.set_user_attr("test_seg_f1", test_seg_f1)
        trial.set_user_attr("test_clf_f1", test_clf_f1)

        return val_latency, val_seg_f1, val_clf_f1

    study = create_study(
        study_name=args.study_name or datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        load_if_exists=True,
        directions=["minimize", "maximize", "maximize"],
        storage=args.db_url,
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=267)
    parser.add_argument("--val_ratio", type=float, default=1.0)
    parser.add_argument("--max_img_size", type=int, default=320)
    parser.add_argument("--work_dir", type=str, default="/dataB1/tommie_kerssies")
    parser.add_argument("--dataset_dir", type=str, default="MVTec/pill")
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
