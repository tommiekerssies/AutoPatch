from argparse import ArgumentParser
from inspect import signature
from logging import WARNING, INFO, basicConfig, getLogger, info
from optuna import create_study
from torch import set_float32_matmul_precision
from feature_extractor import FeatureExtractor
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
            args.batch_size,
            args.seed,
            args.val_ratio,
        )
        for class_ in args.sources
    ]
    if args.target:
        target_datamodule = MVTecDataModule(
            args.dataset_dir,
            args.target,
            args.max_img_size,
            args.batch_size,
            args.seed,
            args.val_ratio,
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

        # For each block in the supernet, suggest whether we should extract from it
        block_extractions = [
            trial.suggest_categorical(f"block_{block_idx}", [True, False])
            for block_idx in range(len(supernet.blocks))
        ]

        # Find the index of the last block we extract from
        try:
            last_block_idx = (
                len(block_extractions) - block_extractions[::-1].index(True) - 1
            )
        except ValueError:
            raise RuntimeError("No extraction blocks selected")

        # Find the index of the last stage we extract from
        for stage_idx, block_indices in enumerate(supernet.block_group_info[::-1]):
            if last_block_idx in block_indices:
                last_stage_idx = stage_idx
                break

        # For each stage before the last stage we extract from, we will set the depth (number of blocks)
        stage_depths = []
        for stage_idx, block_indices in enumerate(supernet.block_group_info):
            # If this is the last stage we extract from, we're done
            if stage_idx == last_stage_idx:
                break

            # Set the minimum and maximum depth
            stage_min_depth = 2
            stage_max_depth = len(block_indices)

            # Update the minimum depth if we are extracting if we are extracting a deeper block from this stage
            for i, block_idx in enumerate(block_indices):
                if block_extractions[block_idx]:
                    stage_min_depth = i + 1

            # If the minimum and maximum depth are the same, we don't need to suggest a depth
            if stage_min_depth == stage_max_depth:
                stage_depths.append(stage_min_depth)

            # Else we suggest a depth for this stage
            else:
                stage_depths.append(
                    trial.suggest_int(
                        f"stage_{stage_idx}_depth",
                        stage_min_depth,
                        stage_max_depth,
                        step=1,
                    )
                )
        supernet.set_active_subnet(d=stage_depths)

        kernel_sizes = []
        expand_ratios = []
        for block_idx in range(1, last_block_idx + 1):
            kernel_sizes.append(
                trial.suggest_int(f"block_{block_idx}_kernel_size", 3, 7, step=2)
            )
            expand_ratios.append(
                trial.suggest_categorical(
                    f"block_{block_idx}_expansion_ratio", [3, 4, 6]
                )
            )
        supernet.set_active_subnet(ks=kernel_sizes, e=expand_ratios)

        backbone = FeatureExtractor(
            supernet,
            [f"blocks.{i}" for i, extract in enumerate(block_extractions) if extract],
        )

        k_nn = trial.suggest_int("k_nn", 1, 8, step=1)
        patch_stride = trial.suggest_int("patch_stride", 1, 8, step=1)
        patch_kernel_size = trial.suggest_int("patch_kernel_size", 1, 16, step=1)
        patch_channels = trial.suggest_int("patch_channels", 8, 1280, step=8)
        img_size = trial.suggest_int("img_size", 128, args.max_img_size, step=32)

        sources_patchcore = PatchCore(
            backbone, k_nn, patch_stride, patch_kernel_size, patch_channels, img_size
        )
        for datamodule in source_datamodules:
            source_trainer = Trainer(**trainer_kwargs)
            info(f"Fitting on source {datamodule.class_}...")
            source_trainer.fit(sources_patchcore, datamodule=datamodule)
            info(f"Testing on source {datamodule.class_}...")
            source_trainer.test(sources_patchcore, datamodule=datamodule)
        sources_latency = sources_patchcore.latency.compute().item()
        sources_average_precision = sources_patchcore.average_precision.compute().item()
        del sources_patchcore

        if args.target:
            target_patchcore = PatchCore(
                backbone,
                k_nn,
                patch_stride,
                patch_kernel_size,
                patch_channels,
                img_size,
            )
            target_trainer = Trainer(**trainer_kwargs)
            info(f"Fitting on target {args.target}...")
            target_trainer.fit(target_patchcore, datamodule=target_datamodule)
            info(f"Testing on target {args.target}...")
            target_trainer.test(target_patchcore, datamodule=target_datamodule)
            trial.set_user_attr(
                "target_latency", target_patchcore.latency.compute().item()
            )
            trial.set_user_attr(
                "target_average_precision",
                target_patchcore.average_precision.compute().item(),
            )

        return [sources_latency, sources_average_precision]

    study = create_study(
        study_name=args.study_name
        or datetime.now().strftime(f"{args.target}_%Y-%m-%d_%H:%M:%S"),
        load_if_exists=True,
        directions=["minimize", "maximize"],
        storage=args.db_url if args.log or args.study_name else None,
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
    parser.add_argument("--batch_size", type=int, default=391)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_img_size", type=int, default=256)
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
