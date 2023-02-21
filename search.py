from argparse import ArgumentParser
from inspect import signature
from logging import WARNING, INFO, basicConfig, getLogger, info
from statistics import mean
from optuna import create_study
from torch import set_float32_matmul_precision
from mvtec import MVTecDataModule
from patchcore import PatchCore
from pytorch_lightning import Trainer, seed_everything
from ofa.model_zoo import ofa_net
from optuna.samplers import NSGAIISampler


def objective(
    trial,
    trainer_kwargs,
    max_img_size,
    max_sampling_time=None,
    source_datamodules=None,
    target_datamodule=None,
):
    supernet = ofa_net(
        trial.suggest_categorical(
            "supernet_name",
            ["ofa_mbv3_d234_e346_k357_w1.0", "ofa_mbv3_d234_e346_k357_w1.2"],
        ),
        pretrained=True,
    )

    # Determine whether to extract from each stage in the supernet
    stage_extractions = [
        trial.suggest_categorical(f"stage_{i}", [True, False])
        for i in range(len(supernet.block_group_info))
    ]

    if not any(stage_extractions):
        raise RuntimeError("No stages selected for extraction.")

    # Find the index of the last stage to extract from
    last_stage_idx = (
        len(supernet.block_group_info) - stage_extractions[::-1].index(True) - 1
    )

    # Store the depth, kernel size, and expansion ratio for each stage
    stage_depths = []
    block_kernel_sizes = []
    block_expand_ratios = []
    patch_kernel_sizes = []
    extraction_blocks = []
    for stage_idx, stage_blocks in enumerate(
        supernet.block_group_info[: last_stage_idx + 1]
    ):
        # Determine the block kernel size and expansion ratio for the stage
        block_kernel_size = trial.suggest_int(
            f"stage_{stage_idx}_block_kernel_size", 3, 7, step=2
        )
        block_kernel_sizes.extend([block_kernel_size] * len(stage_blocks))
        expand_ratio = trial.suggest_categorical(
            f"stage_{stage_idx}_expand_ratio", [3, 4, 6]
        )
        block_expand_ratios.extend([expand_ratio] * len(stage_blocks))

        # Determine the depth of the stage
        stage_depth = (
            len(stage_blocks)
            if stage_idx == last_stage_idx
            else trial.suggest_int(
                f"stage_{stage_idx}_depth", 2, len(stage_blocks), step=1
            )
        )
        stage_depths.append(stage_depth)

        if stage_extractions[stage_idx]:
            # Determine the layer to extract from the stage
            extraction_block = trial.suggest_int(
                f"stage_{stage_idx}_block",
                stage_blocks[0],
                stage_blocks[stage_depth - 1],
                step=1,
            )
            extraction_blocks.append(extraction_block)

            # Determine the patch kernel size for the stage
            patch_kernel_sizes.append(
                trial.suggest_int(f"stage_{stage_idx}_patch_kernel_size", 1, 8, step=1)
            )

    patch_channels = supernet.blocks[extraction_blocks[-1]].conv.out_channels
    coreset_ratio = trial.suggest_float("coreset_ratio", 0.0, 1.0)
    patchcore_kwargs = dict(
        supernet=supernet,
        stage_depths=stage_depths,
        block_kernel_sizes=block_kernel_sizes,
        block_expand_ratios=block_expand_ratios,
        extraction_blocks=extraction_blocks,
        patch_kernel_sizes=patch_kernel_sizes,
        patch_channels=patch_channels,
        coreset_ratio=coreset_ratio,
        img_size=trial.suggest_int("img_size", 128, max_img_size, step=32),
        projection_channels=trial.suggest_int("projection_channels", 1, patch_channels),
        starting_points_ratio=trial.suggest_float(
            "starting_points_ratio", 0.0, coreset_ratio
        ),
    )

    trainer_kwargs |= dict(
        num_sanity_val_steps=0,
        logger=False,
        deterministic="warn",
        detect_anomaly=True,
        max_epochs=1,
    )

    if source_datamodules is not None:
        latencies = []
        avg_precisions = []
        for datamodule in source_datamodules:
            patchcore = PatchCore(
                **patchcore_kwargs, max_sampling_time=max_sampling_time
            )
            max_sampling_time = None
            latency, avg_precision = run(patchcore, trainer_kwargs, datamodule)
            latencies.append(latency)
            avg_precisions.append(avg_precision)
        trial.set_user_attr("source_latency_mean", mean(latencies))
        trial.set_user_attr("source_avg_precision_mean", mean(avg_precisions))

    if target_datamodule is not None:
        patchcore = PatchCore(**patchcore_kwargs)
        latency, avg_precision = run(patchcore, trainer_kwargs, target_datamodule)
        trial.set_user_attr("target_latency", latency)
        trial.set_user_attr("target_avg_precision", avg_precision)

        if source_datamodules is None:
            return patchcore

    return [
        trial.user_attrs["source_latency_mean"],
        trial.user_attrs["source_avg_precision_mean"],
    ]


def run(patchcore, trainer_kwargs, datamodule):
    trainer = Trainer(**trainer_kwargs)
    info(f"Fitting on {datamodule.class_}...")
    trainer.fit(patchcore, datamodule=datamodule)
    info(f"Testing on {datamodule.class_}...")
    trainer.test(patchcore, datamodule=datamodule)
    return patchcore.latency.compute().item(), patchcore.avg_precision.compute().item()


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
            args.test_batch_size,
        )
        for class_ in args.sources
    ]
    if args.target:
        target_datamodule = MVTecDataModule(
            args.dataset_dir,
            args.target,
            args.max_img_size,
            args.batch_size,
            args.test_batch_size,
        )

    study = create_study(
        study_name=args.study_name,
        load_if_exists=True,
        directions=["minimize", "maximize"],
        storage=args.db_url if args.study_name else None,
        sampler=NSGAIISampler(seed=None if args.study_name else args.seed),
    )
    study.optimize(
        lambda trial: objective(
            trial,
            trainer_kwargs,
            args.max_img_size,
            args.max_sampling_time,
            source_datamodules=source_datamodules,
            target_datamodule=target_datamodule if args.target else None,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        catch=RuntimeError,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--study_name", type=str)
    parser.add_argument("--n_trials", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=391)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--max_img_size", type=int, default=224)
    parser.add_argument("--max_sampling_time", type=int, default=60)
    parser.add_argument(
        "--dataset_dir", type=str, default="/dataB1/tommie_kerssies/MVTec"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "wood",
            "carpet",
            "leather",
            "grid",
        ],
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
