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
from feature_extractor import FeatureExtractor
from optuna.samplers import TPESampler
from deepspeed.profiling.flops_profiler import get_model_profile


def objective(
    trial,
    datamodule,
    trainer_kwargs,
    max_img_size,
    search_on_test_set=False,
    return_patchcore=False,
):
    supernet = ofa_net(
        trial.suggest_categorical(
            "supernet_name",
            ["ofa_mbv3_d234_e346_k357_w1.0", "ofa_mbv3_d234_e346_k357_w1.2"],
        ),
        pretrained=True,
    )

    stage_depths = {}
    stage_block = {}
    stage_kernel_size = {}
    stage_expand_ratio = {}
    stage_patch_size = {}

    for stage_idx, stage_blocks in enumerate(supernet.block_group_info):
        stage_kernel_size[stage_idx] = trial.suggest_int(
            f"stage_{stage_idx}_kernel_size", 3, 7, step=2
        )
        stage_expand_ratio[stage_idx] = trial.suggest_categorical(
            f"stage_{stage_idx}_expand_ratio", [3, 4, 6]
        )
        stage_patch_size[stage_idx] = trial.suggest_int(
            f"stage_{stage_idx}_patch_size", 1, 16, step=1
        )
        stage_block[stage_idx] = trial.suggest_categorical(
            f"stage_{stage_idx}_block", [None, *stage_blocks]
        )

        stage_depths[stage_idx] = 2
        if stage_idx in stage_block and stage_block[stage_idx] is not None:
            stage_depths[stage_idx] = max(
                stage_depths[stage_idx], stage_blocks.index(stage_block[stage_idx]) + 1
            )

    ks = []
    e = []
    for stage_idx, kernel_size in stage_kernel_size.items():
        ks.extend(kernel_size for _ in range(len(supernet.block_group_info[stage_idx])))
    for stage_idx, expand_ratio in stage_expand_ratio.items():
        e.extend(expand_ratio for _ in range(len(supernet.block_group_info[stage_idx])))
    supernet.set_active_subnet(ks, e, list(stage_depths.values()))

    extraction_blocks = [block for block in stage_block.values() if block is not None]
    if not extraction_blocks:
        raise RuntimeError("No blocks selected for extraction.")

    feature_extractor = FeatureExtractor(
        supernet, [f"blocks.{block}" for block in extraction_blocks]
    )
    img_size = trial.suggest_int("img_size", 128, max_img_size, step=32)
    flops, _, _ = get_model_profile(
        feature_extractor,
        (1, 3, img_size, img_size),
        print_profile=False,
        as_string=False,
    )

    patch_sizes = [
        patch_size
        for stage_idx, patch_size in stage_patch_size.items()
        if stage_idx in stage_block and stage_block[stage_idx] is not None
    ]
    patch_channels = supernet.blocks[extraction_blocks[-1]].conv.out_channels

    trainer_kwargs |= dict(
        num_sanity_val_steps=0,
        logger=False,
        deterministic="warn",
        detect_anomaly=True,
        max_epochs=1,
    )

    patchcore = PatchCore(
        feature_extractor,
        img_size,
        patch_sizes,
        patch_channels,
    )

    trainer = Trainer(**trainer_kwargs)
    info("Fitting...")
    trainer.fit(patchcore, datamodule=datamodule)
    trial.set_user_attr("val_rwAP", patchcore.rwAP)
    trial.set_user_attr("val_optimal_threshold", patchcore.threshold)
    trial.set_user_attr("val_optimal_rwF1", patchcore.optimal_rwF1)

    info("Testing...")
    trainer.test(patchcore, datamodule=datamodule)
    trial.set_user_attr("test_rwAP", patchcore.rwAP)
    trial.set_user_attr("test_optimal_threshold", patchcore.threshold)
    trial.set_user_attr("test_optimal_rwF1", patchcore.optimal_rwF1)
    trial.set_user_attr("test_rwF1", patchcore.rwF1)

    if return_patchcore:
        return patchcore

    if search_on_test_set:
        return [flops, trial.user_attrs["test_rwAP"]]

    return [
        flops,
        trial.user_attrs["val_rwAP"],
    ]


def main(args, trainer_kwargs):
    seed_everything(args.seed, workers=True)
    set_float32_matmul_precision("medium")
    getLogger("pytorch_lightning").setLevel(WARNING)
    basicConfig(level=INFO)

    study = create_study(
        study_name=args.study_name,
        load_if_exists=True,
        directions=["minimize", "maximize"],
        storage=args.db_url if args.study_name else None,
        sampler=TPESampler(
            seed=None if args.study_name else args.seed,
            multivariate=True,
            constant_liar=True,
        ),
    )

    datamodule = MVTecDataModule(
        args.dataset_dir,
        args.category,
        args.max_img_size,
        args.batch_size,
        args.k,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            datamodule,
            trainer_kwargs,
            args.max_img_size,
            args.search_on_test_set,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        catch=RuntimeError,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--study_name", type=str)
    parser.add_argument("--n_trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--search_on_test_set", action="store_true")
    parser.add_argument("--k", type=int)
    parser.add_argument("--batch_size", type=int, default=391)
    parser.add_argument("--max_img_size", type=int, default=224)
    parser.add_argument("--category", type=str)
    parser.add_argument(
        "--dataset_dir", type=str, default="/dataB1/tommie_kerssies/MVTec"
    )
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
