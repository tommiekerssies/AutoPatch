from argparse import ArgumentParser
from inspect import signature
from logging import WARNING, INFO, basicConfig, getLogger, info
from optuna import create_study
from torch import set_float32_matmul_precision
from mvtec import MVTecDataModule
from model import Model
from pytorch_lightning import Trainer, seed_everything
from ofa.model_zoo import ofa_net
from feature_extractor import FeatureExtractor
from optuna.samplers import TPESampler
from deepspeed.profiling.flops_profiler import get_model_profile


def objective(
    trial,
    datamodule,
    trainer_kwargs,
    img_size,
    fixed_supernet_name=None,
    fixed_kernel_size=None,
    fixed_expand_ratio=None,
    test_set_search=False,
    return_model=False,
):
    supernet = ofa_net(
        trial.suggest_categorical(
            "supernet_name",
            ["ofa_mbv3_d234_e346_k357_w1.0", "ofa_mbv3_d234_e346_k357_w1.2"],
        )
        if fixed_supernet_name is None
        else fixed_supernet_name,
        pretrained=True,
    )

    stage_depths = {}
    stage_block = {}
    stage_kernel_size = {}
    stage_expand_ratio = {}
    stage_patch_size = {}

    for stage_idx, stage_blocks in enumerate(supernet.block_group_info):
        stage_kernel_size[stage_idx] = (
            trial.suggest_int(f"stage_{stage_idx}_kernel_size", 3, 7, step=2)
            if fixed_kernel_size is None
            else fixed_kernel_size
        )
        stage_expand_ratio[stage_idx] = (
            trial.suggest_categorical(f"stage_{stage_idx}_expand_ratio", [3, 4, 6])
            if fixed_expand_ratio is None
            else fixed_expand_ratio
        )
        stage_block[stage_idx] = trial.suggest_categorical(
            f"stage_{stage_idx}_block", [None, *stage_blocks]
        )
        stage_patch_size[stage_idx] = trial.suggest_int(
            f"stage_{stage_idx}_patch_size", 1, 16, step=1
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

    flops, _, _ = get_model_profile(
        feature_extractor,
        (1, 3, img_size, img_size),
        print_profile=False,
        as_string=False,
    )

    trainer_kwargs.update(
        dict(
            num_sanity_val_steps=0,
            logger=False,
            deterministic="warn",
            detect_anomaly=True,
            max_epochs=1,
            limit_val_batches=0 if test_set_search else None,
        )
    )
    trainer = Trainer(**trainer_kwargs)

    model = Model(
        feature_extractor,
        img_size,
        patch_sizes=[
            patch_size
            for stage_idx, patch_size in stage_patch_size.items()
            if stage_idx in stage_block and stage_block[stage_idx] is not None
        ],
        patch_channels=sum(
            supernet.blocks[block].conv.out_channels for block in extraction_blocks
        ),
    )

    info("Fitting...")
    trainer.fit(model, datamodule=datamodule)
    if not test_set_search:
        trial.set_user_attr("val_AUROC", model.AUROC)
        trial.set_user_attr("val_partial_AUROC", model.partial_AUROC)
        trial.set_user_attr("val_AP", model.AP)
        trial.set_user_attr("val_wAP", model.wAP)

    info("Testing...")
    trainer.test(model, datamodule=datamodule)
    trial.set_user_attr("test_AUROC", model.AUROC)
    trial.set_user_attr("test_partial_AUROC", model.partial_AUROC)
    trial.set_user_attr("test_AP", model.AP)
    trial.set_user_attr("test_wAP", model.wAP)

    if return_model:
        return model

    if test_set_search:
        return [flops, trial.user_attrs["test_wAP"]]
    else:
        return [flops, trial.user_attrs["val_wAP"]]


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
            seed=args.seed,
            multivariate=True,
            constant_liar=True,
        ),
    )

    datamodule = MVTecDataModule(
        args.dataset_dir,
        args.category,
        args.img_size,
        args.batch_size,
        args.k,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            datamodule,
            trainer_kwargs,
            args.img_size,
            args.fixed_supernet_name,
            args.fixed_kernel_size,
            args.fixed_expand_ratio,
            args.test_set_search,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        catch=RuntimeError,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--study_name", type=str)
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument(
        "--test_set_search", default=False, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument("--k", type=int)
    parser.add_argument("--batch_size", type=int, default=391)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--category", type=str)
    parser.add_argument("--fixed_supernet_name", type=str)
    parser.add_argument("--fixed_kernel_size", type=int)
    parser.add_argument("--fixed_expand_ratio", type=int)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--db_url", type=str)

    Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    trainer_kwargs = {
        name: vars(args)[name]
        for name in signature(Trainer.__init__).parameters
        if name in vars(args)
    }

    main(parser.parse_args(), trainer_kwargs)
