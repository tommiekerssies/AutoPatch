from argparse import ArgumentParser
from optuna import create_study
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Trainer
from mvtec import MVTecDataModule
from patchcore import PatchCore


def main(args):
    seed_everything(args.seed, workers=True)

    trainer = Trainer(
        deterministic=True,
        detect_anomaly=True,
        accelerator=args.accelerator,
        max_epochs=1,
    )

    patchcore = PatchCore(
        img_size=args.img_size,
        k_nn=args.k_nn,
        patch_channels=args.patch_channels,
        patch_kernel_size=args.patch_kernel_size,
        patch_stride=args.patch_stride,
        extraction_layers=args.extraction_layers,
        supernet_name=args.supernet_name,
        subnet_kernel_size=args.subnet_kernel_size,
        subnet_expansion_ratio=args.subnet_expansion_ratio,
        subnet_depth=args.subnet_depth,
    )

    data_module = MVTecDataModule(
        dataset_path=args.dataset_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )

    def objective(trial):
        trainer.fit(patchcore, data_module)
        latency = patchcore.latency.compute()
        segmentation_f1 = patchcore.segmentation_f1.compute()
        classification_f1 = patchcore.classification_f1.compute()
        trial.set_user_attr("threshold", patchcore.threshold)

        trainer.test(patchcore, data_module)
        trial.set_user_attr("segmentation_f1_test", patchcore.segmentation_f1.compute())
        trial.set_user_attr(
            "classification_f1_test", patchcore.classification_f1.compute()
        )

        return latency, segmentation_f1, classification_f1

    study = create_study(directions=["minimize", "maximize", "maximize"])
    study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument(
        "--dataset_path", type=str, default="/dataB1/tommie_kerssies/MVTec/pill"
    )
    parser.add_argument("--batch_size", type=int, default=267)
    parser.add_argument("--val_ratio", type=float, default=1.0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--k_nn", type=int, default=1)
    parser.add_argument(
        "--supernet_name", type=str, default="ofa_mbv3_d234_e346_k357_w1.0"
    )
    parser.add_argument("--subnet_kernel_size", type=int, default=7)
    parser.add_argument("--subnet_expansion_ratio", type=int, default=6)
    parser.add_argument("--subnet_depth", type=int, default=4)
    parser.add_argument("--patch_channels", type=int, default=112)
    parser.add_argument("--patch_kernel_size", type=int, default=3)
    parser.add_argument("--patch_stride", type=int, default=1)
    parser.add_argument(
        "--extraction_layers", nargs="+", type=int, default=["blocks.6", "blocks.14"]
    )

    main(parser.parse_args())
