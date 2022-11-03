# %%
from itertools import islice
from statistics import mean, stdev
from pytorch_lightning import seed_everything
import wandb
import torch
from app.lightning_module.multi_label_sem_seg.fcn import FCN
from app.lightning_data_module.aoi import AOI
import matplotlib.pyplot as plt


torch.cuda.set_device("cuda:4")
torch.cuda.empty_cache()
seed = 0
seed_everything(seed, workers=True)
wandb.init(mode="disabled")

ldm = AOI(
    work_dir="/dataB1/tommie_kerssies/",
    seed=seed,
    num_workers=0,
    batch_size=1,
    crop_size=256,
    augment=False,
    val_batch_size=1,
    scale_factor=8.0,
).setup()

model_path = "/dataB1/tommie_kerssies/fine-tune_aoi/133mlesc/checkpoints/last.ckpt"
model = FCN.load_from_checkpoint(model_path)
# model = AOI_LM(
#     stem_width=32,
#     body_width=[32, 64, 128, 256],
#     body_depth=[2, 2, 2, 2],
#     num_classes=3,
#     supernet_run_id=None,
# )
# model = model.cuda()


def visualize(figsize=(40, 10), **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


for batch in islice(ldm.train_dataloader(), 11, 12):
    img, masks, ignore_mask = (
        batch["image"],
        batch["masks"],
        batch["ignore_mask"],
    )
    model = model.eval()
    out_raw = model.model.decode_head(model.model.extract_feat(img.float()))
    out, _ = model(batch)
    visualize(
        img=img[0].detach().permute(1, 2, 0).cpu(),
        # ignore_mask=ignore_mask[0].detach().cpu(),
        mask=masks[0][0].detach().float().cpu(),
        out_raw=out_raw[0].detach().sigmoid().permute(1, 2, 0).cpu(),
        out=out[0].detach().sigmoid().permute(1, 2, 0).cpu(),
    )

#%%
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
repetitions = 100
timings = []

model = model.eval()

with torch.no_grad():
    for batch in islice(ldm.val_dataloader(), 0, 10):
        img = batch["image"].cuda().float()
        _ = model(img)

    for batch in islice(ldm.val_dataloader(), 0, repetitions):
        img = batch["image"].cuda().float()
        starter.record()
        out = model(img)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

visualize(out=out[0].detach().sigmoid().permute(1, 2, 0).cpu())
print(f"Mean: {mean(timings)}")
print(f"Stdev: {stdev(timings)}")
