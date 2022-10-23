# %%
import os
from pytorch_lightning import seed_everything
from importlib import reload
import cv2
import wandb
import numpy as np
import torch
from app.lightning_module.aoi import AOI as AOI_LM
from app.lightning_data_module.aoi_multilabel_v6 import AOI as AOI_LDM
import matplotlib.pyplot as plt


torch.cuda.empty_cache()
seed = 0
seed_everything(seed, workers=True)
wandb.init(mode="disabled")

ldm = AOI_LDM(
    work_dir="/dataB1/tommie_kerssies/",
    seed=seed,
    batch_size=1,
    num_workers=2,
    crop_size=512,
    disable_augmentations=True,
).setup()

# model_path = "/dataB1/tommie_kerssies/fine-tune_aoi/2e658nl5/checkpoints/last.ckpt"
model_path = "/dataB1/tommie_kerssies/fine-tune_aoi/2e658nl5/checkpoints/last.ckpt"
model = AOI_LM.load_from_checkpoint(
    model_path,
    ldm=ldm,
    resume_run_id="2e658nl5",
).cuda()


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


# %%
j = 0
for batch in ldm.train_dataloader():
    if j > 10:
        break
    j += 1
    imgs, masks, ignore_mask = (
        batch["image"].cuda(),
        batch["masks"],
        batch["ignore_mask"],
    )
    model = model.eval()
    y_hat = model(imgs.float()).detach()
    masks = torch.stack(masks).permute(1, 0, 2, 3)
    visualize(
        x=imgs[0].detach().permute(1, 2, 0).cpu(),
        masks=masks[0].detach().float().permute(1, 2, 0).cpu(),
        ignore_mask=ignore_mask[0].detach().cpu(),
        y_hat=y_hat[0].detach().sigmoid().permute(1, 2, 0).cpu(),
    )

#%%
j = 0
for batch in ldm.train_dataloader():
    imgs, masks, ignore_mask = (
        batch["image"].cuda(),
        batch["masks"],
        batch["ignore_mask"],
    )
    if j > 5:
        break
    j += 1
    model = model.eval()
    x = model.model.extract_feat(imgs.float())
    y_hat = model.model.decode_head(x)
    visualize(
        x=imgs[0].detach().permute(1, 2, 0).cpu(),
        y_hat=y_hat[0].detach().sigmoid().permute(1, 2, 0).cpu(),
    )
