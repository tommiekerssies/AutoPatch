# %%
from itertools import islice
from statistics import mean, stdev
import cv2
from pytorch_lightning import seed_everything
import wandb
import torch
from app.lightning_module.multi_label_sem_seg.fcn import FCN
from app.lightning_data_module.aoi import AOI
import matplotlib.pyplot as plt


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

torch.cuda.set_device("cuda:0")
torch.cuda.empty_cache()
seed = 0
seed_everything(seed, workers=True)
wandb.init(mode="disabled")
work_dir="/dataB1/tommie_kerssies/"

ldm = AOI(
    work_dir=work_dir,
    seed=seed,
    num_workers=0,
    batch_size=1,
    crop_size=512,
    augment=False,
    val_batch_size=1,
).setup()

#%%
model = FCN(
    work_dir=work_dir,
    stem_width=64,
    body_width=[64, 128],
    body_depth=[2, 2],
    num_classes=1,
    supernet_run_id=None,
    resume_run_id=None,
    weights_file="/home/tommie_kerssies/solo-learn/trained_models/byol/b9kdrl1f/last.ckpt",
    weights_prefix="backbone.",
)
model = model.cuda()

# %%
model_path = "/dataB1/tommie_kerssies/fine-tune_aoi/2xyagx8l/checkpoints/last.ckpt"
model = FCN.load_from_checkpoint(model_path).cuda()

#%%
# model = AOI_LM(
#     stem_width=32,
#     body_width=[32, 64, 128, 256],
#     body_depth=[2, 2, 2, 2],
#     num_classes=3,
#     supernet_run_id=None,
# )
# model = model.cuda()

#%%
for batch in islice(ldm.train_dataloader(), 0, 10):
    img, masks, ignore_mask = (
        batch["image"],
        batch["masks"],
        batch["ignore_mask"],
    )
    out, aux_outs = model.train()(batch)
    visualize(
        img=img[0].detach().permute(1, 2, 0).cpu(),
        # ignore_mask=ignore_mask[0].detach().cpu(),
        mask=masks[0][0].detach().float().cpu(),
        # out_raw=out_raw[0].detach().sigmoid().permute(1, 2, 0).cpu(),
        aux_out=aux_outs[0][0].detach().sigmoid().permute(1, 2, 0).cpu(),
        out=out[0].detach().sigmoid().permute(1, 2, 0).cpu(),
    )

#%%
import os
import cv2
from torchvision.transforms import ToTensor
path = f"{ldm.train_path}/img/"
for file in os.listdir(path)[0:10]:
    img = cv2.imread(path + file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ToTensor()(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    x = model.model.backbone(img.float())[-1]
    # out = model.model.decode_head(x)
    # out = out.detach()
    # out = out.sigmoid()
    # out = out.squeeze(0)
    # out = out.cpu()
    x = x.detach()
    x = x.squeeze(0)
    x = torch.mean(x, dim=0)
    x = x.cpu()
    img = img[0]
    img = img.permute(1, 2, 0)
    img = img.cpu()
    visualize(x=x, img=img)

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
