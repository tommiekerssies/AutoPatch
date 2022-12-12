# %%
# sourcery skip: raise-specific-error
from pytorch_lightning import seed_everything
import wandb
import torch
from app.lightning_data_module.aoi import AOI
from shutil import copy2
import os


seed = 4
seed_everything(seed, workers=True)
wandb.init(mode="disabled")

#%%
ldm = AOI(
    work_dir="/dataB1/tommie_kerssies/",
    seed=seed,
    batch_size=1,
    val_batch_size=1,
    num_workers=2,
    crop_size=None,
    augment=False,
    pin_memory=False,
    scale_factor=1.0,
    train_fraction=1.0,
    train_folder="multilabel_fix/multilabel_fix/img_without-qtech_without-val_buffer00",
    val_folder="multilabel_v6/val/img",
    lbl_folder="multilabel_fix/multilabel_fix/indexedlabel",
).setup()

#%%
from os.path import basename
from torchvision.utils import save_image
dataloader = AOI(
    work_dir="/dataB1/joris/datasets/aoi/archive",
    seed=seed,
    batch_size=1,
    val_batch_size=1,
    num_workers=64,
    crop_size=256,
    augment=False,
    pin_memory=False,
    scale_factor=1.0,
    train_fraction=1.0,
    train_folder="multilabel_fix/train/img",
    val_folder="multilabel_v6/val/img",
    lbl_folder="multilabel_fix/train/indexedlabel",
).setup().train_dataloader()
for epoch in range(0, 1):
    for batch in dataloader:
        img, img_path, wire_mask = (
            batch["image"][0],
            batch["img_path"][0],
            batch["masks"][0][0],
        )
        if torch.count_nonzero(wire_mask) > 0:
            img = img / 255
            save_image(img, f"/dataB1/tommie_kerssies/wire_crops_val2/{epoch}_{basename(img_path)}")
        print(f"epoch: {epoch}", end="\r")
print()
# %%
# num_no_wire = 0
# num_wire = 0
# dst_path = "/dataB1/tommie_kerssies/multilabel_v6/val/img_only-wire"
# for batch in ldm.val_dataloader():
#     img_path, mask_path, wire_mask = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#         batch["masks"][0][0],
#     )
#     if torch.count_nonzero(wire_mask) > 0:
#         num_wire += 1
#         copy2(img_path, dst_path)
#     else:
#         num_no_wire += 1
#     print(
#         f"num_wire: {num_wire}, num_no_wire: {num_no_wire}",
#         end="\r",
#     )
# print()

# %%
# num_no_wire = 0
# num_wire = 0
# dst_path = "/dataB1/tommie_kerssies/multilabel_fix/multilabel_fix/img_without-qtech_without-val_buffer00_only-wire"
# for batch in ldm.train_dataloader():
#     img_path, mask_path, wire_mask = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#         batch["masks"][0][0],
#     )
#     if torch.count_nonzero(wire_mask) > 0:
#         num_wire += 1
#         copy2(img_path, dst_path)
#     else:
#         num_no_wire += 1
#     print(
#         f"num_wire: {num_wire}, num_no_wire: {num_no_wire}",
#         end="\r",
#     )
# print()

# %%
# num_buffer0 = 0
# num_buffer1 = 0
# dst_path = "/dataB1/tommie_kerssies/multilabel_fix/multilabel_fix/img_without-qtech_without-val_buffer00"
# for batch in ldm.train_dataloader():
#     img_path, mask_path = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#     )
#     if "buffer00" in img_path:
#         num_buffer0 += 1
#         copy2(img_path, dst_path)
#     else:
#         num_buffer1 += 1
#     print(
#         f"num_buffer0: {num_buffer0}, num_buffer1: {num_buffer1}",
#         end="\r",
#     )
# print()

# %%
# import os
# num_holdout = 0
# num_keep = 0
# src_path = "/dataB1/tommie_kerssies/multilabel_fix/multilabel_fix/img_without-qtech/"
# dst_path = "/dataB1/tommie_kerssies/multilabel_fix/multilabel_fix/img_without-qtech_without-val/"
# holdout_files = os.listdir("/dataB1/tommie_kerssies/multilabel_v6/val/img/")
# for file in os.listdir(src_path):
#     if file in holdout_files:
#         num_holdout += 1
#     else:
#         num_keep += 1
#         copy2(
#             src_path + file,
#             dst_path,
#         )
#     print(
#         f"num_holdout: {num_holdout}, num_keep: {num_keep}",
#         end="\r",
#     )
# print()

# %%
# import os
# num_holdout = 0
# num_keep = 0
# src_path = "/dataB1/tommie_kerssies/basesets/"
# holdout_files = os.listdir("/dataB1/joris/datasets/aoi/archive/multilabel_fix/train/img/")
# for file in os.listdir(src_path):
#     if file in holdout_files:
#         num_holdout += 1
#     else:
#         num_keep += 1
#         copy2(
#             src_path + file,
#             "/dataB1/tommie_kerssies/basesets_without_multilabel_fix/",
#         )
#     print(
#         f"num_holdout: {num_holdout}, num_keep: {num_keep}",
#         end="\r",
#     )
# print()

# %%
# TODO: remove low res images by hand
# num_copied = 0
# path = "/dataB1/joris/datasets/aoi/basesets_v2/"
# for root, dirs, files in os.walk(path):
#     for file in files:
#         src_path = f"{root}/{file}"
#         dst_path = src_path.replace("/dataB1/joris/datasets/aoi/basesets/", "")
#         dst_path = dst_path.replace("labelled/", "")
#         dst_path = dst_path.replace("unlabelled/", "")
#         dst_path = "/".join(dst_path.split("/")[1:])
#         dst_path = dst_path.replace("WT_DataBase/", "")
#         dst_path = dst_path.replace("data/", "")
#         dst_path = dst_path.replace("RealImage/", "")
#         dst_path = dst_path.replace("record/", "")
#         dst_path = dst_path.replace("/", "_")
#         dst_path = f"/dataB1/tommie_kerssies/basesets/{dst_path}"
#         # TODO: make a regex out of some of the below
#         if file.endswith(".bmp") and "_WireLabel" not in file and "_Label.bmp" and "00130008.bmp" not in file and "00130007.bmp" not in file and "00130006.bmp" not in file and "WTRL_00130005.bmp" and "00130004.bmp" not in file and "00130003.bmp" not in file and "00130002.bmp" not in file and "00130001.bmp" not in file:
#             num_copied += 1
#             copy2(src_path, dst_path)
#         print(
#             f"num_copied: {num_copied}",
#             end="\r",
#         )
# print()

# %%
# num_non_ignore = 0
# num_ignore = 0
# for batch in ldm.train_dataloader():
#     img_path, mask_path, ignore_mask = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#         batch["ignore_mask"][0],
#     )
#     if torch.count_nonzero(ignore_mask) > 0:
#         num_ignore += 1
#     else:
#         num_non_ignore += 1
#         copy2(img_path, "/dataB1/tommie_kerssies/multilabel_v6/train_non_ignore/img/")
#         copy2(mask_path, "/dataB1/tommie_kerssies/multilabel_v6/train_non_ignore/lbl/")
#     print(
#         f"num_non_ignore: {num_non_ignore}, num_ignore: {num_ignore}",
#         end="\r",
#     )
# print()

# %%
# num_buffer0 = 0
# num_buffer1 = 0
# for batch in ldm.train_dataloader():
#     img_path, mask_path = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#     )
#     if "buffer00" in img_path:
#         num_buffer0 += 1
#         copy2(
#             img_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/train_buffer00/img/",
#         )
#         copy2(
#             mask_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/train_buffer00/lbl/",
#         )
#     else:
#         num_buffer1 += 1
#     print(
#         f"num_buffer0: {num_buffer0}, num_buffer1: {num_buffer1}",
#         end="\r",
#     )
# print()

# %%
# num_no_wire = 0
# num_wire = 0
# for batch in ldm.train_dataloader():
#     img_path, mask_path, wire_mask = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#         batch["masks"][0][0],
#     )
#     if torch.count_nonzero(wire_mask) > 0:
#         num_wire += 1
#         copy2(
#             img_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/train_buffer00_only_wire/img/",
#         )
#         copy2(
#             mask_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/train_buffer00_only_wire/lbl/",
#         )
#     else:
#         num_no_wire += 1
#     print(
#         f"num_wire: {num_wire}, num_no_wire: {num_no_wire}",
#         end="\r",
#     )
# print()

#%% ---------------------------------------------------------------------------------------------------------------------------------------------
# num_cropped = 0
# num_not_cropped = 0
# for batch in ldm.val_dataloader():
#     img_tensor, img_path, mask_path = (
#         batch["image"][0],
#         batch["img_path"][0],
#         batch["mask_path"][0],
#     )
#     if img_tensor.shape[-1] == 4096:
#         num_cropped += 1
#         import cv2
#         from pathlib import Path

#         img = cv2.imread(img_path)
#         mask = cv2.imread(mask_path)
#         for r in range(0, img.shape[0], 2048):
#             for c in range(0, img.shape[1], 2048):
#                 cv2.imwrite(
#                     img_path.replace(".bmp", f"_{r}_{c}.bmp").replace(
#                         "/val/", "/val_cropped/"
#                     ),
#                     img[r : r + 2048, c : c + 2048, :],
#                 )
#                 cv2.imwrite(
#                     mask_path.replace(".png", f"_{r}_{c}.png").replace(
#                         "/val/", "/val_cropped/"
#                     ),
#                     mask[r : r + 2048, c : c + 2048, :],
#                 )
#     else:
#         copy2(img_path, "/dataB1/tommie_kerssies/multilabel_v6/val_cropped/img/")
#         copy2(mask_path, "/dataB1/tommie_kerssies/multilabel_v6/val_cropped/lbl/")
#         num_not_cropped += 1
#     print(
#         f"num_not_cropped: {num_not_cropped}, num_cropped: {num_cropped}",
#         end="\r",
#     )
# print()

# %%
# num_non_ignore = 0
# num_ignore = 0
# for batch in ldm.val_dataloader():
#     img_path, mask_path, ignore_mask = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#         batch["ignore_mask"][0],
#     )
#     if torch.count_nonzero(ignore_mask) > 0:
#         num_ignore += 1
#     else:
#         num_non_ignore += 1
#         copy2(
#             img_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/val_cropped_non_ignore/img/",
#         )
#         copy2(
#             mask_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/val_cropped_non_ignore/lbl/",
#         )
#     print(
#         f"num_non_ignore: {num_non_ignore}, num_ignore: {num_ignore}",
#         end="\r",
#     )
# print()

# %%
# num_buffer0 = 0
# num_buffer1 = 0
# for batch in ldm.val_dataloader():
#     img_path, mask_path = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#     )
#     if "buffer00" in img_path:
#         num_buffer0 += 1
#         copy2(
#             img_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/val_cropped_buffer00/img/",
#         )
#         copy2(
#             mask_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/val_cropped_buffer00/lbl/",
#         )
#     else:
#         num_buffer1 += 1
#     print(
#         f"num_buffer0: {num_buffer0}, num_buffer1: {num_buffer1}",
#         end="\r",
#     )
# print()

#%%
# num_no_wire = 0
# num_wire = 0
# for batch in ldm.val_dataloader():
#     img_path, mask_path, wire_mask = (
#         batch["img_path"][0],
#         batch["mask_path"][0],
#         batch["masks"][0][0],
#     )
#     if torch.count_nonzero(wire_mask) > 0:
#         num_wire += 1
#         copy2(
#             img_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/val_cropped_buffer00_only_wire/img/",
#         )
#         copy2(
#             mask_path,
#             "/dataB1/tommie_kerssies/multilabel_v6/val_cropped_buffer00_only_wire/lbl/",
#         )
#     else:
#         num_no_wire += 1
#     print(
#         f"num_wire: {num_wire}, num_no_wire: {num_no_wire}",
#         end="\r",
#     )
# print()

# %%
# num_cropped = 0
# num_not_cropped = 0
# size_before = 2048
# size_after = 256
# old_dir = "val_cropped_buffer00_only_wire"
# new_dir = "val_cropped_buffer00_only_wire_256"
# for batch in ldm.val_dataloader():
#     img_tensor, img_path, mask_path = (
#         batch["image"][0],
#         batch["img_path"][0],
#         batch["mask_path"][0],
#     )
#     if img_tensor.shape[-1] == size_before:
#         num_cropped += 1
#         import cv2
#         from pathlib import Path

#         img = cv2.imread(img_path)
#         mask = cv2.imread(mask_path)
#         for r in range(0, img.shape[0], size_after):
#             for c in range(0, img.shape[1], size_after):
#                 cv2.imwrite(
#                     img_path.replace(".bmp", f"_{r}_{c}.bmp").replace(old_dir, new_dir),
#                     img[r : r + size_after, c : c + size_after, :],
#                 )
#                 cv2.imwrite(
#                     mask_path.replace(".png", f"_{r}_{c}.png").replace(
#                         old_dir, new_dir
#                     ),
#                     mask[r : r + size_after, c : c + size_after, :],
#                 )
#     else:
#         copy2(
#             img_path,
#             f"/dataB1/tommie_kerssies/multilabel_v6/{new_dir}/img/",
#         )
#         copy2(
#             mask_path,
#             f"/dataB1/tommie_kerssies/multilabel_v6/{new_dir}/lbl/",
#         )
#         num_not_cropped += 1
#     print(
#         f"num_not_cropped: {num_not_cropped}, num_cropped: {num_cropped}",
#         end="\r",
#     )
# print()

# %%
# import os
# num_buffer0 = 0
# num_buffer1 = 0
# path = "/dataB1/joris/datasets/aoi/basesets_processed/WT_database_jpeg/"
# for file in os.listdir(path):
#     if "buffer00" in file:
#         num_buffer0 += 1
#         copy2(
#             path + file,
#             "/dataB1/tommie_kerssies/WT_database_jpeg_buffer00",
#         )
#     else:
#         num_buffer1 += 1
#     print(
#         f"num_buffer0: {num_buffer0}, num_buffer1: {num_buffer1}",
#         end="\r",
#     )
# print()