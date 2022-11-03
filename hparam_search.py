# %%
from app.lightning_data_module.aoi import AOI
from time import time


for num_workers in range(20, 2, -2):
    ldm = AOI(
        work_dir="/dataB1/tommie_kerssies/",
        seed=0,
        num_workers=num_workers,
        batch_size=14,
        val_batch_size=14,
        crop_size=256,
        augment=False,
        scale_factor=8.0,
    ).setup()
    start = time()
    for _ in ldm.val_dataloader():
        pass
    end = time()
    print(f"Finish with:{end - start} second, num_workers={num_workers}")
# found num_workers=18

#%%
for batch_size in range(22, 2, -2):
    ldm = AOI(
        work_dir="/dataB1/tommie_kerssies/",
        seed=0,
        num_workers=18,
        batch_size=batch_size,
        val_batch_size=batch_size,
        crop_size=256,
        augment=False,
        scale_factor=8.0,
    ).setup()
    start = time()
    for _ in ldm.val_dataloader():
        pass
    end = time()
    print(f"Finish with:{end - start} second, batch_size={batch_size}")
# found batch_size=14
