# %%
from app.lightning_data_module.aoi import AOI
from time import time
from tqdm import tqdm

# %%
for num_workers in range(14, 20, 2):
    ldm = AOI(
        work_dir="/dataB1/tommie_kerssies/",
        seed=0,
        num_workers=num_workers,
        batch_size=5,
        val_batch_size=2,
        crop_size=None,
        augment=False,
        scale_factor=1.0,
    ).setup()
    start = time()
    for _ in tqdm(ldm.val_dataloader()):
        pass
    end = time()
    print(f"Finish with:{end - start} second, num_workers={num_workers}")
# found num_workers=18

#%%
for batch_size in range(2, 22, 1):
    ldm = AOI(
        work_dir="/dataB1/tommie_kerssies/",
        seed=0,
        num_workers=8,
        batch_size=batch_size,
        val_batch_size=batch_size,
        crop_size=None,
        augment=False,
        scale_factor=1.0,
    ).setup()
    start = time()
    for _ in tqdm(ldm.val_dataloader()):
        pass
    end = time()
    print(f"Finish with:{end - start} second, batch_size={batch_size}")
# found batch_size=14
