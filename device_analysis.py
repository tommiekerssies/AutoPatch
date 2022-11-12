# %%
import os
import re
import numpy as np

def get_devices(dir):
    devices = []
    for file in os.listdir(dir):
        file = re.sub(r"_\d+_", "", file)
        file = re.sub(r"_\d+.png", "", file)
        file = re.sub(r"-\d+_", "", file)
        file = re.sub(r"WTRI\d+_", "", file)
        file = re.sub(r"FBDI\d+_\d", "", file)
        file = re.sub(r"FBDI\d+_", "", file)
        file = re.sub(r"_FBDI\d+", "", file)
        file = re.sub(r"SBDL\d+_", "", file)
        file = re.sub(r"SBDI\d+_", "", file)
        file = re.sub(r"WRTI\d+_", "", file)
        file = re.sub(r"STEI\d+", "", file)
        file = re.sub(r"\d+_WTRL", "", file)
        file = re.sub(r"WTRL_\d+", "", file)
        file = re.sub(r"FBDL\d+_", "", file)
        file = re.sub(r"Case\d+_", "", file)
        file = re.sub(r"SBI\d+_", "", file)
        file = re.sub(r"diag_case\d+_", "", file)
        file = re.sub(r"case\d+_", "", file)
        file = re.sub(r"_buffer00_\d+\.png", "", file)
        file = file.replace("_buffer00.jpg", "")
        file = file.replace("_buffer00", "")
        file = file.replace("buffer00", "")
        file = file.replace("View00", "")
        file = file.replace(".png", "")
        file = file.replace(".bmp", "")
        file = file.replace("WTRI", "")
        file = file.replace("_Sample1", "")
        file = file.replace("__", "")
        file = file.replace(" ", "")
        file = re.sub(r"_$", "", file)
        devices.append(file)
    unique_devices, device_counts = np.unique(devices, return_counts=True)
    return dict(sorted(dict(zip(unique_devices, device_counts)).items(), key=lambda x: x[1], reverse=True))
    
unlabeled = get_devices("/dataB1/tommie_kerssies/WT_database_jpeg")
labeled = get_devices("/dataB1/tommie_kerssies/multilabel_v6/train_buffer00_only_wire/img")

devices_in_both = list(set(unlabeled.keys()).intersection(set(labeled.keys())))
sorted({device: unlabeled[device] for device in devices_in_both}.items(), key=lambda x: x[1], reverse=True)