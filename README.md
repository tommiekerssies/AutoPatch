This project can be used to easily do experiments with models, using Pytorch Lightning (PL), Weights & Biases, mmcv and more.
The specific purpose is SSL-NAS, but the code is general and should not be limited to SSL-NAS experiments.
Throughout the code, lm is an abbreviation for LightningModule and ldm is an abbreviation for LightningDataModule.

Features:
- Logging of experiments through Weights & Biases
- Automatically save best model
- Easy resuming of experiments (using --resume_run)
- Keep running experiment until model has converged or a certain stop time has been reached
- Easy loading of pre-trained weights from any format (using --prefix_old and --prefix_new)
- Learning rate and batch size finder

Example slurm script to train a ResNet50 on CIFAR100 with ImageNet pretrained weights on 16 gpus (8 per node):
```bash
#!/bin/bash
#SBATCH -p ohau
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --error=run.log
#SBATCH --output=run.log
#SBATCH -t 167:00:00

source /home/tommie_kerssies/miniconda3/etc/profile.d/conda.sh
conda activate SSL-NAS

# replace below with the right socket for the nodes to communicate over
export NCCL_SOCKET_IFNAME=horovod

srun python3 train_resnet_cifar100.py \
  --num_nodes 2 \
  --devices 8 \
  --batch_size 64 \
  --lr 0.06 \
  --num_classes 100 \
  --depth 50 \
  --work_dir /dataB1/tommie_kerssies \
  --prefix_old backbone. \
  --prefix_new model.backbone. \
  --project_name fine-tune_cifar100 \
  --weights_file resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth
```