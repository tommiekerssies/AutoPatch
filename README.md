This project can be used to easily do experiments with models, using Pytorch Lightning (PL), Weights & Biases, mmcv and more.
The specific purpose is SSL-NAS (see app.searcher), but the code is general and should not be limited to SSL-NAS experiments.
Throughout the code, lm is an abbreviation for LightningModule and ldm is an abbreviation for LightningDataModule.

Features:
- Logging of experiments through Weights & Biases
- Easy resuming of experiments (using --resume_run)
- Easy loading of pre-trained weights from any format (using --prefix_old and --prefix_new)
- Auto learning rate and batch size finder

Create environment with:
`conda env create -f env.yml`

Example slurm script to train a ResNet50 with MoCo v2 pretrained weights on 16 gpus (8 per node):
```bash
#!/bin/bash

#SBATCH -p ohau
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --error=run.log
#SBATCH --output=run.log
#SBATCH -t 167:00:00

source /home/tommie_kerssies/miniconda3/etc/profile.d/conda.sh
conda activate SSL-NAS
wandb enabled

export NCCL_SOCKET_IFNAME=horovod

srun python3 train_resnet_cifar100.py \
  --num_nodes 2 \
  --devices 8 \
  --batch_size 3125 \
  --lr 0.16 \
  --seed 0 \
  --num_classes 100 \
  --depth 50 \
  --work_dir /dataB1/tommie_kerssies \
  --weights_file moco_v2_800ep_pretrain.pth.tar \
  --prefix_old module.encoder_q. \
  --prefix_new model.backbone. \
  --project_name fine-tune_cifar100 
```