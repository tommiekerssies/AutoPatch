from pytorch_lightning import LightningModule
from mmselfsup.models import build_algorithm
from torch.optim import Adam
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy


class BaseClsLM(LightningModule):
  def __init__(self, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    
    self.train_acc = Accuracy()
    self.val_acc = Accuracy()
    self.log_args = dict(sync_dist=True, on_step=False,
                         on_epoch=True)
    
  def forward(self, imgs):
    features = self.model.extract_feat(imgs)
    preds = self.model.head(features)[0]
    return preds
  
  def step(self, batch, metric, metric_name, loss_name):
    imgs, labels = batch
    preds = self(imgs)
    
    metric(preds, labels)    
    self.log(metric_name, metric, **self.log_args)
    
    loss = cross_entropy(preds, labels)
    self.log(loss_name, loss, **self.log_args)
    
    return loss
  
  def training_step(self, batch, batch_idx):
    return self.step(batch, self.train_acc, 'train_acc', 'train_loss')
  
  def validation_step(self, batch, batch_idx):
    return self.step(batch, self.val_acc, 'val_acc', 'val_loss')
  
  def configure_optimizers(self):
    return Adam(self.parameters())