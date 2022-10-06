from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from app.lm.base_lm import BaseLM
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BaseClsLM(BaseLM):
  def __init__(self, model_cfg, **kwargs):
    super().__init__(model_cfg)
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
    optimizer = SGD(self.parameters(),
                    lr=self.hparams.lr,
                    momentum=self.hparams.momentum,
                    weight_decay=self.hparams.weight_decay,
                    nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    
    return dict(optimizer=optimizer, 
                lr_scheduler=dict(scheduler=scheduler, monitor='val_loss'))

  @staticmethod
  def add_argparse_args(parent_parser):
    parent_parser = super(BaseClsLM, BaseClsLM) \
        .add_argparse_args(parent_parser)
    parser = parent_parser.add_argument_group("BaseClsLM")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float)
    return parent_parser