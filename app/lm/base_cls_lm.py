from torch.optim import SGD
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from app.lm.base_lm import BaseLM


class BaseClsLM(BaseLM):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
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
  
  def optimizer_step(
      self,
      epoch,
      batch_idx,
      optimizer,
      optimizer_idx,
      optimizer_closure,
      on_tpu=False,
      using_native_amp=False,
      using_lbfgs=False,
  ):
    optimizer.step(closure=optimizer_closure)
    
    # learning rate warm-up
    if self.trainer.global_step < self.trainer.num_training_batches:
      lr_scale = float(self.trainer.global_step + 1) \
                 / self.trainer.num_training_batches
      for i, pg in enumerate(optimizer.param_groups):
        pg['lr'] = lr_scale * self.hparams.lr
        print(f"Step {self.trainer.global_step}: increasing learning rate"\
              f"of group {i} to {pg['lr']}.")

  def configure_optimizers(self):
    optimizer = SGD(self.parameters(),
                    lr=self.hparams.lr,
                    momentum=self.hparams.momentum,
                    weight_decay=self.hparams.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    return dict(optimizer=optimizer, 
                lr_scheduler=dict(scheduler=scheduler, monitor='val_loss'))

  @staticmethod
  def add_argparse_args(parent_parser):
    parent_parser = super(BaseClsLM, BaseClsLM) \
        .add_argparse_args(parent_parser)
    parser = parent_parser.add_argument_group("BaseClsLM")
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_classes", type=int)
    return parent_parser