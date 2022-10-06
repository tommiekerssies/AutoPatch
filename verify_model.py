#%%
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, \
  classification_report
from app.ldm.cifar100_ldm import CIFAR100LDM
from app.lm.resnet_cls_lm import ResNetClsLM
import torch
import matplotlib.pyplot as plt

classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 
'aquarium' ,'fish', 'ray', 'shark', 'trout', 
'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 
'bottles', 'bowls', 'cans', 'cups', 'plates', 
'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 
'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
'bed', 'chair', 'couch', 'table', 'wardrobe', 
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 
'bear', 'leopard', 'lion', 'tiger', 'wolf', 
'bridge', 'castle', 'house', 'road', 'skyscraper', 
'cloud', 'forest', 'mountain', 'plain', 'sea', 
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 
'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 
'crab', 'lobster', 'snail', 'spider', 'worm', 
'baby', 'boy', 'girl', 'man', 'woman', 
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 
'maple', 'oak', 'palm', 'pine', 'willow', 
'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 
'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']

model_path = "/dataB1/tommie_kerssies/fine-tune_cifar100/j84fsjlb/checkpoints/last.ckpt"
model = ResNetClsLM.load_from_checkpoint(model_path).cuda()
data_loader = CIFAR100LDM(
  work_dir='/dataB1/tommie_kerssies', seed=0, 
  batch_size=128, num_workers=1,
).setup(None).val_dataloader()

def test_label_predictions(model, data_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

# %%
actuals, predictions = test_label_predictions(model, data_loader)
print('F1 score: %f' % f1_score(actuals, predictions, average='weighted'))
print('Accuracy score: %f' % accuracy_score(actuals, predictions))

# %%
cm = confusion_matrix(actuals, predictions)
print(cm)
fig = plt.figure(figsize=(24,24))
ax = fig.add_subplot(211)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + classes)
ax.set_yticklabels([''] + classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %%
print(classification_report(actuals, predictions, target_names=classes, digits=5))

# %%
plt.imshow(data_loader.dataset[0][0].permute(1, 2, 0))