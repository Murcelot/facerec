import torch
from torch.optim import Adam
from ModArcFaceLoss import ArcFaceLoss
from models.TranSSL import TranSSL
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.v2 import CutMix
from torchvision.transforms.v2 import MixUp
from datasets import RealDataset, WholeDataset
from torch.utils.data import DataLoader, random_split
from random import randint
import os

#Warmup class for more stable learning procces
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_epoch, min_lr = 1e-9):
        self.warmup = warmup
        self.max_num_iters = max_epoch
        self.min_lr = min_lr
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch ==0 :
            return [self.min_lr]
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

#Define modules
model = TranSSL()
optimizer = Adam(model.parameters(), lr=0.0001)
max_epochs = 100
warmup_epochs = 0.2 * max_epochs
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs, max_epochs, 1e-7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8

#Image transforms
original_transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

#Image augmentations
augmentations = [None,
                 [MixUp(num_classes=20000, alpha=0.2),CutMix(num_classes=20000, alpha=0.2)],
                 [MixUp(num_classes=20000, alpha=0.5),CutMix(num_classes=20000, alpha=0.5)],
                 [MixUp(num_classes=20000, alpha=1),CutMix(num_classes=20000, alpha=1)],
                 None
                 ]

#Datasets
train_dataset_real = RealDataset('train', 'meta.json', original_transform)
train_dataset_whole, val_dataset_whole = random_split(WholeDataset('train', 'meta.json', original_transform), [0.8, 0.2])

#Dataloaders
train_dataloader_real = DataLoader(train_dataset_real, batch_size, shuffle=True)
train_dataloader_whole = DataLoader(train_dataset_whole, batch_size, shuffle=True)
val_dataloader_whole = DataLoader(val_dataset_whole, batch_size, shuffle=True)

#Epoch switcher
switch_epochs = [0.1 * max_epochs, 0.2 * max_epochs, 0.3 * max_epochs, 0.4 * max_epochs, max_epochs]

#Criterion
criterion = ArcFaceLoss(in_features=512, num_classes=20000, smoothing_epsilon=0.2)

def training_loop(train_dataloaders, val_dataloader, num_epochs, switch_epochs, model, optimizer, scheduler, criterion, device, save_every, start_epoch):
    
    model = model.to(device)
    criterion = criterion.to(device)
    curr_dataloader = train_dataloaders[0]
    curr_aug_num = 0

    # Training and validating loop
    for epoch in range(start_epoch, num_epochs):
        epoch_train_loss = 0.
        epoch_val_loss = 0.

        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print("Training...")
        model.train()

        #Switching dataloaders
        if epoch == switch_epochs[0]:
            curr_dataloader = train_dataloaders[1]
            curr_aug_num = 1
            print('Dataloader switched!')
        
        if epoch == switch_epochs[curr_aug_num]:
            curr_aug_num += 1

        #Training
        for i, data in enumerate(tqdm(curr_dataloader)):
            #prepare optimizer
            optimizer.zero_grad()

            #prepare batch
            if augmentations[curr_aug_num]:
                mixup_or_cutmix = randint(0,1)
                data = augmentations[curr_aug_num][mixup_or_cutmix](data)   
            
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            #get outputs from model
            outputs = model(images)
            #get loss
            batch_loss = criterion(outputs, labels)
            epoch_train_loss += batch_loss

            #Backprop and step
            batch_loss.backward()
            optimizer.step()
        
        #Scheduler step
        scheduler.step()

        epoch_train_loss /= len(curr_dataloader)
        print('Epoch loss: {:.3f}'.format(epoch_train_loss))
        print('Validating...')

        #Validating
        model.eval()
        for i, data in enumerate(tqdm(val_dataloader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                #get outputs from model
                outputs = model(outputs)
                #get loss
                batch_loss = criterion(outputs, labels)

            epoch_val_loss += batch_loss
        epoch_val_loss /= len(val_dataloader)
        print('Epoch val loss: {:.3f}'.format(epoch_val_loss))

        # Save checkpoint
        if (epoch % save_every == 0) or (epoch == num_epochs - 1):
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'model_opt': optimizer.state_dict(),
                'model_scheduler' : scheduler.state_dict(),
                'criterion' : criterion.state_dict(),
                'train_loss' : epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))
    return 

#Looking for last checkpoint, if exists
directory = os.path.join(os.getcwd(), 'checkpoints')
last_checkpoint_num = 0
last_checkpoint_path = None
if os.path.exists(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for fname in filenames:
            checkpoint_path = os.path.join(dirpath,fname)
            checkpoint_num = int(fname[:-4].split('_')[0])
            if checkpoint_num >= last_checkpoint_num:
                last_checkpoint_num = checkpoint_num
                last_checkpoint_path = checkpoint_path

#Uploading checkpoint weights, if exists
if last_checkpoint_path:
    last_checkpoint = torch.load(last_checkpoint_path, map_location = torch.device('cpu'))
    epoch = last_checkpoint['epoch']
    model.load_state_dict(last_checkpoint['model'])
    optimizer.load_state_dict(last_checkpoint['model_opt'])
    scheduler.load_state_dict(last_checkpoint['model_scheduler'])
    criterion.load_state_dict(last_checkpoint['criterion'])

#Configuring optimizer to CUDA
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

training_loop((train_dataloader_real, train_dataloader_whole), val_dataloader_whole, max_epochs, switch_epochs, model, optimizer, scheduler, criterion, device, 5, epoch)
print('Done!')