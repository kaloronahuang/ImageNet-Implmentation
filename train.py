# train.py
import torch
import os
from torch import nn
import numpy as np
import cv2
from tqdm import tqdm
import time
# ResNet model
from models.ResNet import resnet50
# DALI
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from ImageNetLoader import ImageNetDALIPipeline
import json

# Training Hyperparameters
learning_rate = 0.01
momentum = 0.9
batch_size = 32
epoch_limit = 20

dataset_dir = '/mnt/store/ImageNet/dataset/ILSVRC/'
img_dir = os.path.join(dataset_dir, 'Data', 'CLS-LOC')
annotation_dir = os.path.join(dataset_dir, 'Annotations', 'CLS-LOC', 'val')
training_dataset_dir = os.path.join(img_dir, 'train')
eval_dataset_dir = os.path.join(img_dir, 'eval')

save_dir = './saves/'

current_checkpoint = eval(input('Please input the checkpoint ID: '))
if current_checkpoint == 0:
    model = resnet50().cuda()
    print('New model create.')
else:
    model_path = os.path.join(save_dir, f'model-ResNet-{current_checkpoint}.pth')
    print(f'Loading from {model_path}...')
    model = resnet50()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    print('Loaded!')

records = {
    'epoch_reports' : [

    ],
    'eval_reports' : [

    ],
    'current_iter' : 0
}

def LoadHistory():
    global records
    records = json.load(open('history.json', 'r'))

def SaveHistory():
    json.dump(records, open('history.json', 'w'))

if os.path.exists('history.json'):
    LoadHistory()

pipe = ImageNetDALIPipeline(batch_size=batch_size,
                            num_threads=4,
                            device_id=0,
                            seed=12,
                            data_dir=os.path.join(dataset_dir, "Data", "CLS-LOC", "train"),
                            crop=224,
                            size=256,
                            shard_id=0,
                            num_shards=1,
                            is_training=True)
pipe.build()
training_dataloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.FILL, auto_reset=True)

pipe = ImageNetDALIPipeline(batch_size=batch_size,
                            num_threads=4,
                            device_id=0,
                            seed=12,
                            data_dir=os.path.join(dataset_dir, "Data", "CLS-LOC", "eval"),
                            crop=224,
                            size=256,
                            shard_id=0,
                            num_shards=1,
                            is_training=False)
pipe.build()
eval_dataloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.FILL, auto_reset=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def train_loop(dataloader, model, loss_fn, optimizer, enable_log=True):
    global current_checkpoint

    size = dataloader.size
    pgbr = tqdm(total=size)
    loss_records = []
    print(f'Epoch {current_checkpoint + 1}\n')

    iter_id = records['current_iter']
    epoch_report = {'epoch_id': current_checkpoint + 1, 'start_iter': iter_id}
    
    tic = time.clock()

    for batch, data in enumerate(dataloader):
        for d in data:
            X, Y = d["data"], d["label"].squeeze(-1).long()
            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pgbr.update(batch_size)
            iter_id += 1
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                loss_records.append({'iter': iter_id, 'loss': float(loss)})

    toc = time.clock()
    
    print(f'Epoch {current_checkpoint + 1} Completed')
    torch.save(model.state_dict(), os.path.join(save_dir, f'model-ResNet-{current_checkpoint + 1}.pth'))
    current_checkpoint += 1

    epoch_report['elapse_seconds'] = toc - tic
    epoch_report['end_iter'] = iter_id
    epoch_report['iter_loss'] = loss_records
    if enable_log:
        records['current_iter'] = iter_id
        records['epoch_reports'].append(epoch_report)
        SaveHistory()
    print(f'Model & Record file saved. Elapsed Time: {toc - tic}')

def test_loop(dataloader, model, loss_fn, enable_log=True):
    global current_checkpoint
    size = dataloader.size
    num_batches = 0
    test_loss, correct = 0, 0
    pgbr = tqdm(size)

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            for d in data:
                X, y = d["data"], d["label"].squeeze(-1).long()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                num_batches += 1
                pgbr.update(batch_size)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if enable_log:
        eval_report = {'eval_checkpoint': current_checkpoint, 'avg_loss': float(test_loss), 'accuracy': float(100 * correct), 'time': time.asctime()}
        records['eval_reports'].append(eval_report)
        SaveHistory()
        print('Eval report saved, history file saved.')

while current_checkpoint < epoch_limit:
    train_loop(training_dataloader, model, loss_fn, optimizer)
    test_loop(eval_dataloader, model, loss_fn)
    training_dataloader.reset()
    eval_dataloader.reset()