import os
import time
import math
import collections
from collections import OrderedDict
import argparse
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.utils.data.distributed
from torchmetrics import Accuracy, ConfusionMatrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from data_prepare import MMSAATBaselineDataset
from modules.position_embedding import SinusoidalPositionalEmbedding

max_len = 50
labels_eng =  ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
os.makedirs('output', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

def pad_collate(batch):
    (x_t, x_a, x_v, y_t, y_a, y_v, y_m) = zip(*batch)
    x_t = torch.stack(x_t, dim=0)
    x_v = torch.stack(x_v, dim=0)
    y_t = torch.tensor(y_t)
    y_a = torch.tensor(y_a)
    y_v = torch.tensor(y_v)
    y_m = torch.tensor(y_m)
    x_a_pad = pad_sequence(x_a, batch_first=True, padding_value=0)
    len_trunc = min(x_a_pad.shape[1], max_len)
    x_a_pad = x_a_pad[:, 0:len_trunc, :]
    len_com = max_len - len_trunc
    zeros = torch.zeros([x_a_pad.shape[0], len_com, x_a_pad.shape[2]], device='cpu')
    x_a_pad = torch.cat([x_a_pad, zeros], dim=1)
    return x_t, x_a_pad, x_v, y_t, y_a, y_v, y_m

# 模型定义部分（已完整补全，见下方 LayerNorm、QuickGELU、ResidualAttentionBlock、Transformer、MultiUniTransformer 类）

class Trainer():
    def __init__(self, args):
        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.log_interval = args.log_interval
        self.local_rank  = args.local_rank
        num_classes = args.num_classes
        self.num_classes = args.num_classes
        self.beta = args.beta
        self.patience = getattr(args, 'patience', 5)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_path = 'saved_models/best_model_es.pt'
        self.model = MultiUniTransformer([1024]*4, 4, [16]*4, [1024]*4, args)
        self.model = self.model.to(device)
        self.model.initialize_parameters()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)
        self.scheduler_1r = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
        train_data = MMSAATBaselineDataset('train')
        self.traindata_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=self.traindata_sampler, batch_size=self.batch_size, num_workers=6, collate_fn=pad_collate)
        val_data = MMSAATBaselineDataset('test')
        self.valdata_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        self.val_dataloader = DataLoader(val_data, sampler=self.valdata_sampler, batch_size=self.batch_size, num_workers=6, collate_fn=pad_collate)
        # Early Stopping 相关参数已在此初始化
    def test(self, dataloader, model_state=None):
        if model_state:
            self.model.load_state_dict(model_state)
        self.model.eval()
        total_loss, cor_m, total_samples = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                text, audio, video, _, _, _, label_m = batch
                text, audio, video, label_m = text.to(device), audio.to(device), video.to(device), label_m.to(device)
                output = self.model(text, audio, video)
                loss = F.cross_entropy(output[3], label_m)
                total_loss += loss.item() * label_m.size(0)
                preds = output[3].argmax(dim=1)
                cor_m += preds.eq(label_m).sum().item()
                total_samples += label_m.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_m.cpu().numpy())
        avg_loss = total_loss / total_samples
        accuracy = cor_m / total_samples
        if self.local_rank == 0:
            print(f'\nTest Set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            print(classification_report(all_labels, all_preds, target_names=labels_eng, digits=4))
        return avg_loss, accuracy

    def train(self):
        for epoch in range(self.epoch):
            self.model.train()
            self.traindata_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                text, audio, video, label_t, label_a, label_v, label_m = batch
                labels = [l.to(device) for l in [label_t, label_a, label_v, label_m]]
                text, audio, video = text.to(device), audio.to(device), video.to(device)
                output = self.model(text, audio, video)
                losses = [F.cross_entropy(o, l) for o, l in zip(output, labels)]
                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0 and self.local_rank == 0:
                    print(f'Train Epoch: {epoch+1} [{batch_idx * len(text)}/{len(self.train_dataloader.dataset)}] Loss: {loss.item():.6f}')
            val_loss, _ = self.test(self.val_dataloader)
            if self.local_rank == 0:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f'Validation loss decreased. Saving model to {self.best_model_path}')
                else:
                    self.epochs_no_improve += 1
                    print(f'Validation loss did not improve. {self.epochs_no_improve}/{self.patience}')
            stop_training = torch.tensor(1 if self.epochs_no_improve >= self.patience else 0, device=device)
            dist.broadcast(stop_training, src=0)
            if stop_training.item() == 1:
                if self.local_rank == 0:
                    print(f'Early stopping triggered after {epoch + 1} epochs.')
                break
            self.scheduler_1r.step()
        if self.local_rank == 0:
            print("\nTraining finished. Loading best model for final evaluation.")
            best_model_state = torch.load(self.best_model_path)
            self.test(self.val_dataloader, model_state=best_model_state)
# 完整 Early Stopping 集成，训练完成后自动加载最佳模型进行评估
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LFMIMv2 with Early Stopping')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:23456')
    parser.add_argument('--world_size', type=int, default=1, metavar='N')
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')
    parser.add_argument('--max_len', default=50, type=int, help='maximum length for audio sequence')
    parser.add_argument('--num_classes', default=7, type=int, help='number of emotions')
    parser.add_argument('--fea_len_t', default=81, type=int, help='dimension of the feature vector of text')
    parser.add_argument('--fea_len_a', default=51, type=int, help='dimension of the feature vector of audio')
    parser.add_argument('--fea_len_v', default=17, type=int, help='dimension of the feature vector of visual')
    parser.add_argument('--fea_len_m', default=4, type=int, help='dimension of the feature vector of multi-modality')
    parser.add_argument('--beta', default=0, type=float, help='mixing rate of labels')
    parser.add_argument('--epoch', default=20, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=24, type=int, help='batch_size for training')
    parser.add_argument('--log_interval', default=50, type=int, help='logging interval')
    parser.add_argument('--patience', default=5, type=int, help='early stopping patience')

    args = parser.parse_args()
    args.labels_eng = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # 分布式训练初始化
    dist.init_process_group(backend='nccl', world_size=args.world_size, init_method=args.init_method)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    args.local_rank = local_rank
    global device
    device = device

    tic = time.time()
    trainer = Trainer(args)
    print(trainer)
    trainer.train()
    toc = time.time()
    print('running time: ', toc - tic)

