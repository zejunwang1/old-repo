# coding: UTF-8
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from utils import get_time_dif

def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def evaluate(config, model, data_iter, flag=False):
    model.eval()
    total_loss = 0.0
    pred_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for contents, labels in data_iter:
            outputs = model(contents)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss
            labels = labels.data.cpu().numpy()
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            pred_all = np.append(pred_all, pred)
    acc = metrics.accuracy_score(labels_all, pred_all)
    if flag:
        report = metrics.classification_report(labels_all, pred_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, pred_all)
        return acc, total_loss / len(data_iter), report, confusion
    return acc, total_loss / len(data_iter)

def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                pred = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, pred)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.model_path)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  
        if flag:
            break
    model.load_state_dict(torch.load(config.model_path))
    start_time = time.time()
    dev_acc, dev_loss, dev_report, dev_confusion = evaluate(config, model, dev_iter, flag=True)
    msg = 'Dev Loss: {0:>5.2},  Dev Acc: {1:>6.2%}'
    print(msg.format(dev_loss, dev_acc))
    print("Precision, Recall and F1-Score...")
    print(dev_report)
    print("Confusion Matrix...")
    print(dev_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
