import os

import csv
from matplotlib import pyplot as plt
import nni
from nni.utils import merge_parameter
import time
import json
import math
import copy
import argparse
import warnings
import numpy as np

import pandas as pd
from sklearn import metrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from models import *
from dataloader import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size,
          optimizer, loss_fn, epoch):

    f1_sum = 0.0
    loss_sum = 0.0
    preds_, labels_ = [], []

    batch_num = math.ceil(len(index) / batch_size)
    random.shuffle(index)

    model.train()

    for batch in range(batch_num):
        if batch == batch_num - 1:
            train_idx = index[batch * batch_size:]
        else:
            train_idx = index[batch * batch_size:(batch + 1) * batch_size]

        output, cl_loss = model(ppi_g, prot_embed, ppi_list, train_idx,
                                True)
        loss = loss_fn(output, labels[train_idx]) + cl_loss * param['cl_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        f1_score = evaluat_metrics(output.detach().cpu(),
                                   labels[train_idx].detach().cpu())
        f1_sum += f1_score

        preds_.append((output.sigmoid() > 0.5).cpu().numpy())
        labels_.append(labels[train_idx].cpu().numpy())

    if preds_ != []:
        preds_ = np.concatenate(preds_)
        labels_ = np.concatenate(labels_)
    labels_ = np.array(labels_)
    preds_ = np.array(preds_)
    return loss_sum / batch_num, f1_sum / batch_num, labels_, preds_


def evaluator(model,
              ppi_g,
              prot_embed,
              ppi_list,
              labels,
              index,
              batch_size,
              mode='metric'):

    eval_output_list = []
    eval_labels_list = []
    preds_, labels_ = [], []

    batch_num = math.ceil(len(index) / batch_size)

    model.eval()

    with torch.no_grad():
        for batch in range(batch_num):
            if batch == batch_num - 1:
                eval_idx = index[batch * batch_size:]
            else:
                eval_idx = index[batch * batch_size:(batch + 1) * batch_size]

            output, _ = model(ppi_g, prot_embed, ppi_list, eval_idx, False)
            eval_output_list.append(output.detach().cpu())
            eval_labels_list.append(labels[eval_idx].detach().cpu())
            #
            preds_.append((output.sigmoid() > 0.5).cpu().numpy())
            labels_.append(labels[eval_idx].cpu().numpy())

        if preds_ != []:
            preds_ = np.concatenate(preds_)
            labels_ = np.concatenate(labels_)
        labels_ = np.array(labels_)
        preds_ = np.array(preds_)

        f1_score = evaluat_metrics(torch.cat(eval_output_list, dim=0),
                                   torch.cat(eval_labels_list, dim=0))

    if mode == 'metric':
        return f1_score, labels_, preds_
    elif mode == 'output':
        return torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list,
                                                             dim=0)


def rec_train(param):

    if args.pre_train is None:
        protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(
            param['dataset'], param['split_mode'], param['seed'])
    else:
        protein_data = load_pretrain_data(args.pre_train)

    output_dir = "../results/{}/{}/REC/".format(param['dataset'], timestamp)
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')
    with open(os.path.join(output_dir, "config.json"), 'a+') as tf:
        json.dump(param, tf, indent=2)

    rec_dataloader = DataLoader(protein_data,
                                batch_size=param['rec_batch_size'],
                                shuffle=True,
                                collate_fn=collate)
    rec_model = RecNet(
        param,
        DataLoader(protein_data,
                   batch_size=param['rec_batch_size'],
                   shuffle=False,
                   collate_fn=collate)).to(device)
    rec_optimizer = torch.optim.Adam(rec_model.parameters(),
                                     lr=float(param['learning_rate']),
                                     weight_decay=float(param['weight_decay']))

    for epoch in range(1, param["rec_epoch"] + 1):
        for iter_num, batch_graph in enumerate(rec_dataloader):

            batch_graph.to(device)

            _, recon_loss, mask_recon_loss = rec_model(batch_graph)
            loss_rec = recon_loss + mask_recon_loss * param['mask_recon_loss']

            rec_optimizer.zero_grad()
            loss_rec.backward()
            rec_optimizer.step()

            if (epoch - 1) % param['log_num'] == 0 and iter_num == 0:
                print(
                    "\033[0;30;43m rec-training | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f}\033[0m"
                    .format(epoch, iter_num, loss_rec.item(),
                            recon_loss.item(), mask_recon_loss.item()))
                log_file.write(
                    "rec-training | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f}\n"
                    .format(epoch, iter_num, loss_rec.item(),
                            recon_loss.item(), mask_recon_loss.item()))
                log_file.flush()

    torch.save(
        rec_model.state_dict(),
        os.path.join(output_dir, "rec_model_{}.ckpt".format(param['dataset'])))

    del rec_model
    torch.cuda.empty_cache()


def main(param):

    protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(
        param['dataset'], param['split_mode'], param['seed'])

    rec_model = RecNet(
        param,
        DataLoader(protein_data,
                   batch_size=param['rec_batch_size'],
                   shuffle=False,
                   collate_fn=collate)).to(device)
    if args.ckpt_path is None:
        rec_model.load_state_dict(
            torch.load("../results/{}/{}/REC/rec_model_{}.ckpt".format(
                param['dataset'], timestamp, param['dataset'])))
    else:
        rec_model.load_state_dict(torch.load(args.ckpt_path))
    prot_embed = rec_model.Encoder.forward().to(
        device)

    del rec_model
    torch.cuda.empty_cache()

    output_dir = "../results/{}/{}/SEES_{}/".format(param['dataset'],
                                                    timestamp, param['seed'])

    check_writable(output_dir, overwrite=False)
    log_file = open(
        os.path.join(output_dir,
                     "train_log_{}.txt".format(param['split_mode'])), 'a+')

    model = ProtIR(param).to(device)  # GIN
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(param['learning_rate']),
                                 weight_decay=float(param['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=10,
                                                           verbose=True)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    best_epoch = 0

    for epoch in range(1, param["max_epoch"] + 1):

        train_loss, train_f1_score, _, _ = train(model, ppi_g, prot_embed,
                                                 ppi_list, labels,
                                                 ppi_split_dict['train_index'],
                                                 param['batch_size'],
                                                 optimizer, loss_fn, epoch)

        scheduler.step(train_loss)

        if (epoch - 1) % param['log_num'] == 0:

            val_f1_score, _, _ = evaluator(model, ppi_g, prot_embed, ppi_list,
                                           labels, ppi_split_dict['val_index'],
                                           param['batch_size'])
            test_f1_score, _, _ = evaluator(model, ppi_g, prot_embed, ppi_list,
                                            labels,
                                            ppi_split_dict['test_index'],
                                            param['batch_size'])

            if test_f1_score > test_best:
                test_best = test_f1_score

            if val_f1_score >= val_best:
                val_best = val_f1_score
                test_val = test_f1_score
                state = copy.deepcopy(model.state_dict())
                es = 0
                best_epoch = epoch
            else:
                es += 1

            print(
                "\033[0;30;46m Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\033[0m"
                .format(epoch, train_loss, train_f1_score, val_f1_score,
                        test_f1_score, val_best, test_val, test_best,
                        best_epoch))
            
            log_file.write(
                " Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\n"
                .format(epoch, train_loss, train_f1_score, val_f1_score,
                        test_f1_score, val_best, test_val, test_best,
                        best_epoch))
            log_file.flush()

            if es == 1000:
                print("Early stopping!")
                break
                
    torch.save(
        state,
        os.path.join(
            output_dir, "model_{}_{}.pth".format(param['dataset'],
                                                 param['split_mode'])))
    log_file.close()

    model.load_state_dict(state)
    eval_output, eval_labels = evaluator(model, ppi_g, prot_embed, ppi_list,
                                         labels, ppi_split_dict['test_index'],
                                         param['batch_size'], 'output')
    np.save(
        os.path.join(output_dir,
                     "eval_output_{}.npy".format(param['split_mode'])),
        eval_output.detach().cpu().numpy())
    np.save(
        os.path.join(output_dir,
                     "eval_labels_{}.npy".format(param['split_mode'])),
        eval_labels.detach().cpu().numpy())

    return test_f1_score, test_val, test_best, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str,
                        default="SHS27k") 
    parser.add_argument("--split_mode", type=str,
                        default="bfs") 
    parser.add_argument("--input_dim", type=int, default=7)
    parser.add_argument("--output_dim", type=int, default=7)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--prot_hidden_dim", type=int, default=512)
    parser.add_argument("--resid_hidden_dim", type=int, default=128)
    parser.add_argument("--prot_num_layers", type=int, default=2)
    parser.add_argument("--resid_num_layers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--rec_batch_size", type=int, default=64)
    parser.add_argument("--dropout_ratio", type=float, default=0.2)

    parser.add_argument("--rec_epoch", type=int, default=50)
    parser.add_argument("--pert_ratio", type=float, default=0.25)
    parser.add_argument("--rec_mask_ratio", type=float, default=0.15)
    parser.add_argument("--sce_scale", type=float, default=1.5)
    parser.add_argument("--cl_loss", type=float, default=1.0)
    parser.add_argument("--mask_recon_loss", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_num", type=int, default=1)
    parser.add_argument("--data_mode", type=int, default=0)
    parser.add_argument("--data_split_mode", type=int, default=0)
    parser.add_argument("--pre_train", type=str, default=None)

    parser.add_argument("--ckpt_path", type=str, default=None)

    args = parser.parse_args()
    param = args.__dict__

    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % (
        (time.time() - int(time.time())) * 1000)

    if os.path.exists("../configs/param_configs.json"):
        param = json.loads(
            open("../configs/param_configs.json",
                 'r').read())[param['dataset']][param['split_mode']]

    if param['data_mode'] == 0:
        param['dataset'] = 'SHS27k'
    elif param['data_mode'] == 1:
        param['dataset'] = 'SHS148k'
    elif param['data_mode'] == 2:
        param['dataset'] = 'STRING'

    if param['data_split_mode'] == 0:
        param['split_mode'] = 'random'
    elif param['data_split_mode'] == 1:
        param['split_mode'] = 'bfs'
    elif param['data_split_mode'] == 2:
        param['split_mode'] = 'dfs'

    param.update(nni.get_next_parameter())

    set_seed(param['seed'])
    print(param)

    if args.ckpt_path is None:
        rec_train(param)
    test_f1_score, test_val, test_best, best_epoch = main(param)
    nni.report_final_result(test_val)

    outFile = open('../PerformMetrics_Metrics.csv', 'a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [timestamp]
    for v, k in param.items():
        results.append(k)

    results.append(str(test_f1_score))
    results.append(str(test_val))
    results.append(str(test_best))
    writer.writerow(results)
