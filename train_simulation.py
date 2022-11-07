
import os
import time
import sys
from math import inf
from copy import deepcopy
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn.functional as F

from utils import total_loss, metric_calculate
from model import HiRAND
import utils



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='Disables CUDA training.'
    )
    parser.add_argument(
        '--fastmode', action='store_true', default=False,
        help='Validate during training pass.'
    )
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    parser.add_argument(
        '--epochs', type=int, default=150,  # 150
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.8,
        help='Weight decay (L2 loss on parameters).'
    )
    parser.add_argument(
        '--hidden', type=int, default=32,
        help='Number of hidden units.'
    )
    parser.add_argument(
        '--input_droprate', type=float, default=0.5,
        help='Dropout rate of the input layer (1 - keep probability).'
    )
    parser.add_argument(
        '--hidden_droprate', type=float, default=0.5,
        help='Dropout rate of the hidden layer (1 - keep probability).'
    )
    parser.add_argument(
        '--dropnode_rate', type=float, default= 0.5,
        help='Dropnode rate (1 - keep probability).'
    )
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument(
        '--order', type=int, default=3 ,help='Propagation step'
    )
    parser.add_argument(
        '--sample', type=int, default=4, help='Sampling times of dropnode'
    )
    parser.add_argument(
        '--tem', type=float, default=0.01, help='Sharpening temperature'
    )
    parser.add_argument('--lam', type=float, default=2., help='Lamda')
    parser.add_argument(
        '--cuda_device', type=int, default=1, help='Cuda device'
    )
    parser.add_argument(
        '--use_bn', action='store_true', default=True,
        help='Using Batch Normalization'
    )
    parser.add_argument('--gamma_c',type=int, default=50, help='Gamma_c')
    parser.add_argument(
        '--test_portions',type=float, default=0.2, help='Test_portions'
    )

    parser.add_argument(
        "--early_stop", action="store_true", default=True
    )
    args = parser.parse_args()

    device = torch.device("cuda:0") 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #### setting the save folder
    start = time.time()
    save_name = time.strftime("%y-%m-%d-%H-%M")
    save_folder = os.path.join("./results", save_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # setting the number of labeled data
    train_portions = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,60,90,120,150,180,210]
    train_portions = [2]
    
    ### setting the result format
    # training result
    hist = {"re": [], "epoch": []}   
    for phase in ["train", "valid"]:
        for metric in ["loss", "auc", "acc", "f1"]:
            hist["%s_%s" % (phase, metric)] = []
    # testing result
    hist_te = {"re": [],"portion": []}
    for metric in ["loss", "auc", "acc", "f1"]:
        hist_te[metric] = []

    ### 5-folder training for simulation data
    it = 1
    positive = np.loadtxt('./simulation/generate_gedfn/gedfn%d_position.txt' % it)
    positive = list(map(int, positive))
    gene_A = np.loadtxt('./simulation/generate_gedfn/gedfn%d_gene_A.txt' % it)
    np.fill_diagonal(gene_A, 1)
    x = np.loadtxt('./simulation/generate_gedfn/gedfn%d_x.txt' % it)
    y = np.loadtxt('./simulation/generate_gedfn/gedfn%d_y.txt' % it)

    # input data pre-precessing
    x_cp = x.copy()
    y_cp = y.copy()
    num_cls = int(y.max().item() + 1)
    A = utils.cal_A(x)
    A = A.to(device)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    gene_A = torch.tensor(gene_A).to(device).float()

    ###  feature importance
    gamma_numerator = gene_A.sum(dim=0)
    gamma_denominator = gene_A.sum(dim=0)
    gamma_numerator[torch.where(gamma_numerator > args.gamma_c)] = args.gamma_c

    for train_portion in train_portions:
        spliter = StratifiedShuffleSplit(5, test_size=args.test_portions, random_state=args.seed
        )
        for re, (train_idx_val, idx_test) in enumerate(
            spliter.split(x_cp, y_cp)
        ):
            idx_train, idx_val = train_test_split(
                train_idx_val,
                train_size=train_portion,
                random_state=args.seed, shuffle=True, stratify=y_cp[train_idx_val]
            )
            print(
                're: {0},train size: {1}, valid_size:{2}, test size: {3}'.format(
                    re, idx_train.shape[0], idx_val.shape[0], idx_test.shape[0]
                )
            )

            # model construction
            model = HiRAND(nfeat=x.shape[1],
                            nhid=args.hidden,
                            nclass=num_cls,
                            input_droprate=args.input_droprate,
                            hidden_droprate=args.hidden_droprate,
                            use_bn=args.use_bn,
                            drop_rate=args.dropnode_rate,
                            order=args.order,
                            K=args.sample)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

            # put the model and data into cuda
            model.cuda()
            idx_train = torch.tensor(idx_train, dtype=torch.long, device=device)
            idx_val = torch.tensor(idx_val, dtype=torch.long, device=device)
            idx_test = torch.tensor(idx_test, dtype=torch.long, device=device)
            


            # training
            for epoch in range(args.epochs):
                # Training
                model.train()
                with torch.enable_grad():
                    output_list = model(x, gene_A, A)
                    loss_train = total_loss(
                        output_list, y, idx_train, args.tem, args.lam)					
                    optimizer.zero_grad()  
                    loss_train.backward()
                    optimizer.step()
                train_metrics = metric_calculate(
                    output_list[0][idx_train],
                    y[idx_train]
                )
                train_metrics["loss"] = loss_train.item()
                    
                # Validation
                model.eval()
                output = model(x, gene_A, A)
                with torch.no_grad():
                    loss_val = F.nll_loss(
                        output[idx_val], y[idx_val].long()
                    ) 
                valid_metrics = metric_calculate(
                    output[idx_val], y[idx_val]
                )
                valid_metrics["loss"] = loss_val.item()

                print('Epoch: {:04d}'.format(epoch+1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'auc_train: {:.4f}'.format(train_metrics["auc"]),
                        'acc_train: {:.4f}'.format(train_metrics["f1"]),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'auc_val: {:.4f}'.format(valid_metrics["auc"]),
                        )

                hist["re"].append(re)
                hist["epoch"].append(epoch)
                for phase, metricss in zip(
                    ["train", "valid"], [train_metrics, valid_metrics]
                ):
                    for k, v in metricss.items():
                        hist["%s_%s" % (phase, k)].append(v)
                
                
                # early stopping
                bad_counter = 0
                loss_best = np.inf
                auc_best = 0.0
                loss_mn = np.inf
                auc_mx = 0.0
                best_epoch = 0
                best_model = deepcopy(model.state_dict())
                if args.early_stop:
                    es_metrics = train_metrics
                    if (
                        es_metrics["loss"] <= loss_mn or
                        es_metrics["auc"] >= auc_mx
                    ):
                        if es_metrics["loss"] <= loss_best:
                            loss_best = es_metrics["loss"]
                            auc_best = es_metrics["auc"]
                            best_epoch = epoch
                            best_model = deepcopy(model.state_dict())
                        loss_mn = np.min((es_metrics["loss"], loss_mn))
                        acc_mx = np.max((es_metrics["auc"], auc_mx))
                        bad_counter = 0
                    else:
                        bad_counter += 1
                    if bad_counter == args.patience:
                        print(
                            'Early stop! Min loss: ', loss_mn,
                            ', Max accuracy: ', acc_mx
                        )
                        break
            if args.early_stop:
                model.load_state_dict(best_model)

            ##feature importance value computing
            var_left = torch.sum(torch.abs(model.sgcn.weight * gene_A), 0)
            var_left_mean = var_left / var_left.sum()
            var_right = torch.sum(torch.abs(model.fgcn.layer1.weight), 0)
            var_right_mean = var_right / var_right.sum()
            var_importance = (
                var_left * gamma_numerator
            ) * (1.0 / gamma_denominator) + var_right
            var_importance_mean = var_left_mean + var_right_mean
            # var_importance_mean_all += var_importance_mean
            var_importances_dict = {
                "var_left": var_left.detach().cpu().numpy(),
                "var_left_mean": var_left_mean.detach().cpu().numpy(),
                "var_right": var_right.detach().cpu().numpy(),
                "var_right_mean": var_right_mean.detach().cpu().numpy(),
                "var_importance": var_importance.detach().cpu().numpy(),
                "var_importance_mean": var_importance_mean.detach().cpu().numpy(),
            }
            var_importances = pd.DataFrame(var_importances_dict)
            var_importances.to_csv(os.path.join(save_folder, "VI_re%d.csv" % re))

            #Testing
            model.eval()
            with torch.no_grad():
                output = model(x, gene_A, A)
                loss_test = F.nll_loss(output[idx_test], y[idx_test].long())
            test_metrics = metric_calculate(output[idx_test], y[idx_test])
            test_metrics["loss"] = loss_test.item()
            print('re: {:04d}'.format(re+1),
                    'loss_test: {:.4f}'.format(loss_test.item()),
                    'auc_test: {:.4f}'.format(test_metrics["auc"]),
                    )
            hist_te["re"].append(re)
            hist_te["portion"].append(train_portion)
            for k, v in test_metrics.items():
                hist_te[k].append(v)

            print("-" * 100)
            print()

	# saving result
    hist = pd.DataFrame(hist)
    hist_te = pd.DataFrame(hist_te)
    print(hist_te.apply(lambda x: "%.3f + %.3f" % (np.mean(x), np.std(x))))
    hist.to_csv(os.path.join(save_folder, "relu_hist.csv"))
    hist_te.to_csv(os.path.join(save_folder, "relu_hist_te.csv"))

    end = time.time()
    print("Time:, ", end - start)

if __name__ == "__main__":
    main()

