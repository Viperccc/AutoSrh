import argparse
import os
import time
from shutil import copyfile
import sys
import math

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import pandas as pd

# from data.dataset import get_cf_dataset, get_ctr_dataset, CTR_Dataset
from utils import seed_everything
from model.nas import Dnis, AdamNas, FM

parser = argparse.ArgumentParser()
# parser.add_argument("--data_path", type=str, default='data/ml-20m/ratings.csv')
parser.add_argument("--data_path", type=str, default='data/ml-1m/')
parser.add_argument("--exp", type=str, default='dnis-compare-NIS')
parser.add_argument("--cuda", nargs='*', type=int, default=[2], help='cuda visible devices')
parser.add_argument("--embedding_dim", type=int, default=160)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr_w", type=float, default=1e-3)
parser.add_argument("--lr_a", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--init_alpha", type=float, default=1.0)
parser.add_argument("--alpha_optim", type=str, default='SGD')
parser.add_argument("--load_checkpoint", type=int, default=0)
parser.add_argument("--warm_start_epochs", type=int, default=0)
parser.add_argument("--num_dim_split", type=int, default=160)
parser.add_argument("--search_space", type=str, default='free')
parser.add_argument("--l1", type=float, default=0)
parser.add_argument("--normalize", type=int, default=0)
parser.add_argument("--use_second_grad", type=int, default=0)
parser.add_argument("--model_name", type=str, default='NeuMF')
parser.add_argument("--alpha_upper_round", type=int, default=0)
parser.add_argument("--feature_split", type=str, default='[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]')
parser.add_argument("--dataset_type", type=str, default='cf')
parser.add_argument("--lr_decay", type=int, default=1)
# parser.add_argument("--alpha_grad_scale", type=str, default='[1,1e3,1e3,1e3,1e3]')
parser.add_argument("--alpha_grad_norm", type=int, default=1)
# args = parser.parse_args("".split())
args = parser.parse_args()
wandb.init(project="DNIS", name=args.exp + '-' + args.model_name + '-' + str(args.embedding_dim))
wandb.config.update(args)
dst = os.path.join(wandb.run.dir, "main.py")
copyfile(sys.argv[0], dst)
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'[1:-1]
device = torch.device('cuda')


def get_single_recall_ndcg_mrr(pred_label_list, limit):
    pred_label_list = sorted(pred_label_list, key=lambda x: x[0])[::-1]
    label_list = [item[1] for item in pred_label_list]
    i = label_list.index(1)
    if i>=limit:
        i=float('inf')
    return int(1 in label_list[:limit]), math.log(2) / math.log(i + 2), 1 / (i + 1)


def get_recall_ndcg_mrr(label, pred, group_id, limit=5):
    id_pred_dict = {}
    for l, p, i in zip(label, pred, group_id):
        if i in id_pred_dict:
            id_pred_dict[i] += [(p, l)]
        else:
            id_pred_dict[i] = [(p, l)]
    recall = 0
    ndcg = 0
    mrr = 0
    for key, value in id_pred_dict.items():
        recall_this, ndcg_this, mrr_this = get_single_recall_ndcg_mrr(value, limit)
        recall += recall_this
        ndcg += ndcg_this
        mrr += mrr_this
    recall /= len(id_pred_dict)
    ndcg /= len(id_pred_dict)
    mrr /= len(id_pred_dict)
    return recall, ndcg, mrr

class CF_Dataset(Dataset):
    def __init__(self, data_dict):
        self.x = torch.LongTensor(np.array(list(data_dict.keys())))
        self.y = torch.Tensor(list(data_dict.values()))

    def __getitem__(self, idx):
        return self.x[idx], torch.ones_like(self.x[idx], dtype=torch.float32), self.y[idx]

    def __len__(self):
        return len(self.x)


def get_dataset(path='data/ml-1m/'):
    test_dict = np.load(path + 'test_dict.npy', allow_pickle='TRUE').item()
    test_dataset = CF_Dataset(test_dict)
    num_features = 9746
    train_df = pd.read_csv(path + 'train.csv')
    train_df['rating'] = 1.0
    return train_df, test_dataset, num_features


def generate_train_val_dataloader(train_df, num_neg=4):
    batch_size=4096

    gb_user = train_df.groupby('userId')
    val_df_split = gb_user.apply(lambda x: x[-len(x)//10+1:])
    train_df_split = gb_user.apply(lambda x: x[:-len(x)//10+1])
    train_dict = {item: 1.0 for item in zip(train_df.userId.values, train_df.movieId.values)}
    train_res = {item: 1.0 for item in zip(train_df_split.userId.values, train_df_split.movieId.values)}
    val_res = {item: 1.0 for item in zip(val_df_split.userId.values, val_df_split.movieId.values)}
    item_ids = train_df['movieId'].unique()
    train_res_items = list(train_res.items())
    for key,value in tqdm(train_res_items):
        uid = key[0]
        for i in range(num_neg):
            iid = np.random.choice(item_ids)
            while (uid, iid) in train_dict:
                iid = np.random.choice(item_ids)
            train_res[(uid, iid)] = 0.0
    train_dataset = CF_Dataset(train_res)
    print(f"size of train_dataset: {len(train_dataset)}")
    # train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_res_items = list(val_res.items())
    for key,value in tqdm(val_res_items):
        uid = key[0]
        for i in range(num_neg):
            iid = np.random.choice(item_ids)
            while (uid, iid) in train_dict:
                iid = np.random.choice(item_ids)
            val_res[(uid, iid)] = 0.0
    val_dataset = CF_Dataset(val_res)
    print(f"size of val_dataset: {len(val_dataset)}")
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_dataloader, val_dataloader

def generate_train_dataloader(train_df, num_neg=4):
    train_dict = {item: 1.0 for item in zip(train_df.userId.values, train_df.movieId.values)}
    train_res = {item: 1.0 for item in zip(train_df.userId.values, train_df.movieId.values)}
    item_ids = train_df['movieId'].unique()
    train_res_items = list(train_res.items())
    for key,value in tqdm(train_res_items):
        uid = key[0]
        for i in range(num_neg):
            iid = np.random.choice(item_ids)
            while (uid, iid) in train_dict:
                iid = np.random.choice(item_ids)
            train_res[(uid, iid)] = 0.0
    train_dataset = CF_Dataset(train_res)
    print(f"size of train_dataset: {len(train_dataset)}")
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_dataloader

class DnisTrain(object):
    def __init__(self, model, dataloaders, lrs, load_checkpoint=True):
        # configure model and optimizers
        if load_checkpoint:
            print("Loading checkpoint...")
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in torch.load(f"checkpoint/{args.exp}_warm_start_5.pth",
                                                           map_location=torch.device('cpu')).items() if
                               (k in model_dict) and ('alpha' not in k)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.module = self.model
        if len(args.cuda) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(len(args.cuda))).cuda()
            self.module = self.model.module
        self.train_df, self.val_dataloader, self.test_dataloader = dataloaders
        lr_w, lr_a = lrs
        # if args.dataset_type == "ctr":
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # else:
        #     self.criterion = torch.nn.MSELoss()
        parameters_w = [parameter for name, parameter in self.module.named_parameters() if 'alpha' not in name]
        self.parameters_w = parameters_w
        self.optimizer_w = torch.optim.Adam(parameters_w, lr=lr_w)
        parameters_embedding = [parameter for name, parameter in self.module.named_parameters() if 'feature' in name]
        self.parameters_embedding = parameters_embedding
        if args.use_second_grad:
            self.optimizer_w_nas = AdamNas(parameters_embedding, lr=lr_w)
        else:
            self.optimizer_w_nas = torch.optim.Adam(parameters_embedding, lr=lr_w)
            # if args.alpha_optim == 'SGD':
        #     self.optimizer_a = torch.optim.SGD([self.module.alpha], lr=lr_a, momentum=0.9, nesterov=True)
        # else:
        self.optimizer_a = getattr(torch.optim, args.alpha_optim)([self.module.alpha], lr=lr_a)
        self.scheduler_w = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_w, 'max', verbose=True,
                                                                      patience=0)
        self.scheduler_w_nas = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_w_nas, 'max', verbose=True,
                                                                          patience=1, threshold=1e-7, min_lr=1e-6)
        # self.scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_a, 'min', verbose=True, patience=0)
        self.criterion_alpha = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4.0]).to(device))
        wandb.watch(self.model, log='all')

    def warm_start(self, num_epochs):
        # during the warm start, all embeddings are used for training
        print("Warm starting...")
        best_loss = -np.inf
        for epoch in range(num_epochs):
            print(f"Starting epoch: {epoch} | phase: train | ⏰: {time.strftime('%H:%M:%S')}")
            self.model.train()
            running_loss = 0
            train_dataloader, val_dataloader = generate_train_val_dataloader(self.train_df)
            # train_dataloader = generate_train_dataloader(self.train_df)
            for itr, batch in enumerate(tqdm(train_dataloader)):
                batch = [item.to(self.device) for item in batch]
                feature_ids, feature_vals, labels = batch
                outputs = self.model(feature_ids, feature_vals).squeeze()
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.optimizer_w.step()
                self.model.zero_grad()
                running_loss += loss.item()
            epoch_loss = running_loss / (itr+1)
            print(f"training loss of epoch {epoch}: {epoch_loss}")
            torch.cuda.empty_cache()
            # val_loss, auc = self.val(train_dataloader)
            val_loss, auc = self.val(self.val_dataloader)
            print(f"val loss of epoch {epoch}: {val_loss}")
            print(f"val auc of epoch {epoch}: {auc}")
            self.scheduler_w.step(auc[-2])
            if auc[-2] > best_loss:
                print("******** New optimal found, saving state ********")
                best_loss = auc[-2]
                best_epoch = epoch
                torch.save(self.module.state_dict(), f"checkpoint/{args.exp}_{args.model_name}_{args.embedding_dim}_warm_start.pth")
            if self.optimizer_w.param_groups[0]['lr'] <= 1e-6:
                print('LR less than 1e-6, stop training...')
                break
        print(f"Best val loss is {best_loss} in {best_epoch}.")

    def train_weights(self, batch):
        self.model.train()
        # update weights and keep gradients of w'/alpha
        feature_ids, feature_vals, labels = batch
        if args.alpha_upper_round:
            alpha_checkpoint = self.module.alpha.data
            self.module.alpha.data = self.module.alpha.data.ceil()
            print(f"alpha before weight training: {self.module.alpha.data}")
        outputs = self.model(feature_ids, feature_vals)
        # loss = self.criterion(outputs.squeeze(), labels.squeeze())
        loss = self.criterion_alpha(outputs.squeeze(), labels.squeeze())
        loss.backward(create_graph=True)
        if args.use_second_grad:
            _, w_prime = self.optimizer_w_nas.step()
        else:
            _, w_prime = self.optimizer_w_nas.step(), None
        if args.alpha_upper_round:
            self.module.alpha.data = alpha_checkpoint
            print(f"alpha after weight training: {self.module.alpha.data}")
        self.module.alpha.grad.data.zero_()
        return w_prime, loss.data.detach().cpu().item()

    def train_alpha(self, batch, w_prime, print_alpha):
        self.model.train()
        feature_ids, feature_vals, labels = batch
        outputs = self.model(feature_ids, feature_vals).squeeze()
        train_loss = self.criterion_alpha(outputs, labels.squeeze())
        # train_loss = self.criterion(outputs, labels.squeeze())
        loss = train_loss
        loss += torch.nn.L1Loss(reduction='sum')(self.module.alpha,
                                                 torch.zeros_like(self.module.alpha)) * args.l1
        # loss = mse_loss + torch.nn.L1Loss(reduction='sum')(self.module.alpha, torch.round(self.module.alpha)) * args.l1
        grad_all = torch.autograd.grad(loss, self.parameters_w + [self.module.alpha])
        # print(f"alpha sum: {self.module.alpha.sum(dim=1)}")
        self.module.alpha.grad.data = grad_all[-1]
        if print_alpha:
            print(f"alpha: {self.module.alpha}")
            print(f"alpha.grad: {self.module.alpha.grad}")
        if args.alpha_grad_norm:
            self.module.alpha.grad.data /= self.module.alpha.grad.data.abs().mean(dim=1, keepdims=True) + 1e-8
        # else:
        #     # scale alpha grad to prevent extremely high frequent features from affecting optimization
        #     self.module.alpha.grad.data *= torch.Tensor(eval(args.alpha_grad_scale)).unsqueeze(1).to(self.device)
        # if print_alpha:
        #     print(f"alpha.grad: {self.module.alpha.grad}")
        if args.use_second_grad:
            p_grad = [t.grad for t in self.parameters_w]
            # Jacobi matrix vector product
            w_prime_grad_p_grad = torch.autograd.grad(
                w_prime, p_grad, grad_all[:-1], allow_unused=True)
            # replace nan values with zero
            for t in w_prime_grad_p_grad:
                t[t != t] = 0
            second_order_grad = torch.autograd.grad(
                p_grad, self.module.alpha, w_prime_grad_p_grad, allow_unused=True)[0]
            # print(f"second_order_grad now: {second_order_grad}")
            self.module.alpha.grad.data += second_order_grad
        self.optimizer_a.step()
        self.model.zero_grad()
        return train_loss.data.detach().cpu().item()

    def train(self, num_epochs, warm_start_epochs=1):
        if warm_start_epochs > 0:
            print(f"Use warm start for {warm_start_epochs} epochs...")
            self.warm_start(warm_start_epochs)
        checkpoint = torch.load(f"checkpoint/{args.exp}_{args.model_name}_{args.embedding_dim}_warm_start.pth",
                                map_location=torch.device('cpu'))
        self.module.load_state_dict(checkpoint)
        test_loss, auc = self.val(self.test_dataloader)
        print(f"Test loss: {test_loss}.")
        print(f"Test auc: {auc}.")
        best_val_loss = -np.inf
        self.module.alpha.data = (torch.ones_like(self.module.alpha) * args.init_alpha).to(self.module.alpha)
        print(f"alpha: {self.module.alpha}")
        for epoch in range(num_epochs):
            running_train_loss = 0
            running_val_loss = 0
            wandb.log({"w learning rate": self.optimizer_w_nas.param_groups[0]['lr']}, step=epoch)
            print(f"Start training epoch {epoch}")
            # if epoch==0:
            train_dataloader, val_dataloader = generate_train_val_dataloader(self.train_df)
            # else:
            #     train_dataloader, _ = generate_train_val_dataloader(self.train_df)
            # val_dataloader = generate_train_dataloader(self.train_df)
            for itr, [batch_train, batch_val] in tqdm(enumerate(zip(train_dataloader, val_dataloader))):
                print_alpha = ((itr % 20) == 0)
                batch_train = [item.to(self.device) for item in batch_train]
                w_prime, train_loss = self.train_weights(batch_train)
                print(f"train loss: {train_loss}")
                batch_val = [item.to(self.device) for item in batch_val]
                val_loss = self.train_alpha(batch_val, w_prime, print_alpha=print_alpha)
                if args.search_space == 'feature_block':
                    self.module.alpha.data.clamp_(0., self.module.num_feature_split)
                elif args.search_space == 'embedding_block':
                    indices = (self.module.alpha.data.ceil() - self.module.alpha.data) < 0.1
                    self.module.alpha.data[indices] = self.module.alpha.data[indices].ceil() + 1e-3
                    self.module.alpha.data.clamp_(1., self.module.num_dim_split)
                elif args.search_space == 'free':
                    self.module.alpha.data.clamp_(0., 1.)
                else:
                    raise NotImplementedError
                print(f"val loss: {val_loss}")
                running_train_loss += train_loss
                running_val_loss += val_loss
            running_train_loss /= (itr + 1)
            running_val_loss /= (itr + 1)
            print(f"train loss of epoch {epoch}: {running_train_loss}")
            wandb.log({"train_epoch_loss": running_train_loss}, step=epoch)
            print(f"running val loss of epoch {epoch}: {running_val_loss}")
            val_epoch_loss, auc = self.val(self.val_dataloader,self.criterion_alpha)
            print(f"val_epoch_loss of epoch {epoch}: {val_epoch_loss}")
            print(f"metrics of epoch {epoch}: {auc}")
            wandb.log({"val_epoch_loss": val_epoch_loss, "val_epoch_auc": auc}, step=epoch)
            wandb.log({"alpha": self.module.alpha.data.detach().cpu()}, step=epoch)
            if  auc[-2] > best_val_loss and epoch>5:
                best_val_loss = auc[-2]
                best_epoch=epoch
                print(f"Best val loss {best_val_loss} achieved in epoch {best_epoch}. Start testing...")
                test_loss, test_auc = self.val(self.test_dataloader)
                print(f"Test loss: {test_loss}.")
                print(f"Test auc: {test_auc}.")
                wandb.log({"test_loss": test_loss, "test_auc": test_auc}, step=epoch)
                # print(self.module.state_dict())nn:q；：:
                torch.save(self.module.state_dict(), os.path.join(wandb.run.dir, f"{args.exp}.tar"))
            torch.save(self.module.state_dict(), f"checkpoint/{args.exp}_{args.model_name}_{args.embedding_dim}_{epoch}.pth")
            if args.lr_decay:
                self.scheduler_w_nas.step(-running_val_loss, epoch=epoch)
            # self.scheduler_a.step(running_val_loss, epoch=epoch)
            if self.optimizer_w_nas.param_groups[0]['lr'] <= 1e-5:
                print('LR less than 1e-5, stop training...')
                print(f"Best val loss {best_val_loss} achieved in epoch {best_epoch}.")
                print(f"Test loss: {test_loss}.")
                print(f"Test auc: {test_auc}.")
                break

    # def test(self):
    #     self.model.eval()
    #     running_loss = 0
    #     with torch.no_grad():
    #         for itr, batch_test in enumerate(self.test_dataloader):
    #             batch_test = [item.to(self.device) for item in batch_test]
    #             feature_ids, feature_vals, labels = batch_test
    #             outputs = self.model(feature_ids, feature_vals)
    #             loss = self.criterion(outputs.squeeze(), labels.squeeze())
    #             running_loss += loss.data.detach().cpu().item()
    #         test_loss = running_loss / (itr + 1)
    #     torch.cuda.empty_cache()
    #     print(f"Test loss: {test_loss}.")
    #     return test_loss

    def val(self, dataloader, criterion=None):
        self.model.eval()
        running_loss = 0
        pred_arr = np.array([])
        label_arr = np.array([])
        group_id_arr = np.array([])
        with torch.no_grad():
            for itr, batch in tqdm(enumerate(dataloader)):
                batch = [item.to(device) for item in batch]
                feature_ids, feature_vals, labels = batch
                group_id_arr = np.hstack(
                    [group_id_arr, feature_ids[:, 0].data.detach().cpu()]) if group_id_arr.size else feature_ids[:,
                                                                                                     0].data.detach().cpu()
                outputs = self.model(feature_ids, feature_vals)
                if criterion is None:
                    criterion = self.criterion
                loss = criterion(outputs.squeeze(), labels.squeeze())
                running_loss += loss.data.detach().cpu().item()
                pred_arr = np.hstack(
                    [pred_arr, outputs.data.detach().cpu()]) if pred_arr.size else outputs.data.detach().cpu()
                label_arr = np.hstack(
                    [label_arr, labels.data.detach().cpu()]) if label_arr.size else labels.data.detach().cpu()
            val_loss = running_loss / (itr + 1)
            torch.cuda.empty_cache()
        recall_1,ndcg_1,mrr_1 = get_recall_ndcg_mrr(label_arr, pred_arr, group_id_arr, limit=1)
        recall_5,ndcg_5,mrr_5 = get_recall_ndcg_mrr(label_arr, pred_arr, group_id_arr, limit=5)
        recall_10,ndcg_10,mrr_10 = get_recall_ndcg_mrr(label_arr, pred_arr, group_id_arr, limit=10)
        return val_loss, [recall_1,ndcg_1,mrr_1,recall_5,ndcg_5,mrr_5,recall_10,ndcg_10,mrr_10]


def main():
    train_df, test_dataset, num_features = get_dataset()
    num_fields = 2
    val_dataset = test_dataset
    batch_size = args.batch_size
    # train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dnis = Dnis(num_features, args.embedding_dim, num_fields=num_fields, num_dim_split=args.num_dim_split,
                search_space=args.search_space, normalize=args.normalize, model_name=args.model_name,
                feature_split=eval(args.feature_split))
    dnis_train = DnisTrain(dnis, [train_df, val_dataloader, test_dataloader], [args.lr_w, args.lr_a],
                           load_checkpoint=args.load_checkpoint)
    dnis_train.train(args.num_epochs, args.warm_start_epochs)


if __name__ == '__main__':
    seed_everything()
    start_time = time.time()
    main()
    end_time = time.time()
    print("{} minutes used for {}.".format((end_time - start_time) / 60, args.model_name))
