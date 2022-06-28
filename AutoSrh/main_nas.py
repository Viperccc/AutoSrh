import argparse
import os
import time
from shutil import copyfile
import sys

import torch
import wandb
import numpy as np
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from data.dataset import get_cf_dataset, get_ctr_dataset, CTR_Dataset ,get_avazu_dataset,Avazu_Dataset
from utils import seed_everything
from model.nas import Dnis, AdamNas, FM

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='data/ml-20m/ratings.csv')
#parser.add_argument("--data_path", type=str, default='data/criteo/datasets.pickle')
#parser.add_argument("--data_path", type=str, default='data/avazu/dataset.pickle')
#parser.add_argument("--data_path", type=str, default='data/avazu/click.pickle')
parser.add_argument("--exp", type=str, default='DNIS-CF-10split')
parser.add_argument("--cuda", nargs='*', type=int, default=[2], help='cuda visible devices')
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr_w", type=float, default=1e-2)
parser.add_argument("--lr_a", type=float, default=1e-3)
#parser.add_argument("--num_epochs", type=int, default=100)
#for avazu
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--init_alpha", type=float, default=1.0)
parser.add_argument("--alpha_optim", type=str, default='SGD')
parser.add_argument("--load_checkpoint", type=int, default=0)
parser.add_argument("--warm_start_epochs", type=int, default=20)
parser.add_argument("--num_dim_split", type=int, default=64)
parser.add_argument("--search_space", type=str, default='feature_block')
parser.add_argument("--l1", type=float, default=0.00001)
parser.add_argument("--normalize", type=int, default=0)
parser.add_argument("--use_second_grad", type=int, default=0)
parser.add_argument("--model_name", type=str, default='FM')
parser.add_argument("--alpha_upper_round", type=int, default=0)
parser.add_argument("--feature_split", type=str, default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')
parser.add_argument("--dataset_type", type=str, default='ctr')
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
        if hasattr(torch.cuda, 'empty_cache'):
           torch.cuda.empty_cache()
        #self.model=nn.DataParallel(model,device_ids=[0,1])
        self.model = model.to(self.device)
        self.module = self.model
        if len(args.cuda) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(len(args.cuda))).cuda()
            self.module = self.model.module
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        lr_w, lr_a = lrs
        if args.dataset_type == "ctr":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.MSELoss()
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
        self.scheduler_w = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_w, 'min', verbose=True,
                                                                      patience=0)
        self.scheduler_w_nas = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_w_nas, 'min', verbose=True,
                                                                          patience=1, threshold=1e-7, min_lr=1e-6)
        # self.scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_a, 'min', verbose=True, patience=0)
        wandb.watch(self.model, log='all')

    def warm_start(self, num_epochs):
        # during the warm start, all embeddings are used for training
        print("Warm starting...")
        best_loss = np.inf
        for epoch in range(num_epochs):
            print(f"Starting epoch: {epoch} | phase: train | ⏰: {time.strftime('%H:%M:%S')}")
            self.model.train()
            running_loss = 0
            # if epoch == 8:
            #     self.optimizer_w.param_groups[0]['lr'] *= 0.1
            for itr, batch in tqdm(enumerate(self.train_dataloader)):
                batch = [item.to(self.device) for item in batch]
                feature_ids, feature_vals, labels = batch
                outputs = self.model(feature_ids, feature_vals).squeeze()
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.optimizer_w.step()
                self.model.zero_grad()
                running_loss += loss.item()
            epoch_loss = running_loss / itr
            print(f"training loss of epoch {epoch}: {epoch_loss}")
            torch.cuda.empty_cache()
            val_loss, auc = self.val(self.val_dataloader)
            print(f"val loss of epoch {epoch}: {val_loss}")
            print(f"val auc of epoch {epoch}: {auc}")
            self.scheduler_w.step(val_loss)
            if val_loss < best_loss:
                print("******** New optimal found, saving state ********")
                best_loss = val_loss
                best_epoch = epoch
                torch.save(self.module.state_dict(), f"checkpoint/{args.exp}_{args.model_name}_warm_start-64-L1-0.00001-deepfm-ctr.pth")
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
        loss = self.criterion(outputs.squeeze(), labels.squeeze())
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
        mse_loss = self.criterion(outputs, labels.squeeze())
        loss = mse_loss
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
            print(self.module.alpha.grad.data.abs().mean(dim=-1, keepdims=True))
            self.module.alpha.grad.data /= self.module.alpha.grad.data.abs().mean(dim=-1, keepdims=True)+1e-8
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
        return mse_loss.data.detach().cpu().item()

    def train(self, num_epochs, warm_start_epochs=1):
        if warm_start_epochs > 0:
            print(f"Use warm start for {warm_start_epochs} epochs...")
            self.warm_start(warm_start_epochs)
            checkpoint = torch.load(f"checkpoint/{args.exp}_{args.model_name}_warm_start-64-L1-0.00001-deepfm-ctr.pth",
                                    map_location=torch.device('cpu'))
            self.module.load_state_dict(checkpoint)
        test_loss, auc = self.val(self.test_dataloader)
        print(f"Test loss: {test_loss}.")
        print(f"Test auc: {auc}.")
        best_val_loss = np.inf
        self.module.alpha.data = (torch.ones_like(self.module.alpha) * args.init_alpha).to(self.module.alpha)
        print(f"alpha: {self.module.alpha}")
        for epoch in range(num_epochs):
            running_train_loss = 0
            running_val_loss = 0
            wandb.log({"w learning rate": self.optimizer_w_nas.param_groups[0]['lr']}, step=epoch)
            print(f"Start training epoch {epoch}")
            for itr, [batch_train, batch_val] in tqdm(enumerate(zip(self.train_dataloader, self.val_dataloader))):
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
            val_epoch_loss, auc = self.val(self.val_dataloader)
            wandb.log({"val_epoch_loss": val_epoch_loss, "val_epoch_auc": auc}, step=epoch)
            wandb.log({"alpha": self.module.alpha.data.detach().cpu()}, step=epoch)
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                print(f"Best val loss {best_val_loss} achieved in epoch {epoch}. Start testing...")
                test_loss, auc = self.val(self.test_dataloader)
                print(f"Test loss: {test_loss}.")
                print(f"Test auc: {auc}.")
                wandb.log({"test_loss": test_loss, "test_auc":auc}, step=epoch)
                # print(self.module.state_dict())nn:q；：:
                torch.save(self.module.state_dict(), os.path.join(wandb.run.dir, f"{args.exp}.tar"))
            if args.lr_decay:
                self.scheduler_w_nas.step(val_epoch_loss, epoch=epoch)
            # self.scheduler_a.step(running_val_loss, epoch=epoch)
            
            #if self.optimizer_w_nas.param_groups[0]['lr'] <= 1e-6:
            #for auazu
            if self.optimizer_w_nas.param_groups[0]['lr'] <= 1e-6:
                print('LR less than 1e-5, stop training...')
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

    def val(self, dataloader):
        self.model.eval()
        running_loss = 0
        pred_arr = np.array([])
        label_arr = np.array([])
        with torch.no_grad():
            for itr, batch in tqdm(enumerate(dataloader)):
                batch = [item.to(device) for item in batch]
                feature_ids, feature_vals, labels = batch
                outputs = self.model(feature_ids, feature_vals)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                running_loss += loss.data.detach().cpu().item()
                pred_arr = np.hstack(
                    [pred_arr, outputs.data.detach().cpu()]) if pred_arr.size else outputs.data.detach().cpu()
                label_arr = np.hstack(
                    [label_arr, labels.data.detach().cpu()]) if label_arr.size else labels.data.detach().cpu()
            val_loss = running_loss / (itr + 1)
            torch.cuda.empty_cache()
        if args.dataset_type == "ctr":
            auc = roc_auc_score(label_arr, pred_arr)
            return val_loss, auc
        if args.dataset_type == "ava":
            auc = roc_auc_score(label_arr, pred_arr)
            return val_loss, auc
        return val_loss, 0


def main():
    if args.dataset_type == "ctr":
        train_dataset, val_dataset, test_dataset, num_features = get_ctr_dataset(args.data_path)
        val_dataset = val_dataset[0]
        test_dataset = test_dataset[0]
        num_fields = 39
    #else: 
        train_dataset, val_dataset, test_dataset, num_features = get_avazu_dataset(args.data_path)
        num_fields = 23   
    else:
        train_dataset, val_dataset, test_dataset, num_features = get_cf_dataset(args.data_path)
        num_fields = 2
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(num_features)
    dnis = Dnis(num_features, args.embedding_dim, num_fields=num_fields, num_dim_split=args.num_dim_split,
                search_space=args.search_space, normalize=args.normalize, model_name=args.model_name,
                feature_split=eval(args.feature_split))
    dnis_train = DnisTrain(dnis, [train_dataloader, val_dataloader, test_dataloader], [args.lr_w, args.lr_a],
                           load_checkpoint=args.load_checkpoint)
    dnis_train.train(args.num_epochs, args.warm_start_epochs)


if __name__ == '__main__':
    seed_everything()
    start_time = time.time()
    main()
    end_time = time.time()
    print("{} minutes used for {}.".format((end_time - start_time) / 60, args.model_name))
