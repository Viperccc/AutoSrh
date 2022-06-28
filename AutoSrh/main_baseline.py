import argparse
import torch
import wandb
import os
import time
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from apex import amp
from data.dataset import get_cf_dataset, get_ctr_dataset, CTR_Dataset
from utils import seed_everything
from model import base_model
from notification import WeChatPub
from shutil import copyfile

parser = argparse.ArgumentParser()
# parser.add_argument("--data_path", type=str, default='data/ml-20m/ratings.csv')
parser.add_argument("--data_path", type=str, default='data/criteo/datasets.pickle')
parser.add_argument("--exp", type=str, default='dnis-5')
parser.add_argument("--cuda", type=str, default="[0]", help='cuda visible devices')
parser.add_argument("--embedding_dim", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--model_name", type=str, default='FM')
parser.add_argument("--lr_decay", type=int, default=1)
parser.add_argument("--l2", type=int, default=0)
parser.add_argument("--dataset_type", type=str, default='ctr')
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--use_apex", type=int, default=0)

args = parser.parse_args()
wandb.init(project="DNIS", name=args.exp + '-' + args.model_name + '-' + str(args.embedding_dim))
wandb.config.update(args)
dst = os.path.join(wandb.run.dir, "base_model.py")
copyfile("model/base_model.py", dst)
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'[1:-1]
device = torch.device('cuda')
patience = args.patience


def main():
    # data
    if args.dataset_type == "ctr":
        train_dataset, val_dataset, test_dataset, num_features = get_ctr_dataset(args.data_path)
        val_dataset = val_dataset[0]
        test_dataset = test_dataset[0]
    else:
        train_dataset, val_dataset, test_dataset, num_features = get_cf_dataset(args.data_path)
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
    num_batch_train = len(train_dataset) // batch_size + 1

    if args.dataset_type == "ctr":
        criterion = torch.nn.BCEWithLogitsLoss()
        num_fields = 39
    else:
        criterion = torch.nn.MSELoss()
        num_fields = 2

    model = getattr(base_model, args.model_name)(num_features, args.embedding_dim, num_fields)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    if len(eval(args.cuda)) > 1:
        model = torch.nn.DataParallel(model, device_ids=range(len(eval(args.cuda)))).cuda()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=0)
    wandb.watch(model, log='all')

    # train
    best_loss = np.inf
    best_epoch = -1
    for epoch in range(args.num_epochs):
        print(f"Starting epoch: {epoch} | phase: train | ⏰: {time.strftime('%H:%M:%S')}")
        model.train()
        running_loss = 0
        wandb.log({"w learning rate": optimizer.param_groups[0]['lr']}, step=epoch)
        for itr, batch in tqdm(enumerate(train_dataloader), total=num_batch_train):
            batch = [item.to(device) for item in batch]
            feature_ids, feature_vals, labels = batch
            outputs = model(feature_ids, feature_vals).squeeze()
            loss = criterion(outputs, labels)
            l2_reg = 0
            for name, param in model.named_parameters():
                if 'embedding' not in name:
                    l2_reg += torch.sum(param ** 2)
                else:
                    pass
                    # print("non l2:")
                    # print(name)
            total_loss = loss + args.l2 * l2_reg
            if args.use_apex:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if itr % (num_batch_train // 10) == 0:
                print(f"training loss in epoch {epoch} batch {itr}: {loss.item()}")
            running_loss += loss.item()
        epoch_loss = running_loss / num_batch_train
        print(f"training loss of epoch {epoch}: {epoch_loss}")
        wandb.log({"train_epoch_loss": epoch_loss}, step=epoch)
        torch.cuda.empty_cache()

        # validate
        if args.use_apex:
            state = {
                "epoch": epoch,
                "best_loss": best_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'amp': amp.state_dict(),
            }
        else:
            state = {
                "epoch": epoch,
                "best_loss": best_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        print(f"Starting epoch: {epoch} | phase: val | ⏰: {time.strftime('%H:%M:%S')}")
        val_loss, auc = val(model, val_dataloader, val_dataset)
        if args.lr_decay:
            scheduler.step(val_loss)
        print(f"validation loss of epoch {epoch}: {val_loss}, auc: {auc}")
        wandb.log({"val_epoch_loss": val_loss, "val_epoch_auc": auc}, step=epoch)
        test_loss, auc = val(model, test_dataloader, test_dataset)
        print(f"test loss of epoch {epoch}: {test_loss}, auc: {auc}")
        wandb.log({"test_epoch_loss": test_loss, "test_epoch_auc": auc}, step=epoch)
        if val_loss < best_loss:
            print("******** New optimal found, saving state ********")
            patience = args.patience
            state["best_loss"] = best_loss = val_loss
            best_epoch = epoch
            # torch.save(state, os.path.join(wandb.run.dir, f"{args.exp}.tar"))
            torch.save(state, f"checkpoint/{args.exp + '-' + args.model_name + '-' + str(args.embedding_dim)}.tar")
        else:
            patience -= 1
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            print('LR less than 1e-6, stop training...')
            break
        if patience == 0:
            print('patience == 0, stop training...')
            break

    # test, load the best checkpoint on val set
    print(f"Starting test | ⏰: {time.strftime('%H:%M:%S')}")
    # checkpoint = torch.load(os.path.join(wandb.run.dir, f"{args.exp}.tar"), map_location=torch.device('cpu'))
    checkpoint = torch.load(f"checkpoint/{args.exp + '-' + args.model_name + '-' + str(args.embedding_dim)}.tar", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if args.use_apex:
        amp.load_state_dict(checkpoint['amp'])
    test_loss, auc = val(model, test_dataloader, test_dataset)
    print(f"test loss of the best checkpoint in epoch {best_epoch}: {test_loss}, auc: {auc}")
    wandb.log({"test_loss": test_loss, "test_auc": auc})
    torch.save(model.feature_embeddings, "checkpoint/embeddings.pth")


def val(model, val_dataloader, val_data):
    model.eval()
    running_loss = 0
    if args.dataset_type == "ctr":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss()
    pred_arr = np.array([])
    label_arr = np.array([])
    with torch.no_grad():
        for itr, batch in tqdm(enumerate(val_dataloader), total=len(val_data) // args.batch_size):
            batch = [item.to(device) for item in batch]
            feature_ids, feature_vals, labels = batch
            outputs = model(feature_ids, feature_vals).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred_arr = np.hstack(
                [pred_arr, outputs.data.detach().cpu()]) if pred_arr.size else outputs.data.detach().cpu()
            label_arr = np.hstack(
                [label_arr, labels.data.detach().cpu()]) if label_arr.size else labels.data.detach().cpu()
        val_loss = running_loss / (itr + 1)
        torch.cuda.empty_cache()
    if args.dataset_type == "ctr":
        auc = roc_auc_score(label_arr, pred_arr)
        return val_loss, auc
    return val_loss, None


def send_message(msg):
    wechat = WeChatPub()
    wechat.send_msg(msg)


if __name__ == '__main__':
    seed_everything()
    start_time = time.time()
    main()
    end_time = time.time()
    print("{} minutes used for {}.".format((end_time - start_time) / 60, args.model_name))
    file_size = os.path.getsize('checkpoint/embeddings.pth')
    print("{} KB, i.e. {} MB used for the embedding layer.".format(file_size >> 10, file_size >> 20))
    # try:
    #     send_message(f"{args.exp} started")
    #     main()
    #     send_message(f"{args.exp} finished")
    # except Exception as e:
    #     print(e)
    #     send_message(f"{args.exp} failed: {e}")
