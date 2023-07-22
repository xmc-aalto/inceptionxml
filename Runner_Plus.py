import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from scipy import sparse

class Runner:
    def __init__(self, train_dl, test_dl, inv_prop, top_k=5):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.num_train, self.num_test = len(train_dl.dataset), len(test_dl.dataset)
        self.top_k = top_k
        self.inv_prop = torch.from_numpy(inv_prop).cuda()

    def save_model(self, model, epoch, name):
        checkpoint = {
            'state_dict': model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            'dense_optimizer': self.dense_optimizer.state_dict(),
            'sparse_optimizer': self.sparse_optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, name)

    def load_model(self, model, name):
        # model_name = name.split('/')[-2]
        # print("Loading model: " + model_name)

        checkpoint = torch.load(name)
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            model.state_dict()['ext_classif_embed.weight'][:-1] = checkpoint['state_dict']['ext_classif_embed.weight']
            model.state_dict()['ext_classif_embed.weight'][-1].data.fill_(0)

        self.dense_optimizer.load_state_dict(checkpoint['dense_optimizer'])
        self.sparse_optimizer.load_state_dict(checkpoint['sparse_optimizer'])
        init = checkpoint['epoch']
        return model, init
    
    def predict_cluster(self, preds, y_true):
        for pred, yt in zip(preds, y_true):
            tr = torch.nonzero(yt, as_tuple=True)[0]
            match = (pred[..., None] == tr.cuda()).any(-1)
            self.group_count += torch.cumsum(match, dim=0)

    def predict(self, preds, y_true, ext=None):
        ext = self.extreme_count if ext is None else ext
        for pred, tr in zip(preds, y_true):
            match = (pred[..., None] == tr.cuda()).any(-1)
            ext += torch.cumsum(match, dim=0)
        
    def psp(self, preds, y_true, num=None, den=None):
        num = self.num if num is None else num
        den = self.den if den is None else den
        for pred, tr in zip(preds, y_true):
            match = (pred[..., None] == tr.cuda()).any(-1).double()
            match[match > 0] = self.inv_prop[pred[match > 0]]
            num += torch.cumsum(match, dim=0)

            inv_prop_sample = torch.sort(self.inv_prop[tr], descending=True)[0]

            match = torch.zeros(self.top_k).cuda()
            match_size = min(tr.shape[0], self.top_k)
            match[:match_size] = inv_prop_sample[:match_size]
            den += torch.cumsum(match, dim=0)

    def fit_one_epoch(self, model, params, epoch):
        train_loss = 0.0
        self.extreme_count = torch.zeros(self.top_k, dtype=np.int).cuda()
        self.group_count = torch.zeros(self.top_k, dtype=np.int).cuda()

        model.train()
        
        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}")
        for sample in pbar:
            x_batch, extreme_labels, group_labels = sample[0], sample[1], sample[2]
            x_batch, group_labels = x_batch.cuda(), group_labels.cuda()
            
            self.dense_optimizer.zero_grad()
            self.sparse_optimizer.zero_grad()

            loss, probs, group_probs, candidates = model(x_batch, extreme_labels,group_labels, epoch)
            loss.backward()
            self.dense_optimizer.step()
            self.sparse_optimizer.step()

            self.dense_cycle_scheduler.step()
            self.sparse_cycle_scheduler.step()

            train_loss += loss.item()
            
            if probs is not None:
                preds = torch.topk(probs, self.top_k)[1]
                preds = candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds]
                # preds = candidates[preds] # Only with shortlist
                self.predict(preds, extreme_labels)
            
            group_preds = torch.topk(group_probs, self.top_k)[1]
            self.predict_cluster(group_preds, group_labels)
            # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})
            
        train_loss /= self.num_train

        print(f"Epoch: {epoch}, LR: {self.dense_cycle_scheduler.get_last_lr()},  Train Loss: {train_loss}")
        prec = self.extreme_count.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1))
        group_prec = self.group_count.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1))

        print(f'Extreme Training Scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}')
        print(f'Group   Training Scores: P@1: {group_prec[0]:.2f}, P@3: {group_prec[2]:.2f}, P@5: {group_prec[4]:.2f}')

        # if train_loss < self.best_train_Loss:
        #     self.best_train_Loss = train_loss
        #     self.save_model(model, epoch, params.model_name + "/model_best_epoch.pth")
        
        if epoch % 5 == 0 or epoch >= params.num_epochs-10:
            self.test(model, params, epoch)

    def train(self, model, params, shortlist=False):
        self.best_train_Loss = float('Inf')
        self.best_test_acc = 0
        lr = params.lr

        model = model.cuda()

        # self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.dense_optimizer = optim.Adam([p for n, p in model.named_parameters() if 
                                                n not in ['lookup.weight', 'ext_classif_embed.weight']] , lr=lr)
        if params.sparse:
            self.sparse_optimizer = optim.SparseAdam([p for n, p in model.named_parameters() if 
                                                n in ['lookup.weight', 'ext_classif_embed.weight']] , lr=lr)
        else:
            self.sparse_optimizer = optim.Adam([p for n, p in model.named_parameters() if 
                                                n in ['lookup.weight', 'ext_classif_embed.weight']] , lr=lr)

        init = 0
        last_batch = -1
        steps_per_epoch = len(self.train_dl)

        if len(params.load_model) or shortlist:
            model, init = self.load_model(model, params.load_model)
            last_batch = (init-1)*steps_per_epoch

        if params.test:
            self.test(model, params, init)
            return

        self.sparse_cycle_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=self.sparse_optimizer, max_lr=params.lr,
            epochs=params.num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.33,
            div_factor=10, final_div_factor=1e4, last_epoch=last_batch)

        self.dense_cycle_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=self.dense_optimizer, max_lr=params.lr,
            epochs=params.num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.33,
            div_factor=10, final_div_factor=1e4, last_epoch=last_batch)
        
        for epoch in range(init, params.num_epochs):
            self.fit_one_epoch(model, params, epoch+1)


    def test(self, model, params, epoch=0):
        model.eval()
        with torch.no_grad():
            testLoss = 0.0
            self.extreme_count = torch.zeros(self.top_k, dtype=torch.int32).cuda()
            self.num = torch.zeros(self.top_k).cuda()
            self.den = torch.zeros(self.top_k).cuda()
            if params.model != "InceptionXML":
                self.comb_extreme_count = torch.zeros(self.top_k, dtype=torch.int32).cuda()
                self.comb_num = torch.zeros(self.top_k).cuda()
                self.comb_den = torch.zeros(self.top_k).cuda()

            for x_batch, y_tr in tqdm(self.test_dl, desc=f"Epoch {epoch}"):
                x_batch = x_batch.cuda()
                candidates, probs, comb_probs = model(x_batch)

                preds = torch.topk(probs, self.top_k)[1]
                preds = candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds]
                
                self.predict(preds, y_tr)
                self.psp(preds, y_tr)
                
                if params.model != "InceptionXML":
                    comb_preds = torch.topk(comb_probs, self.top_k)[1]
                    comb_preds = candidates[np.arange(comb_preds.shape[0]).reshape(-1, 1), comb_preds]
                    self.predict(comb_preds, y_tr, self.comb_extreme_count)
                    self.psp(comb_preds, y_tr, self.comb_num, self.comb_den)

            # testLoss /= self.num_test
            # print(f"Test Loss: {testLoss}")
            prec = self.extreme_count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1))
            psp = (self.num * 100 / self.den).detach().cpu().numpy()
            if params.model != "InceptionXML":
                comb_prec = self.comb_extreme_count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1))
                comb_psp = (self.comb_num * 100 / self.comb_den).detach().cpu().numpy()

            print(f"Test scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}, PSP@1: {psp[0]:.2f}, PSP@3: {psp[2]:.2f}, PSP@5: {psp[4]:.2f}")
            if params.model != "InceptionXML":
                print(f"Comb Test scores: P@1: {comb_prec[0]:.2f}, P@3: {comb_prec[2]:.2f}, P@5: {comb_prec[4]:.2f}, PSP@1: {comb_psp[0]:.2f}, PSP@3: {comb_psp[2]:.2f}, PSP@5: {comb_psp[4]:.2f}\n")

        if(prec[0]+prec[2]+prec[4]+psp[0]+psp[2]+psp[4] > self.best_test_acc and not params.test):
            # print("Best Accuracy Model Epoch {}".format(epoch))
            # best_test_loss = testLoss
            self.best_test_acc = prec[0]+prec[2]+prec[4]+psp[0]+psp[2]+psp[4]
            self.save_model(model, epoch, params.model_name + "/model_best_test.pth")
