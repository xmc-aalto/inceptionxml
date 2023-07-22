import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from scipy import sparse


class Runner:
    def __init__(self,train_dl, test_dl, inv_prop, top_k=5):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.num_train, self.num_test = len(train_dl.dataset), len(test_dl.dataset)
        self.top_k = top_k
        self.inv_prop = torch.from_numpy(inv_prop).cuda()

    def save_model(self, model, epoch, name):
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, name)

    def load_model(self, model, name):
        model_name = name.split('/')[-2]
        print("Loading model: " + model_name)
        
        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        init = checkpoint['epoch']
        return model, init
    
    def predict(self, preds, y_true):
        for pred, yt in zip(preds, y_true):
            tr = torch.nonzero(yt, as_tuple=True)[0]
            match = (pred[..., None] == tr).any(-1)
            self.correct_count += torch.cumsum(match, dim=0)

    def psp(self, preds, y_true):
        for pred, yt in zip(preds, y_true):
            tr = torch.nonzero(yt, as_tuple=True)[0]
            match = (pred[..., None] == tr).any(-1).double()

            match[match > 0] = self.inv_prop[pred[match > 0]]
            self.num += torch.cumsum(match, dim=0)

            inv_prop_sample = torch.sort(self.inv_prop[tr], descending=True)[0]

            match = torch.zeros(self.top_k).cuda()
            match_size = min(tr.shape[0], self.top_k)
            match[:match_size] = inv_prop_sample[:match_size]
            self.den += torch.cumsum(match, dim=0)


    def fit_one_epoch(self, model, params, epoch):
        trainLoss = 0.0
        self.correct_count = torch.zeros(self.top_k, dtype=np.int).cuda()

        model.train()
        
        for x_batch, y_batch in tqdm(self.train_dl, desc = f"Epoch {epoch}"):
            self.optimizer.zero_grad()
            x_batch, y_batch = x_batch.long().cuda(), y_batch.cuda()
            output, loss = model(x_batch, y_batch)
            
            loss.backward()
            self.optimizer.step()
            self.cycle_scheduler.step()
            trainLoss += loss.item()
            # trainloop.set_description("Epoch {}: loss = {}".format(epoch, loss.item()/params.batch_size))
            # trainloop.refresh()

            preds = torch.topk(output, self.top_k)[1]
            self.predict(preds, y_batch)
        
        trainLoss /= self.num_train
        
        print(f"Epoch: {epoch}, LR: {self.cycle_scheduler.get_last_lr()},  Train Loss: {trainLoss}")
        prec = self.correct_count.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1))
        print(f'Training Scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}')

        # if trainLoss < self.best_train_Loss:
        #     self.best_train_Loss = trainLoss
        #     self.save_model(model, epoch, params.model_name + "/model_best_epoch.pth")
    
        if epoch % 5 == 0 or epoch >= params.num_epochs-10:
            self.test(model, params, epoch)


    def train(self, model, params):
        self.best_train_Loss = float('Inf')
        self.best_test_acc = 0
        total_epochs = params.num_epochs
        lr = params.lr
        steps_per_epoch = len(self.train_dl)
        
        model = model.cuda()

        self.optimizer = optim.Adam(model.parameters(), lr = lr)
        
        init = 0
        last_batch = -1

        if len(params.load_model):
            model, init = self.load_model(model, params.load_model)
            last_batch = (init-1)*steps_per_epoch

        if params.test:
            self.test(model, params, init)
            return

        self.cycle_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer, max_lr=params.lr,
            epochs=total_epochs, steps_per_epoch=steps_per_epoch, 
            div_factor=10, final_div_factor=1e4, last_epoch=last_batch)

        for epoch in range(init, params.num_epochs):
            self.fit_one_epoch(model, params, epoch+1)

    def test(self, model, params, epoch = 0):
        model.eval()
        with torch.no_grad():
            self.correct_count = torch.zeros(self.top_k, dtype=torch.int32).cuda()
            self.num = torch.zeros(self.top_k).cuda()
            self.den = torch.zeros(self.top_k).cuda()

            for i, batch_data in enumerate(tqdm(self.test_dl, desc = f"Epoch {epoch}")):
                x_batch, y_batch = batch_data[0].long().cuda(), batch_data[1].cuda()
                output = model(x_batch)

                preds = torch.topk(output, self.top_k)[1]
                self.predict(preds, y_batch)
                self.psp(preds, y_batch)
            
            prec = self.correct_count.detach().cpu().numpy() * 100.0 / (self.num_test * torch.arange(1, self.top_k+1))
            psp = (self.num * 100 / self.den).detach().cpu().numpy()
            
            print(f"Test scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}, PSP@1: {psp[0]:.2f}, PSP@3: {psp[2]:.2f}, PSP@5: {psp[4]:.2f}\n")
        
        if(prec[0]+prec[2]+prec[4] > self.best_test_acc and not params.test):
            self.best_test_acc = prec[0]+prec[2]+prec[4]
            self.save_model(model, epoch, params.model_name + "/model_best_test.pth")
