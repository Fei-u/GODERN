import torch
import math
import os
import time
import copy
import numpy as np
import gc
from torch import distributions
from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.TrainInits import print_model_parameters

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler, device,
                 w):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.mae = torch.nn.L1Loss().to(args.device)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.device = device
        self.w = w

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
            # for iter, batch in enumerate(val_dataloader):
                batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
                valid_coeffs, target = batch
                # data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(valid_coeffs)
                # output = self.scaler.inverse_transform(output)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                else:
                    label = self.scaler.inverse_transform(label)
                    output = self.scaler.inverse_transform(output)
                loss = self.mae(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        if self.args.tensorboard:
            self.w.add_scalar(f'valid/loss', val_loss, epoch)
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
            train_coeffs, target = batch
            # data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()
            output = self.model(train_coeffs)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            else:
                label = self.scaler.inverse_transform(label)
                output = self.scaler.inverse_transform(output)
            # likelihood_constructor = {"laplace": distributions.Laplace, "normal": distributions.Normal}["normal"]
            # likelihood = likelihood_constructor(loc=target, scale=0.1)
            # logpy = likelihood.log_prob(output).sum(dim=0).mean(dim=0)
            # logpy = logpy.sum(dim=0).mean(dim=0)
            loss = self.loss(output.cuda(), label)
            mae = self.mae(output.cuda(), label)
            loss.backward()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += mae.item()
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, mae.item()))
            # gc.collect()
            # torch.cuda.empty_cache()
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        if self.args.tensorboard:
            self.w.add_scalar(f'train/loss', train_epoch_loss, epoch)

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            
            # if epoch%10==0:#test
            #     self.model.load_state_dict(best_model)
            #     #self.val_epoch(self.args.epochs, self.test_loader)
            #     self.test_simple(self.model, self.args, self.test_loader, self.scaler, self.logger, None, self.times)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        # if epoch==10:#test
        #     self.model.load_state_dict(best_model)
        #     #self.val_epoch(self.args.epochs, self.test_loader)
        #     self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, None, self.times)
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, None)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
                test_coeffs, target = batch
                label = target[..., :args.output_dim]
                output = model(test_coeffs)
                y_true.append(label)
                y_pred.append(output)
        if args.real_value:
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        np.save(args.log_dir+'/{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save(args.log_dir+'/{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))

    @staticmethod
    def test_simple(model, args, data_loader, scaler, logger, path):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
            # for batch_idx, (data, target) in enumerate(data_loader):
                batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
                test_coeffs, target = batch
                # data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(test_coeffs)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))


def _add_weight_regularisation(total_loss, regularise_parameters, scaling=0.03):
    for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
    return total_loss