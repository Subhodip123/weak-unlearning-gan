import numpy as np
import torch
import wandb
from model import BaseClassifier
from datagen import Data
from config import config
from torch.optim import Adamax

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def evaluation(
    data_loader,
    label_loader,
    length,
    model_best=None,
    epoch=None,
    loss_type="validation"):
    """Given the test data it will give the best models performance/avg loss"""
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(f"./results/class-bangs/classifier{length}.pt")
    model_best.eval()
    val_loss = 0.0
    N = 0.0
    for data_batch, label_batch in zip(data_loader, label_loader):
        data_batch = data_batch.to(device=device)
        label_batch = label_batch.to(device=device)
        loss_t = model_best.loss(data_batch, label_batch)
        val_loss = val_loss + loss_t.sum().item()
        # print(loss_t)
        N = N + data_batch.shape[0]
    val_loss = val_loss / N
    if epoch is None:
      print(f"FINAL LOSS: loss={val_loss}")
    else:
      print(f"Total loss at Epoch: {epoch}, {loss_type} loss={val_loss}")
    return val_loss


def training(model, model_optim, no_epochs, train_dataloader, train_labelloader, 
             val_dataloader, val_labelloader, length, early_stop):
    """Train the classifier model"""
    # Main loop
    patience = 0
    for epoch in range(no_epochs):
        # TRAINING
        model.train()
        for input_batch, input_labels in zip(train_dataloader, train_labelloader):
            # data from training loader and stack the user ref image
            input_batch = input_batch.to(device)
            input_labels = input_labels.to(device)
            # loss and update
            loss = model.loss(input=input_batch, labels=input_labels).sum()
            # print(loss)
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()
            input_batch.detach().cpu()
            input_labels.detach().cpu()

        #Training Loss
        train_loss = evaluation(train_dataloader,
                                train_labelloader,
                                model_best=model,
                                length=length,
                                epoch=epoch,
                                loss_type="training")
        wandb.log({'train_loss': train_loss})
        
        # Validation loss
        val_loss = evaluation(
            val_dataloader,
            val_labelloader,
            model_best=model,
            epoch=epoch,
            length=length
        )
        wandb.log({'val_loss': val_loss})
        
        # save for plotting
        if epoch == 0:
            best_nll = val_loss
            torch.save(model, f"./results/class-bangs/classifier{length}.pt")
            print("saved! at epoch" + str(epoch))
        else:
            if val_loss < best_nll:
                torch.save(model, f"./results/class-bangs/classifier{length}.pt")
                print("saved! at epoch" + str(epoch))
                best_nll = val_loss
                patience = 0
            else:
                patience +=1
        if patience > early_stop:
            break

