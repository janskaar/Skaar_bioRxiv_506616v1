import torch
import numpy as np
import os

DEVICE = "cpu"
MAX_EPOCHS = 5000
BATCH_SIZE = 100
EVALUATION_FREQUENCY = 10 # evaluate model every x epochs
STOP_PERSISTENCE = 10      # number of evaluations to train for without loss improvement before stopping
STOP_THRESHOLD = 1.       # value to consider "not significantly improving"

def train(model, train_batch_fn, eval_batch_fn, train_loader, test_loader, logfile=None, savefile=None):
    min_loss = 1e10
    for i in range(MAX_EPOCHS):
        for j, (x_batch, y_batch) in enumerate(train_loader):
            train_batch_fn(x_batch, y_batch)

        ## Evaluate on training/test 
        if i % EVALUATION_FREQUENCY ==  0:
            with torch.no_grad():
                test_loss = []
                for x_batch, y_batch in test_loader:
                    loss = eval_batch_fn(x_batch, y_batch)
                    test_loss.append(loss)
                test_loss = np.array(test_loss)

                train_loss = []
                for x_batch, y_batch in train_loader:
                    loss = eval_batch_fn(x_batch, y_batch)
                    train_loss.append(loss)
                train_loss = np.array(train_loss)

                # Track mean log prob on training data
                # in order to evaluate training progress
                current_loss = train_loss.mean()
                print(f"Epoch {i}, loss: {current_loss:.2f}")
                if logfile:
                    with open(logfile, 'a') as f:
                        f.writelines(f'{train_loss.mean()},{test_loss.mean()}\n')

                # If improvement, save model 
                if current_loss < (min_loss-STOP_THRESHOLD):
                    if savefile:
                        torch.save(model, savefile)
                    min_loss = current_loss
                    rounds_wo_improvements = 0

                # If no improvement for STOP_PERSISTENCE epochs, stop training
                else:
                    rounds_wo_improvements += 1
                    if rounds_wo_improvements > STOP_PERSISTENCE:
                        print("NO IMPROVEMENT, BREAKING")
                        break


