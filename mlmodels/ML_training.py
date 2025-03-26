# Nathan Holmes-King
# 2025-03-02

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
PEP-8 compliant.
"""

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def load_tensors(jdic):
    all_had = []
    all_req = []

    for a in jdic:
        try:
            for h in a['skills']:
                if h not in all_had:
                    all_had.append(h)
        except TypeError:
            pass
        try:
            for h in a['skills_required'].split('\n'):
                if len(h) > 0 and h not in all_req:
                    all_req.append(h)
        except AttributeError:
            pass

    X = []
    for i in range(len(jdic)):
        X_one = []
        for a in all_had:
            if jdic[i]['skills'] is None:
                X_one.append(0)
                continue
            if a in jdic[i]['skills']:
                X_one.append(1)
            else:
                X_one.append(0)
        for a in all_req:
            if jdic[i]['skills_required'] is None:
                X_one.append(0)
                continue
            if a in jdic[i]['skills_required'].split('\n'):
                X_one.append(1)
            else:
                X_one.append(0)
        X.append(X_one)
    X = torch.tensor(np.array(X, dtype=np.float32)).to(device)

    y = []
    for i in range(len(jdic)):
        y.append([jdic[i]['matched_score']])
    y = torch.tensor(np.array(y, dtype=np.float32)).to(device)
    return X, y


class BasicPredictor(nn.Module):
    def __init__(self, num_cols, hidden_sizes):
        super(BasicPredictor, self).__init__()

        layers = []
        input_size = num_cols
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            input_size = hidden
        # Final output layer
        layers.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)
        return output


def train_model(X_train, X_test, y_train, y_test, epochs):
    byrjun = time.time()
    model = BasicPredictor(
        num_cols=X_train.shape[1],
        hidden_sizes=[64, 32, 16, 8]
    ).to(device)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    train_loss = []
    test_loss = []

    for e in range(epochs):
        model.eval()
        with torch.no_grad():
            rl = 0.0
            for i in range(len(X_train)):
                y_pred = model(X_train[i])
                loss = criterion(y_pred, y_train[i])
                rl += loss.item()
            train_loss.append(rl / X_train.shape[0])
            rl = 0.0
            for i in range(len(X_test)):
                y_pred = model(X_test[i])
                loss = criterion(y_pred, y_test[i])
                rl += loss.item()
            test_loss.append(rl / X_test.shape[0])
        torch.mps.empty_cache()

        model.train()
        for i in range(len(X_train)):
            y_pred = model(X_train[i])
            loss = criterion(y_pred, y_train[i])
            if np.isnan(loss.item()):
                print('ERROR: NaN', i)
                print(train_loss)
                print(test_loss)
                break

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            torch.mps.synchronize()
            if i % 10 == 0:
                torch.mps.empty_cache()

        torch.mps.empty_cache()

        # Print loss for the current epoch
        print(f'Epoch [{e+1}/{epochs}], Time: {time.time() - byrjun}')

    print('Train loss:', train_loss[-1])
    print('Test loss:', test_loss[-1])

    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.ylim(bottom=0)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            jdic = json.load(f)
        X, y = load_tensors(jdic)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        train_model(X_train, X_test, y_train, y_test, 10)
    else:
        print('ERROR! Must include filename.')
