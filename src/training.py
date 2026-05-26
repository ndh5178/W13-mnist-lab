# -*- coding: utf-8 -*-
"""Training, evaluation, and plotting helpers."""

import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """Train the model with mini-batch gradient descent."""
    loss_history = []
    num_train = x_train.shape[0]

    for _ in range(epochs):
        indices = np.random.permutation(num_train)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        for start in range(0, num_train, batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            current_batch_size = x_batch.shape[0]

            y_pred = model.forward(x_batch, train=True)
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_loss += loss * current_batch_size

            dout = y_pred.copy()
            dout[np.arange(current_batch_size), y_batch] -= 1
            dout /= current_batch_size

            model.backward(dout)
            optimizer.update(model.params, model.grads)

        loss_history.append(epoch_loss / num_train)

    return loss_history


def evaluate(model, x, y):
    """Return accuracy percentage and total parameter count."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """Plot the training loss curve."""
    import matplotlib.pyplot as plt

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
