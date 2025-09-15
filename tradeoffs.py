import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 21)

# Simulated curves
# Good fit
train_loss_good = np.exp(-epochs/8) + 0.1*np.random.rand(len(epochs))
val_loss_good = np.exp(-epochs/7.5) + 0.15*np.random.rand(len(epochs))

train_acc_good = 0.6 + 0.4*(1-np.exp(-epochs/5)) + 0.02*np.random.rand(len(epochs))
val_acc_good = 0.55 + 0.4*(1-np.exp(-epochs/6)) + 0.02*np.random.rand(len(epochs))

# Overfitting
train_loss_over = np.exp(-epochs/8) * 0.5
val_loss_over = np.exp(-epochs/12) + 0.3*np.maximum(0, epochs-10)/10

train_acc_over = 0.7 + 0.3*(1-np.exp(-epochs/5))
val_acc_over = 0.65 + 0.25*(1-np.exp(-epochs/7)) - 0.01*np.maximum(0, epochs-10)

# Underfitting
train_loss_under = 1.0 - 0.02*epochs
val_loss_under = 1.1 - 0.015*epochs

train_acc_under = 0.4 + 0.01*epochs
val_acc_under = 0.35 + 0.008*epochs

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# Good fit
axes[0].plot(epochs, train_loss_good, label="Train Loss", color="blue")
axes[0].plot(epochs, val_loss_good, label="Val Loss", color="orange")
axes[0].plot(epochs, train_acc_good, "--", label="Train Acc", color="blue")
axes[0].plot(epochs, val_acc_good, "--", label="Val Acc", color="orange")
axes[0].set_title("Good Fit")
axes[0].set_xlabel("Epochs")
axes[0].legend()

# Overfitting
axes[1].plot(epochs, train_loss_over, label="Train Loss", color="blue")
axes[1].plot(epochs, val_loss_over, label="Val Loss", color="orange")
axes[1].plot(epochs, train_acc_over, "--", label="Train Acc", color="blue")
axes[1].plot(epochs, val_acc_over, "--", label="Val Acc", color="orange")
axes[1].set_title("Overfitting")
axes[1].set_xlabel("Epochs")
axes[1].legend()

# Underfitting
axes[2].plot(epochs, train_loss_under, label="Train Loss", color="blue")
axes[2].plot(epochs, val_loss_under, label="Val Loss", color="orange")
axes[2].plot(epochs, train_acc_under, "--", label="Train Acc", color="blue")
axes[2].plot(epochs, val_acc_under, "--", label="Val Acc", color="orange")
axes[2].set_title("Underfitting")
axes[2].set_xlabel("Epochs")
axes[2].legend()

plt.suptitle("Training vs Validation Curves: Good Fit, Overfitting, Underfitting", fontsize=14, weight="bold")
plt.show()
