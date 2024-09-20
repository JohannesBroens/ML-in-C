import matplotlib.pyplot as plt

# Load losses from file
epochs = []
train_losses = []
val_losses = []

with open('/home/broens/github/ML-in-C/src/MLP/losses.txt', 'r') as f:
    for line in f:
        epoch, train_loss, val_loss = line.strip().split()
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

# Plot the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()
