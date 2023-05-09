import matplotlib.pyplot as plt


plt.figure(figsize=(8, 8))

# open files
file1 = open("augmentation/with_augmentation.txt")
file2 = open("augmentation/no_augmentation.txt")

# read files
file1_acc = [float(x) for x in file1.readline().strip().split(',')]
file1_val_acc = [float(x) for x in file1.readline().strip().split(',')]
file1_loss = [float(x) for x in file1.readline().strip().split(',')]
file1_val_loss = [float(x) for x in file1.readline().strip().split(',')]
file2_acc = [float(x) for x in file2.readline().strip().split(',')]
file2_val_acc = [float(x) for x in file2.readline().strip().split(',')]
file2_loss = [float(x) for x in file2.readline().strip().split(',')]
file2_val_loss = [float(x) for x in file2.readline().strip().split(',')]

# plot accuracies
epochs_range = range((len(file1_acc)))
plt.plot(epochs_range, file1_acc, label="Training Accuracy [Augmentations]")
plt.plot(epochs_range, file1_val_acc, label="Validation Accuracy [Augmentations]")
plt.plot(epochs_range, file2_acc, label="Training Accuracy [No augmentations]")
plt.plot(epochs_range, file2_val_acc, label="Validation Accuracy [No augmentations]")
plt.axis(ymin=0.00, ymax=1.2)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train [Augmentations]', 'validation [Augmentations]', 'train [No augmentations]', 'validation [No augmentations]'])
plt.show()
plt.savefig('output-plot.png')

# plot losses
plt.figure(figsize=(8, 8))
epochs_range = range((len(file1_acc)))
plt.plot(epochs_range, file1_acc, label="Training Loss [Augmentations]")
plt.plot(epochs_range, file1_val_loss, label="Validation Loss [Augmentations]")
plt.plot(epochs_range, file2_loss, label="Training Loss [No augmentations]")
plt.plot(epochs_range, file2_val_loss, label="Validation Loss [No augmentations]")
plt.axis(ymin=0.00, ymax=20)
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train [Augmentations]', 'validation [Augmentations]', 'train [No augmentations]', 'validation [No augmentations]'])
plt.show()
plt.savefig('output-plot.png')
