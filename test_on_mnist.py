import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# collect indices for labels 4 and 8
indices_4 = [i for i, (_, y) in enumerate(train_dataset) if y == 4]
indices_8 = [i for i, (_, y) in enumerate(train_dataset) if y == 8]

# random selection
selected_4 = random.sample(indices_4, 50)
selected_8 = random.sample(indices_8, 50)

selected_indices = selected_4 + selected_8
random.shuffle(selected_indices)

# load selected data
images = torch.stack([train_dataset[i][0] for i in selected_indices])
labels = torch.tensor([train_dataset[i][1] for i in selected_indices])

print(images.shape)  # (100, 1, 28, 28)
print(labels.shape)  # (100,)

num_show = 20
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()