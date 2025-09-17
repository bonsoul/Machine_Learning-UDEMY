import matplotlib.pyplot as plt
from torchvision import datasets, transforms


transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(root='./data', train=True,transform=transform, download=True)


#visualize sample images
#fig, axes = plt.subplots(1,5, figsize=(12,3))
#for i in range(5):
    #image, label = train_dataset[i]
    #axes[i].imshow(image.permute(1, 2, 0))
    #axes[i].axis('off')
    #axes[i].set_title(f"Label : {label}")
#plt.show()

#DISPLAY PEXEL VALUE
image, label = train_dataset[0]
print(f"Label: {label}")
print(f"Image Shape: {image.shape}")
print(f"Pixel Values:")
print(image)