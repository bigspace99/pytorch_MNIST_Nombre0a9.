import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Transformation des données d'entraînement
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Téléchargement de l'ensemble de données MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Définition du réseau neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Entraînement du réseau
for epoch in range(2):  # 2 epochs pour l'exemple
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
            running_loss = 0.0

print("Finished Training")

# Export du modèle
torch.save(net.state_dict(), 'mnist_net.pth')
print("Model saved successfully")

#========================================================
#test du modele avec les images de la bibliothèque 
#========================================================

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Charger le modèle PyTorch
model.load_state_dict(torch.load('mnist_net.pth'))
model.eval()

# Charger le dataset MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


for images, labels in test_loader:
    # Afficher l'image
    image = images[0].numpy().squeeze()
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Faire la prédiction
    with torch.no_grad():
        output = model(images)
        _, predicted = torch.max(output, 1)

    # Vérifier si la prédiction est correcte
    if predicted.item() == labels.item():
        plt.title(f'Prediction: {predicted.item()}', color='green')
    else:
        plt.title(f'Prediction: {predicted.item()} (wrong)', color='red')

    plt.show()


#========================================================
#test du modele avec les images de la bibliothèque 
#========================================================

from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from google.colab import files


# Charger le modèle PyTorch
model.load_state_dict(torch.load('mnist_net.pth'))
model.eval()

# Charger une image depuis l'ordinateur local
uploaded = files.upload()

# Récupérer le chemin de l'image chargée
image_path = list(uploaded.keys())[0]

# Charger l'image
image = Image.open(image_path)

# Convertir l'image en niveaux de gris (L) si elle est en couleur
image = image.convert('L')

# Prétraitement de l'image
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Redimensionner à la taille attendue
    transforms.ToTensor(),        # Convertir en un tenseur
    transforms.Normalize((0.5,), (0.5,))  # Normalisation
])

# Appliquer les transformations à l'image
input_tensor = transform(image).unsqueeze(0)  # Ajouter une dimension de batch

# Faire une prédiction
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

# Afficher le résultat de la prédiction
plt.imshow(image, cmap='gray')  # Afficher en niveaux de gris
plt.axis('off')

# Vérifier si la prédiction est correcte
if predicted.item() == 0:  # Adapter cette condition selon votre modèle
    plt.title(f'Prediction: {predicted.item()}', color='green')
else:
    plt.title(f'Prediction: {predicted.item()} (wrong)', color='red')

plt.show()

