import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
""""""""""""""""""""""""""""""""""""""""""

class Custom_Net(nn.Module):
    def __init__(self):
        super(Custom_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 25 * 25, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
#########################
class Gender_Model:
    def __init__(self, df,num_epochs=10,patience=3):
        self.df = df.copy()
        self.df['Gender'] = self.df['Gender'].replace({'female': 1, 'male': 0})

        self.Train_Test_Split()
        self.num_epochs = num_epochs
        self.patience = patience
        self.best_accuracy = 0
        self.counter = 0

        self.losses = []
        self.accuracies = []
    def Train_Test_Split(self):

        self.x_train_gender = []
        self.x_test_gender = []
        self.y_train_gender = []
        self.y_test_gender = []

        for i in range(len(self.df)):
            if self.df["Purpose"].iloc[i] == "training":
                array_training_im = np.asarray(self.df['Image'].iloc[i])
                self.x_train_gender.append(array_training_im)
                self.y_train_gender.append(int(self.df['Gender'].iloc[i]))
            elif self.df["Purpose"].iloc[i] == "validation":
                array_validation = np.asarray(self.df['Image'].iloc[i])
                self.x_test_gender.append(array_validation)
                self.y_test_gender.append(int(self.df['Gender'].iloc[i]))

        print(f"x_train_gender: {len(self.x_train_gender)}, x_test_gender: {len(self.x_test_gender)}, y_train_gender: {len(self.y_train_gender)}, y_test_gender: {len(self.y_test_gender)}")

        self.x_train_gender = torch.tensor(np.array(self.x_train_gender) / 255.0, dtype=torch.float32).view(-1,3,52,52)
        self.x_test_gender = torch.tensor(np.array(self.x_test_gender) / 255.0, dtype=torch.float32).view(-1,3,52,52)
        self.y_train_gender = torch.tensor(np.array(self.y_train_gender), dtype=torch.float32).view(-1, 1)
        self.y_test_gender = torch.tensor(np.array(self.y_test_gender), dtype=torch.float32).view(-1, 1)

        train_dataset_gender = TensorDataset(self.x_train_gender, self.y_train_gender)
        test_dataset_gender = TensorDataset(self.x_test_gender, self.y_test_gender)

        batch_size = 64
        self.trainloader = torch.utils.data.DataLoader(train_dataset_gender, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(test_dataset_gender, batch_size=batch_size, shuffle=True)

    def Build_Gender_Model(self):
        self.gender_model = Custom_Net()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.gender_model.parameters(), lr=0.001)

        for epoch in range(self.num_epochs):
            for inputs, labels in self.trainloader:
                optimizer.zero_grad()
                outputs = self.gender_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.gender_model.eval()
                gender_predictions = []
                true_labels = []

                for inputs, labels in self.testloader:
                    outputs = self.gender_model(inputs)
                    gender_predictions.extend((outputs > 0.5).int().numpy())
                    true_labels.extend(labels.numpy())

                accuracy = accuracy_score(true_labels, gender_predictions)
                self.losses.append(loss.item())
                self.accuracies.append(accuracy)

                print(f"Epoch {epoch + 1}: Loss {loss.item()}, Accuracy {accuracy}")

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print("Early stopping triggered.")
                        break

    def Loss_accuracy_charts(self):

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.losses) + 1), self.losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("../Analysis/Loss.png")
        plt.show()

        # Rysowanie wykresu dokładności
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.accuracies) + 1), self.accuracies, label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("../Analysis/Accuracy.png")
        plt.show()

    def Save_model(self,filename):
        torch.save({'weights': self.gender_model.state_dict()}, filename)