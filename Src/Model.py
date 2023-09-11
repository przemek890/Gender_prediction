import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
""""""""""""""""""""""""""""""""""""""""""
import torch.nn as nn

class Custom_Net(nn.Module):
    def __init__(self):
        super(Custom_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

#########################
class Gender_Model:
    def __init__(self, df,num_epochs=10):
        self.df = df.copy()
        self.df['Gender'] = self.df['Gender'].replace({'female': 1, 'male': 0})

        self.Train_Test_Split()
        self.num_epochs = num_epochs

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

        self.x_train_gender = torch.tensor(np.array(self.x_train_gender) / 255.0, dtype=torch.float32).view(-1, 52, 52,1)
        self.x_test_gender = torch.tensor(np.array(self.x_test_gender) / 255.0, dtype=torch.float32).view(-1, 52, 52, 1)
        self.y_train_gender = torch.tensor(np.array(self.y_train_gender), dtype=torch.float32).view(-1, 1)
        self.y_test_gender = torch.tensor(np.array(self.y_test_gender), dtype=torch.float32).view(-1, 1)

    def Build_Gender_Model(self):
        self.gender_model = Custom_Net()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.gender_model.parameters(), lr=0.001)

        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            optimizer.zero_grad()
            outputs = self.gender_model(self.x_train_gender)

            loss = criterion(outputs, self.y_train_gender)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.gender_model.eval()
                gender_predictions = self.gender_model(self.x_test_gender)
                gender_predictions = (gender_predictions > 0.5).int()
                accuracy = accuracy_score(self.y_test_gender.numpy(), gender_predictions.numpy())
                self.losses.append(loss.item())
                self.accuracies.append(accuracy)


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
        torch.save(self.gender_model.state_dict(), filename)