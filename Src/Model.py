import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
""""""""""""""""""""""""""""""""""""""""""
class Custom_Net(nn.Module):
    def __init__(self,dropout_prob=0.5):
        super(Custom_Net, self).__init__()

        kernel_s = 3

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_s, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_s, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(11*11*64, 256)

        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.reshape(-1, 11*11*64)

        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x



#########################
class Gender_Model:
    def __init__(self, df,num_epochs=15,patience=3):
        self.df = df.copy()
        self.df['Gender'] = self.df['Gender'].replace({'female': 1, 'male': 0})

        self.Train_Test_Split()
        self.num_epochs = num_epochs
        self.patience = patience

        self.best_accuracy = 0.0
        self.best_accuracy_loss = 0.0

        self.best_model_weights = None
        self.losses = []
        self.accuracies = []

        self.no_improvement_count = 0
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

        self.x_train_gender = torch.tensor(np.array(self.x_train_gender) / 255.0, dtype=torch.float32).reshape(-1,3,52,52)
        self.x_test_gender = torch.tensor(np.array(self.x_test_gender) / 255.0, dtype=torch.float32).reshape(-1,3,52,52)
        self.y_train_gender = torch.tensor(np.array(self.y_train_gender), dtype=torch.float32).reshape(-1, 1)
        self.y_test_gender = torch.tensor(np.array(self.y_test_gender), dtype=torch.float32).reshape(-1, 1)

        train_dataset_gender = TensorDataset(self.x_train_gender, self.y_train_gender)
        test_dataset_gender = TensorDataset(self.x_test_gender, self.y_test_gender)

        batch_size = 64
        self.trainloader = torch.utils.data.DataLoader(train_dataset_gender, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(test_dataset_gender, batch_size=batch_size, shuffle=True)

    def Build_Gender_Model(self):
        self.gender_model = Custom_Net()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.gender_model.parameters(), lr=0.0005)

    def train(self):
        for epoch in range(self.num_epochs):
            for inputs, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.gender_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

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

                print(f"Epoch {epoch + 1}: Loss {round(loss.item(),3)}, Accuracy {round(accuracy,3)}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_accuracy_loss = loss
                self.best_model_weights = self.gender_model.state_dict()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                print(f"Stop learning, no improvement for {self.patience} epochs.")
                break

        if self.best_model_weights is not None:
            self.gender_model.load_state_dict(self.best_model_weights)


    def Loss_accuracy_charts(self):

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.losses) + 1), self.losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f"Loss_{self.best_accuracy_loss}")
        plt.savefig(f"Analysis/Loss.png")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.accuracies) + 1), self.accuracies, label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f"Accuracy_{self.best_accuracy}")
        plt.savefig(f"Analysis/Accuracy.png")
        plt.show()

    def Save_model(self,filename):
        torch.save({'weights': self.gender_model.state_dict()}, filename)