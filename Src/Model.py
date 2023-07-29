from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
""""""""
class Model:
    def __init__(self, df):
        self.df = df
        self.x_train_age = None
        self.x_test_age = None
        self.y_train_age = None
        self.y_test_age = None
        self.x_train_gender = None
        self.x_test_gender = None
        self.y_train_gender = None
        self.y_test_gender = None
        self.epochs = 20

        self.age_history = None
        self.gender_history = None

        self.age_h5 = None
        self.gender_h5 = None

    def Train_Test_Split(self):
        x = []
        y_age = []
        y_gender = []

        for i in range(len(self.df)):
            ar = np.asarray(self.df['Images'].iloc[i])
            x.append(ar)
            agegen = [int(self.df['Ages'].iloc[i]), int(self.df['Genders'].iloc[i])]
            y_age.append(agegen[0])
            y_gender.append(agegen[1])
        x = np.array(x)

        self.x_train_age, self.x_test_age, self.y_train_age, self.y_test_age = train_test_split(x, y_age, test_size=0.2, stratify=y_age)
        self.x_train_gender, self.x_test_gender, self.y_train_gender, self.y_test_gender = train_test_split(x, y_gender, test_size=0.2, stratify=y_gender)

        print("x_train_age: {}, x_test_age: {}, y_train_age: {}, y_test_age: {}".format(len(self.x_train_age), len(self.x_test_age), len(self.y_train_age), len(self.y_test_age)))
        print("x_train_gender: {}, x_test_gender: {}, y_train_gender: {}, y_test_gender: {}".format(len(self.x_train_gender), len(self.x_test_gender), len(self.y_train_gender), len(self.y_test_gender)))

    def Build_Age_Model(self):
        def Age_conv_arr():
            self.y_train_age = np.array(self.y_train_age)
            self.y_test_age = np.array(self.y_test_age)
            self.x_train_age = np.array(self.x_train_age)
            self.x_test_age = np.array(self.x_test_age)
        Age_conv_arr()

        # Age_model
        age_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.x_train_age.shape[1:] + (1,)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        # Model compilation
        age_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        age_model.summary()

        # Model fitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # Early Stopping
        self.age_history = age_model.fit(
            self.x_train_age,
            self.y_train_age,
            batch_size=32,
            epochs=self.epochs,
            validation_data=(self.x_test_age, self.y_test_age),
            callbacks=[early_stopping]
        )

        # Prediction on test data
        age_predictions = age_model.predict(self.x_test_age)

        # Convert predictions to labels
        age_predictions = age_predictions.flatten().round().astype(int)

        # Model evaluation
        age_accuracy = accuracy_score(self.y_test_age, age_predictions)

        # Display results
        self.age_h5 = age_model
        print("Accuracy of the age prediction model:", age_accuracy)
    def Build_Gender_Model(self):
        def Gender_conv_arr():
            self.x_train_gender = np.array(self.x_train_gender)
            self.x_test_gender = np.array(self.x_test_gender)
            self.y_train_gender = np.array(self.y_train_gender)
            self.y_test_gender = np.array(self.y_test_gender)
        Gender_conv_arr()

        # Gender_model
        gender_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.x_train_gender.shape[1:] + (1,)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Model compilation
        gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        gender_model.summary()

        # Model fitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # Early Stopping
        self.gender_history = gender_model.fit(
            self.x_train_gender,
            self.y_train_gender,
            epochs=self.epochs,
            validation_data=(self.x_test_gender,self.y_test_gender),
            callbacks=[early_stopping]
        )

        # Prediction on test data
        gender_predictions = gender_model.predict(self.x_test_gender)

        # Convert predictions to labels
        gender_predictions = (gender_predictions > 0.5).astype(int)

        # Model evaluation
        gender_accuracy = accuracy_score(self.y_test_gender, gender_predictions)

        # Display results
        self.gender_h5 = gender_model
        print("Accuracy of the gender prediction model:",gender_accuracy)


    def Save_age_model(self):
        self.age_h5.save("age_model.h5")
    def Save_gender_model(self):
        self.gender_h5.save("gender_model.h5")

    @staticmethod
    def Age_learning_chart(age_history):
        plt.plot(age_history.history['loss'])
        plt.plot(age_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig("./Analysis/Age_learning_chart.png")
        plt.show()

    @staticmethod
    def Gender_learning_chart(gender_history):
        plt.plot(gender_history.history['accuracy'])
        plt.plot(gender_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig("./Analysis/Gender_learning_chart.png")
        plt.show()