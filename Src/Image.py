import pandas as pd
import os
import cv2
import random
""""""""
class Image:
    def __init__(self):
        self.path_training_male = os.getcwd() + "/Dataset/Training/male/"
        self.path_training_female = os.getcwd() + "/Dataset/Training/female/"
        self.path_validation_male = os.getcwd() + "/Dataset/Validation/male/"
        self.path_validation_female = os.getcwd() + "/Dataset/Validation/female/"

        self.training_male = [file for file in os.listdir(self.path_training_male) if file.endswith(".jpg")]
        self.training_female = [file for file in os.listdir(self.path_training_female) if file.endswith(".jpg")]
        self.validation_male = [file for file in os.listdir(self.path_validation_male) if file.endswith(".jpg")]
        self.validation_female = [file for file in os.listdir(self.path_validation_female) if file.endswith(".jpg")]

        self.images = []
        self.genders = []
        self.purposes = []

    @staticmethod
    def load_image_with_extension(filename, extension=".jpg"):
        if filename.endswith(extension):
            image = cv2.imread(filename)
            if image is not None and not image.size == 0:
                return image
        return None

    def select_faces(self):
        for file in self.training_male:
            image = Image.load_image_with_extension(self.path_training_male  + file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.genders.append("male")
            self.purposes.append("training")

        for file in self.training_female:
            image = Image.load_image_with_extension(self.path_training_female+ file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.genders.append("female")
            self.purposes.append("training")

        for file in self.validation_male:
            image = Image.load_image_with_extension(self.path_validation_male + file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.genders.append("male")
            self.purposes.append("validation")

        for file in self.validation_female:
            image = Image.load_image_with_extension(self.path_validation_female + file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.genders.append("female")
            self.purposes.append("validation")

    def make_DataFrame(self):
        self.select_faces()
        self.df = pd.DataFrame({"Image":self.images, "Gender":self.genders, "Purpose":self.purposes})

    def check_loading_correctness(self):
        self.size = len(self.df)
        print("Total samples:", self.size)
        rand = random.choice(self.training_male + self.training_female + self.validation_male + self.validation_female)
        print("Example: ", rand)

        men = self.df[self.df["Gender"] == "male"]
        women = self.df[self.df["Gender"] == "female"]
        print(f"Final samples: \nMale: {len(men)} \nFemale: {len(women)}",)

    @property
    def getter_df(self):
        return self.df
