import pandas as pd
import os
import cv2
from PIL import Image
import random
""""""""
class Image:
    def __init__(self):
        self.path  = os.getcwd() + "/../Dataset/"
        self.training_male = [file for file in os.listdir(self.path) if file.endswith(".jpg") and file.split("_")[1] == '0' and file.split("_")[2].split(".")[0] == '0']
        self.training_female = [file for file in os.listdir(self.path) if file.endswith(".jpg") and file.split("_")[1] == '1' and file.split("_")[2].split(".")[0] == '0']
        self.validation_male = [file for file in os.listdir(self.path) if file.endswith(".jpg") and file.split("_")[1] == '0' and file.split("_")[2].split(".")[0] == '1']
        self.validation_female = [file for file in os.listdir(self.path) if file.endswith(".jpg") and file.split("_")[1] == '1' and file.split("_")[2].split(".")[0] == '1']

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
        for file in self.training_male +  self.training_female +  self.validation_male + self.validation_female:
            gender,purpose = "",""
            image = Image.load_image_with_extension(self.path + file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is None: break
            elif file.split("_")[1] == '0' and file.split("_")[2].split(".")[0] == '0':
                gender = "male"
                purpose = "training"
            elif file.split("_")[1] == '1' and file.split("_")[2].split(".")[0] == '0':
                gender = "female"
                purpose = "training"
            elif file.split("_")[1] == '0' and file.split("_")[2].split(".")[0] == '1':
                gender = "male"
                purpose = "validation"
            elif file.split("_")[1] == '1' and file.split("_")[2].split(".")[0] == '1':
                gender = "female"
                purpose = "validation"

            self.images.append(image)
            self.genders.append(gender)
            self.purposes.append(purpose)

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
