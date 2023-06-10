import pandas as pd
import os
import cv2
""""""""
class Image:
    def __init__(self):
        self.path = "/Users/przemek899/desktop/Age_prediction_app/UTKFace/"
        self.files = [file for file in os.listdir(self.path) if file.lower().endswith(".jpg")]
        self.size = len(self.files)
        self.images = []
        self.ages = []
        self.genders = []
        self.max_sample =  self.size
    def select_people(self):
        counter = 0
        for file in self.files:
            counter += 1
            image = self.load_image_with_extension(self.path + file)
            if image is not None:
                if counter > self.max_sample:
                    break
                image = cv2.resize(image, dsize=(128, 128))
                image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.images.append(image)
                split_var = file.split('_')
                self.ages.append(int(split_var[0]))
                self.genders.append(int(split_var[1]))
    def check_loading_correctness(self):
        print("Total samples:", self.size)
        print(self.files[0])

    @staticmethod
    def load_image_with_extension(filename, extension=".jpg"):
        if filename.endswith(extension):
            image = cv2.imread(filename)
            if image is not None and not image.size == 0:
                return image
        return None

    def getter_df(self):
        images = pd.Series(self.images, name='Images')
        ages = pd.Series(self.ages, name='Ages')
        genders = pd.Series(self.genders, name='Genders')
        df = pd.DataFrame({'Images': images, 'Ages': ages, 'Genders': genders})
        return df
    def getter_max_sample(self):
        return self.max_sample