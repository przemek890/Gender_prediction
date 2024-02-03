import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
""""""
class Charts:
    def __init__(self, df):
        self.df = df

    def gender_distribution(self):
        sns.set_theme()
        palette = sns.color_palette("pastel")
        sns.countplot(data=self.df, x='Gender', palette=palette)
        plt.xlabel('Gender')
        plt.ylabel('Quantity')
        plt.title("Gender distribution in the population")
        plt.gca().set_xticks([0, 1])
        plt.gca().set_xticklabels(['Male', 'Female'])
        plt.savefig("/Analysis/gender_distribution.png")
        plt.show()

    def sample(self):
        idx = random.randint(0, len(self.df) - 1)
        img = self.df.iloc[idx]
        plt.imshow(img["Image"])
        plt.show()
        print("Example image info:: ""Gender:", img["Gender"], )
        print("Example shape: ", img["Image"].shape)

    @property
    def getter_df(self):
        return self.df