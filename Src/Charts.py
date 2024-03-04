import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.utils import resample
import numpy as np
""""""
class Charts:
    def __init__(self, df):
        self.df = df
        df_male = self.df[self.df['Gender'] == 'male']
        df_female = self.df[self.df['Gender'] == 'female']

        # Downsampling
        df_male_downsampled = resample(df_male,
                                       replace=False,
                                       n_samples=len(df_female),
                                       random_state=123)

        self.df_downsampled = pd.concat([df_male_downsampled, df_female])

    def gender_distribution(self):
        sns.set_theme()
        palette = sns.color_palette("pastel", n_colors=len(self.df_downsampled['Gender'].unique()))
        sns.countplot(data=self.df_downsampled, x='Gender', hue='Gender', palette=palette, legend=False)
        plt.xlabel('Gender')
        plt.ylabel('Quantity')
        plt.title("Gender distribution in the population")
        plt.gca().set_xticks([0, 1])
        plt.gca().set_xticklabels(['Male', 'Female'])
        plt.savefig("Analysis/gender_distribution.png")
        plt.show()

    def sample(self):
        idx = random.randint(0, len(self.df) - 1)
        img = self.df.iloc[idx]
        plt.imshow(img["Image"])
        plt.show()
        print("Example image gender info: ", img["Gender"])
        print("Example shape: ", img["Image"].shape)

    @property
    def getter_df(self):
        return self.df