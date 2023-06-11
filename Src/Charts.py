import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
""""""
class Charts:
    def __init__(self, df):
        self.df = df
        self.c_samp = 1000
        self.bins = 80

        bins = pd.cut(df['Ages'], self.bins)

        counts = bins.value_counts()
        bins_to_trim = counts[counts > self.c_samp].index

        trimmed_data = []
        for bin_value in bins.unique():
            if bin_value in bins_to_trim:
                bin_samples = df[bins == bin_value].sample(n=self.c_samp, random_state=42)
                trimmed_data.append(bin_samples)
            else:
                trimmed_data.append(df[bins == bin_value])

        trimmed_df = pd.concat(trimmed_data)
        trimmed_df = trimmed_df[trimmed_df["Ages"] <= 80]
        self.df = trimmed_df
        print("Final samples: ",len(self.df))

    def age_distribution(self):
        sns.set_theme()
        sns.histplot(data=self.df, x='Ages', kde=True, bins=self.bins)
        plt.title("Age distribution in the population")
        plt.show()
    def gender_distribution(self):
        sns.set_theme()
        sns.countplot(data=self.df, x='Genders')
        plt.xlabel('Gender')
        plt.ylabel('Quantity')
        plt.title("Gender distribution in the population")
        plt.gca().set_xticks([0, 1])
        plt.gca().set_xticklabels(['Male', 'Female'])
        plt.show()

    def sample(self):
        idx = random.randint(0, len(self.df) - 1)
        img = self.df.iloc[idx]
        plt.imshow(img["Images"])
        plt.set_cmap('gray')
        plt.show()
        print("Example image info:: ""Gender:", img["Genders"], "Age:", img["Ages"])

    def getter_df(self):
        return self.df