import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
""""""
class Charts:
    def __init__(self, df):
        self.df = df[(df["Ages"] >= 15) & (df["Ages"] <= 80)]  # Obetnij dane na przedział: [15,80]
        self.c_samp = 10000 # Obcinanie bins
        self.bins = 30  # ilosc bins

        """Trimmed Gender"""
        men = self.df[self.df["Genders"] == 0]
        women = self.df[self.df["Genders"] == 1]
        n_samples = min(len(men), len(women))
        men = men.sample(n=n_samples, random_state=42)
        women = women.sample(n=n_samples, random_state=42)
        self.df = pd.concat([men, women], axis=0)

        print("Final samples: ",len(self.df))

    def age_distribution(self):
        sns.set_theme()
        sns.histplot(data=self.df, x='Ages', kde=True, bins=self.bins)
        plt.title("Age distribution in the population")
        plt.savefig("./Analysis/age_distribution.png")
        plt.show()

    def gender_distribution(self):
        sns.set_theme()
        sns.countplot(data=self.df, x='Genders')
        plt.xlabel('Gender')
        plt.ylabel('Quantity')
        plt.title("Gender distribution in the population")
        plt.gca().set_xticks([0, 1])
        plt.gca().set_xticklabels(['Male', 'Female'])
        plt.savefig("./Analysis/gender_distribution.png")
        plt.show()

    def sample(self):
        idx = random.randint(0, len(self.df) - 1)
        img = self.df.iloc[idx]
        plt.imshow(img["Images"])
        plt.set_cmap('gray')
        plt.savefig("./Analysis/sample.png")
        plt.show()
        print("Example image info:: ""Gender:", img["Genders"], "Age:", img["Ages"])
        print("Example shape: ",img["Images"].shape)

    def getter_df(self):
        return self.df