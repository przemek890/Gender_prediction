from setuptools import setup, find_packages

setup(
    name='Age_prediction_app',
    version='1.0',
    description="An application for predicting a person's age based on camera input",
    author='przemek890',
    packages=find_packages()
)

# """Trimmed age"""
# bins = pd.cut(df['Ages'], self.bins)
# counts = bins.value_counts()
# bins_to_trim = counts[counts > self.c_samp].index
#
# trimmed_data = []
#
# for bin_value in bins.unique():
#     if bin_value in bins_to_trim:
#         bin_samples = self.df[bins == bin_value].sample(n=self.c_samp, random_state=42)
#         trimmed_data.append(bin_samples)
#     else:
#         trimmed_data.append(self.df.loc[bins == bin_value])
#
# trimmed_df = pd.concat(trimmed_data)
# self.df = trimmed_df