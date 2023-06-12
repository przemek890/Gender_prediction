import Src.Image as Im
import Src.Charts as Ct
import Src.Model as Md

""""""""
"""Image"""
image = Im.Image()
image.check_loading_correctness()
image.select_people()

"""Charts"""
charts = Ct.Charts(image.getter_df())
charts.age_distribution()
charts.gender_distribution()
charts.sample()

"""Model"""
model = Md.Model(charts.getter_df())
model.Train_Test_Split()

# model.Build_Age_Model()
# Md.Model.Age_learning_chart(model.age_history)
# model.Save_age_model()

model.Build_Gender_Model()
Md.Model.Gender_learning_chart(model.gender_history)
model.Save_gender_model()