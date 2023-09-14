import Src.Image as Im
import Src.Charts as Ct
import Src.Model as Md
""""""""""""""""""""""""
"""Image"""
image = Im.Image()
image.make_DataFrame()
image.check_loading_correctness()
#
"""Charts"""
charts = Ct.Charts(image.getter_df)
charts.gender_distribution()
charts.sample()
#
"""Gender_Model"""
model = Md.Gender_Model(charts.getter_df)
model.Build_Gender_Model()
model.Loss_accuracy_charts()
model.Save_model('gender_model.pth')
