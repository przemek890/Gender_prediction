import src.Image as Im
import src.Charts as Ct
""""""""

if __name__ == '__main__':
    """Image"""
    image = Im.Image()
    image.check_loading_correctness()
    image.select_people()

    """Charts"""
    charts = Ct.Charts(image.getter_df())
    charts.age_distribution()
    charts.gender_distribution()
    charts.sample()

"""Camera"""
# face_detector = fd()
# face_detector.detect_faces()

