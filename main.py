import Src.Camera as Cm
""""""""
if __name__ == '__main__':
    """Camera"""
    # face_detector = Cm.FaceDetector()
    # face_detector.detect_faces()

    from PIL import Image
    import os

    def foo(image_paths):
        # Inicjalizacja wartości minimalnych szerokości i wysokości
        min_width = float('inf')
        min_height = float('inf')
        min_width_image = None
        min_height_image = None

        for path in image_paths:
            image = Image.open(path)

            width, height = image.size

            if width < min_width:
                min_width = width
                min_width_image = path

            if height < min_height:
                min_height = height
                min_height_image = path

        image_1 = Image.open(min_width_image)
        image_2 = Image.open(min_height_image)

        print("Obraz o najmniejszej szerokości:", min_width_image,image_1.size)
        print("Obraz o najmniejszej wysokości:", min_height_image,image_2.size)



    path = ["/Dataset/Training/female","/Dataset/Training/male","/Dataset/Validation/female","/Dataset/Validation/male"]
    for i in path:
        image_paths = os.listdir(os.getcwd() + i)
        paths = [os.getcwd() + i + "/" + x for x in image_paths if x.endswith(".jpg")]
        foo(paths)