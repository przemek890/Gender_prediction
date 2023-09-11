from PIL import Image
import os

def check_shape(image_paths):
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

# for i in path:
#     image_paths = os.listdir(os.getcwd() + "/../" + i)
#     paths = [os.getcwd() + "/../" + i + "/" + x for x in image_paths if x.endswith(".jpg")]
#     check_shape(paths)

#######################
def preprocess_image(image_path):
    for pathh in image_path:
        image = Image.open(pathh)
        gray_image = image.convert('L')
        cropped_image = gray_image.resize((52, 52))
        cropped_image.save(pathh)

for i in path:
    image_paths = os.listdir(os.getcwd() + "/../" + i)
    paths = [os.getcwd() + "/../" + i + "/" + x for x in image_paths if x.endswith(".jpg")]
    preprocess_image(paths)


