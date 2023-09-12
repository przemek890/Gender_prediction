from PIL import Image
import cv2
import os
######################
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
def resize_image(image_paths,x=52,y=52):
    for path_ in image_paths:
        image = Image.open(path_)
        cropped_image = image.resize((x, y))
        cropped_image.save(path_)
def check_channels(image_paths):
    image = cv2.imread(os.getcwd() + '/../Dataset/Training/female/131422.jpg')
    channels = cv2.split(image)
    num_channels = len(channels)
    print(f'Ilość kanałów na obrazie: {num_channels}')

    for path_ in image_paths:
        image_ = cv2.imread(path_)
        channels_ = cv2.split(image_)
        num_channels_ = len(channels_)
        if num_channels_ is not num_channels:
                print(f'Obraz {path_} z odstajaca liczba kanałów: {num_channels_}')
def remove_double_jpg(image_paths):
    for path in image_paths:
        if path.endswith(".jpg.jpg"):
            path_spl = path.split('.')[:-1]  # Poprawiony fragment
            new_path = '.'.join(path_spl)
            os.rename(path, new_path)
def is_rgb_image(image_paths):
    for path in image_paths:
        image = Image.open(path)
        if image.mode == 'RGB':
            return True
        else:
            print(f'Zdjęcie {path} nie jest formatu RGB')
def rgb_to_grayscale(image_paths):
    for path in image_paths:
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(path, gray_image)

path = ["Dataset/Training/female","Dataset/Training/male","Dataset/Validation/female","Dataset/Validation/male"]
for i in path:
    image_paths = os.listdir(os.getcwd() + "/../" + i)
    paths = [os.getcwd() + "/../" + i + "/" + x for x in image_paths if x.endswith(".jpg")]
    # remove_double_jpg(paths)
    # resize_image(paths)
    # check_shape(paths)
    # rgb_to_grayscale(paths)
    # check_channels(paths)
    # is_rgb_image(paths)



