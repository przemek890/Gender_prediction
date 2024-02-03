from PIL import Image
import cv2
import os
from Src.Model import Custom_Net
import torch
import numpy as np
import random
from tabulate import tabulate
######################
def remove_double_jpg(image_paths):
    for path in image_paths:
        if path.endswith(".jpg.jpg"):
            path_spl = path.split('.')[:-1]
            new_path = '.'.join(path_spl)
            os.rename(path, new_path)
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

    print("Image with the smallest width:", min_width_image, image_1.size)
    print("Image with the smallest height:", min_height_image, image_2.size)
def resize_image(image_paths,x=52,y=52):
    for path in image_paths:
        image = Image.open(path)
        cropped_image = image.resize((x, y))
        cropped_image.save(path)
def check_channels(image_paths):
    random_image_path = random.choice(image_paths)
    image = cv2.imread(random_image_path)
    channels = cv2.split(image)
    num_channels = len(channels)
    print(f'Number of channels in the image: {num_channels}')

    for path_ in image_paths:
        image_ = cv2.imread(path_)
        channels_ = cv2.split(image_)
        num_channels_ = len(channels_)
        if num_channels_ is not num_channels:
            print(f'Image {path_} with deviating number of channels: {num_channels_}')
def is_rgb_image(image_paths):
    for path in image_paths:
        image = Image.open(path)
        if image.mode != 'RGB':
            print(f'Image {path} is not in RGB format')
            return False
    print('All images are in RGB format')
    return True
def rgb_to_grayscale(image_paths):
    for path in image_paths:
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(path, gray_image)
def rename_image(paths):
    counter = 0
    for path in paths:
        new_path = path.split("/")[:-1]
        new_path = "/".join(str(x) for x in new_path) + f"/{counter}.jpg"
        os.rename(path, new_path)
        print(new_path)
        counter += 1

def check_model(model_name="gender_model"):
    female_dir = os.getcwd() + "/../Dataset/Validation/female"
    male_dir = os.getcwd() + "/../Dataset/Validation/male"

    female_images = [os.path.join(female_dir, img) for img in os.listdir(female_dir) if img.endswith(".jpg")]
    male_images = [os.path.join(male_dir, img) for img in os.listdir(male_dir) if img.endswith(".jpg")]

    all_images = female_images + male_images

    selected_images = random.sample(all_images, 10)

    gender_model = Custom_Net()
    checkpoint = torch.load(os.getcwd() + f"/../Models/{model_name}.pth")
    gender_model.load_state_dict(checkpoint['weights'])
    gender_model.eval()

    data = []
    for path in selected_images:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = []
        image_tensor.append(np.asarray(image))

        image_tensor = torch.tensor(np.array(image_tensor) / 255.0, dtype=torch.float32).reshape(-1, 3, 52, 52)

        gender_prediction = gender_model(image_tensor)

        gender_label = "Female" if gender_prediction[0,0].item() > 0.5 else "Male"
        actual_gender = "Female" if "/female/" in path else "Male"

        data.append([path.split('/')[-1], gender_label, actual_gender])

    table = tabulate(data, headers=['Image', 'Predicted Gender', 'Actual Gender'], tablefmt='pretty')
    print(table)


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

paths = ["Dataset/Training/female","Dataset/Training/male","Dataset/Validation/female","Dataset/Validation/male"]
for path in paths:
    image_paths = os.listdir(os.getcwd() + "/../" + path)
    im_paths = [os.getcwd() + "/../" + path + "/" + x for x in image_paths if x.endswith(".jpg")]
    # remove_double_jpg(im_paths)
    # check_shape(im_paths)
    # resize_image(im_paths)
    # check_channels(im_paths)
    # is_rgb_image(im_paths)
    # rename_image(im_paths)
    # rgb_to_grayscale(im_paths) # WARNING

check_model()



