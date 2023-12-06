"""
This script performs color separation and nuclei segmentation on multiple images in a directory.

Usage:
- Provide the path to a directory containing TMA core images.
- Results are saved in separate directories labeled inside the 'results' directory.

"""

from matplotlib import pyplot as plt
import os
from skimage import io
import time
from color_separate_functions import color_separate_only  #Color separation only for this exercise


# Give path to all cores extracted from TMA
path = "dir_for_core_images/"
base_dir_name = os.path.dirname(path).split("/")[0]
results_dir_name = "results"

if not os.path.isdir(base_dir_name + "/" + results_dir_name + "/"):
    os.mkdir(base_dir_name + "/" + results_dir_name + "/")

for image_name in os.listdir(path):  # iterate through each file to perform some action
    # print(image_name)
    start_time = time.time()
    print("Starting the color separation and nuclei detection process for image - ", image_name)

    file_name = image_name.split(".")[0]

    image_path = path + image_name

    # Create directory to save results from a specific image
    if not os.path.isdir(base_dir_name + "/" + results_dir_name + "/" + file_name + "/"):
        os.mkdir(base_dir_name + "/" + results_dir_name + "/" + file_name + "/")

    image = io.imread(image_path)
    orig_image, H, brown = color_separate_only(image)

    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_original_image.png",
               orig_image)
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_H_Image.png", H)
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_Brown_image.png", brown)

    end_time = time.time()

    print("Finished color separation of image ", image_name, " in ", (end_time - start_time), " seconds")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
