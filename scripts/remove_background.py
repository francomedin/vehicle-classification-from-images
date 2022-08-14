"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import cv2
import os
from utils import detection


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.
        data/data_ims_v1
    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
   

    for dirpath, _, files in os.walk(data_folder):
        for filename in files:

            source_path = os.path.join(dirpath, filename)
            img = cv2.imread(source_path)

            # Crop image
            print("Getting coordinates...")
            left, top, right, bottom = detection.get_vehicle_coordinates(img)
            img_croped = img[top:bottom, left:right]

            # Creating subfolders
            if not os.path.exists(output_data_folder):
                os.makedirs(output_data_folder)
            # if not os.path.exists(os.path.join(output_data_folder, 'train')):
            class_name = dirpath.split("/")[-1]
            image_name = filename
            image_label = dirpath.split("/")[-2]

            subset_folder = os.path.join(output_data_folder, image_label)
            if not os.path.exists(subset_folder):
                os.makedirs(subset_folder)
            # create the class folder
            class_folder = os.path.join(subset_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            # copy the image to the new folder structure
            cv2.imwrite(os.path.join(class_folder, image_name), img_croped)
    print("All images are copied")


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
