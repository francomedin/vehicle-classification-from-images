"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...
"""
import argparse
import boto3
import os
from dotenv import load_dotenv
from pyrsistent import b
from sklearn.semi_supervised import LabelSpreading
import pandas as pd

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )

    args = parser.parse_args()

    return args


def get_data(data_folder, labels):
    # fetch credentials from env variables
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    # setup a AWS S3 client/resource
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    bucket = s3.Bucket("anyoneai-ay22-01")

    # Get labels
    if not os.path.exists(labels):

        print("Downloading car_dataset_labels.csv")

        with open(labels, "wb") as data:
            bucket.download_fileobj("training-datasets/car_dataset_labels.csv", data)
    else:
        print("car_dataset_labels.csv already exists")

    # Get Images and extract them to data_folder
    if not os.path.exists(data_folder):
        path_tgz = os.path.join("data", "car_ims.tgz")
        if not os.path.exists(path_tgz):
            print("Downloading car_ims.tgz")
            with open(path_tgz, "wb") as data:
                bucket.download_fileobj("training-datasets/car_ims.tgz", data)
        else:
            print("car_ims.tgz already exists")

        print("Extracting car_ims.tgz")
        os.system(f"tar -xzvf {path_tgz} -C data")

    else:
        print("car_ims already exists")


def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to raw images folder.

    labels : str
        Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """
   

    get_data(data_folder, labels)

    # Load labels and create train/test folders
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)
    print("Loading labels CSV file")
    df = pd.read_csv(labels)

    # iterate over each row in the CSV
    if not os.path.exists(os.path.join(output_data_folder, "train")):
        print(
            "Iterating over each row in the CSV, creating train/test and class folders"
        )
        for index, row in df.iterrows():
            class_name = row["class"]
            image_name = row["img_name"]
            image_label = row["subset"]
            subset_folder = os.path.join(output_data_folder, image_label)
            if not os.path.exists(subset_folder):
                os.makedirs(subset_folder)
            # create the class folder
            class_folder = os.path.join(subset_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            # copy the image to the new folder structure
            src = os.path.join(data_folder, image_name)
            dest = os.path.join(class_folder, image_name)
            os.link(src, dest)

        print("All images are copied")

    else:
        print("Images alerady exists")


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
