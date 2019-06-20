import json

from shutil import copyfile

from os.path import join

json_file = "/save/2017018/bdegue01/datasets/GTA_dataset/dict_conversion.json"

train_path = "/save/2017018/bdegue01/datasets/GTA_dataset/original_images/train/img/"
validation_path = "/save/2017018/bdegue01/datasets/GTA_dataset/original_images/val/img/"
output_path = "/save/2017018/bdegue01/datasets/GTA_dataset/original_images_dataset/"

with open(json_file) as file:
    data = json.load(file)

data = data[0]

for key in data:
    _, val_train, _, folder, file_name = data[key].split("/")
    file_name = file_name.replace(".png", ".jpg")
    if val_train == "val":
        file_path = join(validation_path, folder, file_name)
    elif val_train == "train":
        file_path = join(train_path, folder, file_name)
    else:
        print("Error, val_train not correct : {}".format(val_train))
    
    copyfile(file_path, join(output_path, key.replace(".png", ".jpg")))
