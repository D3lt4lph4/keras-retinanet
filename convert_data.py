csv_file = "/save/2017018/bdegue01/datasets/GTA_dataset/datasets/miisst/boxes_miisst_val_original.csv"

ratio_x = 1920 / 2048
ratio_y = 1080 / 1024

with open(csv_file) as file:
    lines = file.readlines()

with open(csv_file + "_1", "w") as file_1:
    for line in lines:
        path, x1, y1, x2, y2, class_object = line.split(",")

        if path.startswith("/save/2017018/bdegue01/datasets/MIISST_ca") or x1 == "":
            file_1.write(line)
            continue

        x1, y1, x2, y2 = int(x1) * ratio_x, int(y1) * ratio_y, int(x2) * ratio_x, int(y2) * ratio_y
        file_1.write("{},{},{},{},{},{}".format(path, int(x1), int(y1), int(x2), int(y2), class_object))
