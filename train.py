from ultralytics import YOLO
from ultralytics.data.converter import convert_coco


import os

def create_train_txt(directory, output_file='train.txt'):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    relative_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                rel_path = os.path.join('./images/test/', file)
                relative_paths.append(rel_path)
    with open(output_file, 'w') as f:
        for path in relative_paths:
            f.write(path + '\n')
    return relative_paths

#create_train_txt("/media/genadiy/C/data/VisDrone/images/test/", output_file='/media/genadiy/C/data/VisDrone/test.txt')


if False:
    convert_coco(
    labels_dir="/media/genadiy/C/data/coco/annotations",
    save_dir="yolo_labels/",
    use_segments=False,  # Set True for segmentation tasks
    use_keypoints=False,
    cls91to80=False  # Map 91 COCO classes to original 80 classes
)

# Load a model
#model = YOLO("yolo11n.yaml").load("./chkpts/yolo11n.pt")  # build from YAML and transfer weights
model = YOLO('ultralytics/cfg/models/11/pc-yolo11s_.yaml')
#model = YOLO('ultralytics/cfg/models/11/standard-yolo11s.yaml')

# Train the model
results = model.train(
    data="/media/genadiy/C/data/VisDrone/data.yaml", 
    epochs=100, 
    deterministic=False,
    imgsz=240)

#metrics = model.val(data="coco2.yaml")
