import os
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread
from VOClabelcolormap import color_map
from anno import ImageAnnotation
import pandas as pd 
import cv2
import yaml

# Convert Pascal_Voc bb to Yolo
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return ((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h

def save_yolo_annotations(class_dictionary, img_ann, output_dir):
    width, height = img_ann.imsize[:2]
    print('h = ', height, 'w = ', width)
    lines = []

    for obj in img_ann.objects:
        # Object bbox in row * column format thus y * x
        ymin, xmin, ymax, xmax = obj.props.bbox
        if ((xmin == xmax) or (ymin == ymax)):
                print(f"Object {obj} in image {img_ann.imname} is too small (left upper ({xmin}, {ymin}) right bottom ({xmax}, {ymax})")
        x_center, y_center, bbox_width, bbox_height = pascal_voc_to_yolo(xmin, ymin, xmax, ymax, height, width)
        lines.append(f"{class_dictionary[obj.class_name]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # Part bboxes
        for part in obj.parts:
            # Part bbox in row * column format thus y * x
            ymin_part, xmin_part, ymax_part, xmax_part = part.props.bbox
            if ((xmin_part == xmax_part) or (ymin_part == ymax_part)):
                print(f"Part {part} from object {obj} in image {img_ann.imname} is too small (left upper ({xmin_part}, {ymin_part}) right bottom ({xmax_part}, {ymax_part})")
            x_center_part, y_center_part, bbox_width_part, bbox_height_part = pascal_voc_to_yolo(xmin_part, ymin_part, xmax_part, ymax_part, height, width)

            # Get the composite name
            composite_name = obj.class_name
            # Get the part name
            part_name = part.part_name
            # Create the class name based on the object and its part
            class_name = composite_name + '-' + part_name
            part_class_id = class_dictionary[class_name]

            lines.append(f"{part_class_id} {x_center_part:.6f} {y_center_part:.6f} {bbox_width_part:.6f} {bbox_height_part:.6f}")
            # print(lines)

    # Save to file
    basename = os.path.splitext(os.path.basename(img_ann.impath))[0]
    outpath = os.path.join(output_dir, f"{basename}.txt")
    with open(outpath, 'w') as f:
        f.write("\n".join(lines))
    # print(f"Saved: {outpath}")

with open('/data/christophe/hierarchical/pascal-part-py/voc_parts_data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

class_number = data.get("nc")
class_names = data.get("names") # sequence de noms
class_dictionary = {name: index for index, name in enumerate(class_names,0)}
index_to_class = {index: name for index, name in enumerate(class_names,0)}
with open('/data/christophe/hierarchical/pascal-part-py/PascalPart.yaml', 'w') as pascal_part_yaml_file:
    yaml_line = 'names:\n'
    for index in index_to_class:
        yaml_line = '  ' + str(index) + ' : ' + index_to_class[index] + '\n'
        pascal_part_yaml_file.write(yaml_line)

input_dir_images = '/data/christophe/hierarchical/OriginalPascalPart/Images'
input_dir_anno = '/data/christophe/hierarchical/OriginalPascalPart/Annotations_Part'
output_dir = '/data/christophe/hierarchical/OriginalPascalPart/YOLO_Annotations_Part'

images_name = []
anno_name = []

for fname_im in sorted(os.listdir(input_dir_images)):
    images_name.append(fname_im)

for fname_anno in sorted(os.listdir(input_dir_anno)):
    anno_name.append(fname_anno)

for i in range(len(images_name)):
    # print(i)
    img_ann = ImageAnnotation(os.path.join(input_dir_images, images_name[i]), os.path.join(input_dir_anno, anno_name[i]))
    path = output_dir
    save_yolo_annotations(class_dictionary, img_ann, output_dir=path)