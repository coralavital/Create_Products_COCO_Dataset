import os
from PIL import Image
import datetime
import json
import math

import pandas as pd
from transformers import DetrFeatureExtractor, DetrModel
import json 

metadata = {
    "train": pd.read_csv('data/metadata_train.csv'),
    "val": pd.read_csv('data/metadata_val.csv')
}

for keyword in ["val", "train"]:
    # prepare json for training:
    region_json = {}
    metadata_obj = metadata[keyword]
    for index, row in metadata_obj.iterrows():
        if row['filename'] not in region_json:
            region_json[row['filename']] = {
                "fileref": "",
                "size": row['w']*row['h'],
                "filename": row['filename'],
                "base64_img_data": "",
                "file_attributes": {},
                "regions": {
                    str(index): {
                        "region_attributes": {
                            "label": row["class"]
                        },
                        "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [
                            row["xmin"],
                            row["xmin"],
                            row["xmax"],
                            row["xmax"],
                        ],
                        "all_points_y": [
                            row["ymin"],
                            row["ymax"],
                            row["ymin"],
                            row["ymax"],
                        ]
                        }
                    }
                }
            }
        else :
            region_json[row['filename']]["regions"][str(index)] = {
                "region_attributes": {
                    "label": row["class"]
                },
                "shape_attributes": {
                        "all_points_x": [
                            row["xmin"],
                            row["xmin"],
                            row["xmax"],
                            row["xmax"],
                        ],
                        "all_points_y": [
                            row["ymin"],
                            row["ymax"],
                            row["ymax"],
                            row["ymin"],
                        ]
                        }
            }

    # Serializing json
    json_object = json.dumps(region_json, indent=4)
    
    # Writing to sample.json
    with open("data/pics/{}/region_data.json".format(keyword), "w") as outfile:
        outfile.write(json_object)
        
        


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def GetAreaOfPolyGon(points_x, points_y):
    points = []
    for index in range(len(points_x)):
        points.append(Point(points_x[index], points_y[index]))
    area = 0
    if len(points) < 3:

        raise Exception("error")

    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[1]
        p3 = points[2]

        vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
        vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)

        vecMult = vecp1p2.x * vecp2p3.y - vecp1p2.y * vecp2p3.x
        sign = 0
        if vecMult > 0:
            sign = 1
        elif vecMult < 0:
            sign = -1

        triArea = GetAreaOfTriangle(p1, p2, p3) * sign
        area += triArea
    return abs(area)


def GetAreaOfTriangle(p1, p2, p3):

    area = 0
    p1p2 = GetLineLength(p1, p2)
    p2p3 = GetLineLength(p2, p3)
    p3p1 = GetLineLength(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)
    area = math.sqrt(area)
    return area


def GetLineLength(p1, p2):

    length = math.pow((p1.x - p2.x), 2) + math.pow((p1.y - p2.y), 2)
    length = math.sqrt(length)
    return length


def main():
    points = []
    x = [1, 0, 0, 1]
    y = [0, 0, 1, 1]
    for index in range(len(x)):
        points.append(Point(x[index], y[index]))

    area = GetAreaOfPolyGon(points)
    print(area)


def create_image_info(
    image_id,
    file_name,
    image_size,
    date_captured=datetime.datetime.utcnow().isoformat(" "),
    license_id=1,
    coco_url="",
    flickr_url="",
):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }

    return image_info


def create_annotation_info(
    annotation_id, image_id, category_id, is_crowd, area, bounding_box, segmentation
):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation,  # [polygon]
    }

    return annotation_info


def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]


def convert(
    imgdir,
    annpath,
    categories=None,
    super_categories=None,
    output_file_name=None,
    first_class_index=1,  # typically, 0 or 1
):
    """
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """

    if categories is None:
        categories = ["rib", "clavicle"]

    default_category = categories[0]

    category_dict = dict()
    for (cat_id, cat_name) in enumerate(categories, start=first_class_index):
        category_dict[cat_name] = cat_id

    if super_categories is None:
        default_super_category = "bone"
        super_categories = [default_super_category for _ in categories]

    coco_output = {}
    coco_output["info"] = {
        "description": "Example Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2018,
        "contributor": "waspinator",
        "date_created": datetime.datetime.utcnow().isoformat(" "),
    }
    coco_output["licenses"] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        }
    ]
    coco_output["categories"] = [
        {
            "id": category_dict[cat_name],
            "name": cat_name,
            "supercategory": super_categories[0],
        }
        for (cat_name) in categories
    ]
    coco_output["images"] = []
    coco_output["annotations"] = []

    ann = json.load(open(annpath))
    # annotations id start from zero
    ann_id = 0
    # in VIA annotations, keys are image name
    for img_id, key in enumerate(ann.keys()):

        filename = ann[key]["filename"]
        img = Image.open(imgdir + filename)
        # make image info and storage it in coco_output['images']
        image_info = create_image_info(
            img_id, os.path.basename(filename), image_size=img.size
        )
        # Caveat: image shapes are conventionally (height, width) whereas image sizes are conventionally (width, height)
        # References:
        # -   https://note.nkmk.me/en/python-opencv-pillow-image-size/
        # -   https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/detection_utils.py#L189
        coco_output["images"].append(image_info)
        regions = ann[key]["regions"]
        # for one image ,there are many regions,they share the same img id
        for region in regions:
            region_attributes = regions[region]["region_attributes"]
            try:
                cat_name = region_attributes["label"]
            except KeyError:
                cat_name = default_category
            try:
                cat_id = category_dict[cat_name]
            except KeyError:
                print("Skipping unknown category {} in {}".format(cat_name, filename))
                continue
            iscrowd = 0
            shape_attributes = regions[region]["shape_attributes"]
            points_x = shape_attributes["all_points_x"]
            points_y = shape_attributes["all_points_y"]
            area = GetAreaOfPolyGon(points_x, points_y)
            min_x = min(points_x)
            max_x = max(points_x)
            min_y = min(points_y)
            max_y = max(points_y)
            box = [min_x, min_y, max_x - min_x, max_y - min_y]
            segmentation = get_segmenation(points_x, points_y)
            # make annotations info and storage it in coco_output['annotations']
            ann_info = create_annotation_info(
                ann_id, img_id, cat_id, iscrowd, area, box, segmentation
            )
            coco_output["annotations"].append(ann_info)
            ann_id = ann_id + 1

    if output_file_name is not None:
        print("Saving to {}".format(output_file_name))

        with open(output_file_name, "w") as f:
            json.dump(coco_output, f)

    return coco_output

def via2coco():
    data_path = 'custom/'

    first_class_index = 0
    for keyword in ['train2017', 'val2017']:

        input_dir = data_path + keyword + '/'
        input_json = input_dir + 'region_data.json'
        categories = ['butter', 'cottage', 'milk','mustard', 'cream']
        super_categories = ['N/A']
        output_json = input_dir + 'output.json'

        print('Converting {} from VIA format to COCO format'.format(input_json))

        coco_dict = convert(
            imgdir=input_dir,
            annpath=input_json,
            categories=categories,
            super_categories=super_categories,
            output_file_name=output_json,
            first_class_index=first_class_index,
        )

