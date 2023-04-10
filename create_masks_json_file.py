import json
import os
from PIL import Image 
import json, operator

dictionary1 = {}

directory = r'/Users/coralavital/Desktop/VSCODE/Custom_Coco_Dataset/images/butter/masks'

filelist = os.listdir(directory)
names = []
for image in filelist:
    names.append(image.replace("butter", ""))
filelist = sorted(names,key=lambda x: int(os.path.splitext(x)[0]))
print(filelist)

for filename in filelist:
    imgObj = {
            "mask": "masks/butter"+filename,
            "color_categories": {
                "(0, 0, 255)": {"category": "butter", "super_category": "null"}
            }
    }
    dictionary1["images/butter"+filename] = imgObj
    

# Data to be written
dictionary2 = {
    "masks": dictionary1,
    "super_categories": 
    {
        "null": ["butter"]
    }
}

# Serializing json
json_object = json.dumps(dictionary2, indent=4)

# Writing to sample.json
with open("mask_definitions.json", "w") as outfile:
    outfile.write(json_object)

