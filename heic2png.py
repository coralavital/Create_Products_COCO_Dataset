from PIL import Image
import pillow_heif
import os
i=1
for filename in os.listdir('sweetCream'):
    print(filename)
    heif_file = pillow_heif.read_heif('/Users/coralavital/Desktop/VSCODE/Custom_Coco_Dataset/images/sweetCream/IMG_1713.HEIC')
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
    
    )

    image.save(f"./cream{i}.png", format("png"))
    i=i+1