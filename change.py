import os
import sys
import argparse
from PIL import Image

l = sys.argv
folder = l[1]
image_files = os.listdir(folder)

for image in image_files:
    file_name, file_extension = os.path.splitext(image)
    if file_extension != '.jpg':
        os.rename(folder+image, folder+file_name+'.jpg')
        im = Image.open(folder+image)
        im.save(folder+file_name+'.jpg','JPEG')