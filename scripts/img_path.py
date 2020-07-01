import json
import os
img_root_path = '/media/VG_100K'
input_json = '../data/captions/para_paragraph_caption.json'
imgs = json.load(open(input_json, 'r'))
imgs = imgs['images']
f = open("../data/imgs_all_path.txt", "w")

for img in imgs:
    img_name = img['id']
    img_path = os.path.join(img_root_path, str(img_name)+'.jpg')
    f.write(img_path + "\n")
f.close()