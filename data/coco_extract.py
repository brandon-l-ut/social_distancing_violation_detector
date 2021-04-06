#
#
# Script to modify annotation files so they only include person class bounding boxes
#
#
# Assumes the following file structure
# data/
# |--- annotations/
#      |--- captions_train2014.json
#       ....

from pycocotools.coco import COCO
import json

if __name__ == "__main__":
    print("Starting Extraction")

    id_text = "person"
    annType = 'bbox'
    annFileTrain = "annotations/instances_train2014"
    annFileVal = "annotations/instances_val2014"
    annFiles = [annFileVal, annFileTrain]
    
    for file in annFiles:
        coco = COCO(file+".json")
        id_num = coco.getCatIds([id_text])
        ann_ids = coco.getAnnIds(catIds = id_num)

        # annotations of all pics with person
        anns = coco.loadAnns(ann_ids) 
        tmp_dict = {"categories": [{'supercategory': 'person', 'id': 1, 'name': 'person'}]}
        tmp_dict["annotations"] = anns

        img_ids = list({ann["image_id"] for ann in anns})
        print(len(img_ids))
        imgs = coco.loadImgs(img_ids)
        tmp_dict["images"] = imgs

        with open(file + "_person.json", "w") as fp:
            json.dump(tmp_dict, fp)
            




