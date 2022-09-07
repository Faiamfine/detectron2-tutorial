import sys
import os
sys.path.append(os.getcwd())
sys.path.append('../')
    

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.evaluation import (
#     CityscapesInstanceEvaluator,
#     CityscapesSemSegEvaluator,
#     COCOEvaluator,
#     COCOPanopticEvaluator,
#     DatasetEvaluators,
#     LVISEvaluator,
#     PascalVOCDetectionEvaluator,
#     SemSegEvaluator,
#     inference_on_dataset,
#     print_csv_format,
# )
import os
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image
from detectron2.engine import DefaultTrainer
pylab.rcParams['figure.figsize'] = (8.0, 10.0)# Import Libraries

def annotate():
    # I am visualizing some images in the 'val/' directory

    dataDir='/home/faii/detectron2/car_damage/val'
    dataType='COCO_val_annos'
    mul_dataType='COCO_mul_val_annos'
    annFile='{}/{}.json'.format(dataDir,dataType)
    mul_annFile='{}/{}.json'.format(dataDir,mul_dataType)
    img_dir = "/home/faii/detectron2/car_damage/img"

        # initialize coco api for instance annotations
    coco=COCO(annFile)
    mul_coco=COCO(mul_annFile)

    # display categories and supercategories

    #Single Class #Damage dataset
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories for damages: \n{}\n'.format(', '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories for damages: \n{}\n'.format(', '.join(nms)))

    #Multi Class #Parts dataset

    mul_cats = mul_coco.loadCats(mul_coco.getCatIds())
    mul_nms=[cat['name'] for cat in mul_cats]
    print('COCO categories for parts: \n{}\n'.format(', '.join(mul_nms)))

    mul_nms = set([mul_cat['supercategory'] for mul_cat in mul_cats])
    print('COCO supercategories for parts: \n{}\n'.format(', '.join(mul_nms)))

    # get all images containing 'damage' category, select one at random
    catIds = coco.getCatIds(catNms=['damage']);
    imgIds = coco.getImgIds(catIds=catIds );

    random_img_id = random.choice(imgIds)
    print("{} image id was selected at random from the {} list".format(random_img_id, imgIds))

    # Load the image
    imgId = coco.getImgIds(imgIds = [random_img_id])
    img = coco.loadImgs(imgId)[0]
    print("Image details \n",img)

    #get damage annotations
    annIds = coco.getAnnIds(imgIds=imgId,iscrowd=None)
    anns = coco.loadAnns(annIds)

    dataset_dir = "/home/faii/detectron2/car_damage"
    img_dir = "/home/faii/detectron2/car_damage/img"
    train_dir = "/home/faii/detectron2/car_damage/train"
    val_dir = "/home/faii/detectron2/car_damage/val"

    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("car_dataset_train", {}, os.path.join(dataset_dir,train_dir,"COCO_train_annos.json"), os.path.join(dataset_dir,img_dir))
    register_coco_instances("car_dataset_val", {}, os.path.join(dataset_dir,val_dir,"COCO_val_annos.json"), os.path.join(dataset_dir,img_dir))

    dataset_dicts = DatasetCatalog.get("car_dataset_train")
    metadata_dicts = MetadataCatalog.get("car_dataset_train")

annotate()