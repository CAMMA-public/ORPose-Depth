# coding: utf-8
'''
Copyright (c) University of Strasbourg. All Rights Reserved.
'''
import torch
import torchvision
import os
from PIL import Image
from pycocotools.coco import COCO

class MVORDatasetTest(torch.utils.data.Dataset):
    def __init__(self, ann_file, root):
        super(MVORDatasetTest, self).__init__()
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = sorted(list(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        path = coco.loadImgs(img_id)[0]["file_name"]
        full_path = os.path.join(self.root, path)
        img = torchvision.transforms.functional.to_tensor(Image.open(full_path))
        return img, img_id

    def __len__(self):
        return len(self.ids)

