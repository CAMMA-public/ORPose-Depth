import cv2
import numpy as np
from .eval_pck_2dpose import coco_to_camma


class VisUtils(object):
    def __init__(self, width=640, height=480):
        super(VisUtils, self).__init__()
        self.intensity_val = 150
        self.width = width
        self.height = height
        self.camma_part_names = [
            "head",
            "neck",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ]
        self.camma_colors = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
        self.camma_pairs = [
            [1, 2],
            [2, 4],
            [4, 8],
            [4, 6],
            [8, 10],
            [2, 3],
            [3, 5],
            [3, 7],
            [7, 9],
            [5, 6],
        ]
        self.camma_colors_skeleton = ["y", "g", "g", "g", "g", "m", "m", "m", "m", "m"]
        self.cc = {
            "r": (0, 0, self.intensity_val),
            "g": (0, self.intensity_val, 0),
            "b": (self.intensity_val, 0, 0),
            "c": (self.intensity_val, self.intensity_val, 0),
            "m": (self.intensity_val, 0, self.intensity_val),
            "y": (0, self.intensity_val, self.intensity_val),
            "w": (self.intensity_val, self.intensity_val, self.intensity_val),
            "k": (0, 0, 0),
            "t1": (205, 97, 85),
            "t2": (33, 97, 140),
            "t3": (23, 165, 137),
            "t4": (125, 102, 8),
            "t5": (230, 126, 34),
            "t6": (211, 84, 0),
            "t7": (52, 73, 94),
            "t8": (102, 255, 153),
            "t9": (51, 0, 204),
            "t10": (255, 0, 204),
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (25, 25)
        self.fontScale = 0.7
        self.color = (255, 255, 255) # white
        self.thickness = 2           

    def render(self, im, kpt_dict=None, title = "", apply_color_map=True, bgr2rgb=True):
        if apply_color_map:
            min1, max1 = np.min(im), np.max(im)
            im = np.uint8(np.float32(im) * (255.0 / (max1 - min1)) - min1)
            im = cv2.applyColorMap(np.uint8(im), cv2.COLORMAP_OCEAN)
        if bgr2rgb:
            b, g, r = cv2.split(im)
            im = cv2.merge([r, g, b])
        im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        if kpt_dict:
            for ann in kpt_dict:
                arr = np.array(ann["keypoints"]).reshape(-1, 3)[:, :2]
                pose = coco_to_camma(arr)
                for idx in range(len(self.camma_colors_skeleton)):
                    pt1 = (
                        int(np.clip(pose[self.camma_pairs[idx][0] - 1, 0], 0, self.width)),
                        int(np.clip(pose[self.camma_pairs[idx][0] - 1, 1], 0, self.height)),
                    )
                    pt2 = (
                        int(np.clip(pose[self.camma_pairs[idx][1] - 1, 0], 0, self.width)),
                        int(np.clip(pose[self.camma_pairs[idx][1] - 1, 1], 0, self.height)),
                    )
                    if 0 not in pt1 + pt2:
                        cv2.line(im, pt1, pt2, self.cc[self.camma_colors_skeleton[idx]], 3, cv2.LINE_AA)
                """ draw the skelton points """
                for idx_c, color in enumerate(self.camma_colors):
                    pt = (
                        int(np.clip(pose[idx_c, 0], 0, self.width)),
                        int(np.clip(pose[idx_c, 1], 0, self.height)),
                    )
                    if 0 not in pt:
                        cv2.circle(im, pt, 3, self.cc[color], 2, cv2.LINE_AA)
        if title:         
            im = cv2.putText(im, title, self.org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        return im

