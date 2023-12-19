from detectron2.engine import DefaultTrainer, default_setup
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
import os
import time
import json


'''
def main():
    # configuration load
    cfg = get_cfg()
    cfg.merge_from_file(
        "./configs/Base-RCNN-FPN.yaml"
    )
    cfg.DATASETS.TRAIN = ("base_birdnet2_train",)
    cfg.DATASETS.TEST = ("base_birdnet2_val", )
    cfg.MODEL.PIXEL_MEAN = [0.0, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    
    print("Registering", cfg.DATASETS.TRAIN[0])
    train_path = "datasets/bv_kitti/annotations/train_coco_base.json"
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, train_path, "")

    print("Registering", cfg.DATASETS.TEST[0])
    valid_path = "datasets/bv_kitti/annotations/valid_coco_base.json"
    register_coco_instances(cfg.DATASETS.TEST[0], {}, valid_path, "")
    default_setup(cfg, None)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 

    # TRAINING MODEL
    # Set False here to take a pth model for starting, else it will take a pkl or the last pth if exists
    trainer.resume_or_load(resume=False) 
    trainer.train()

if __name__ == "__main__":
    main()
'''















#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


from enum import Enum

class DatasetTypes(Enum):
    BASE_BIRDNET = ('datasets/bv_kitti/annotations/train_coco_base.json', 'datasets/bv_kitti/annotations/valid_coco_base.json', 3)
    PLANE_BIRDNET = ('datasets/bv_kitti/annotations/train_coco_plane_floor.json','datasets/bv_kitti/annotations/valid_coco_plane_floor.json', 7)
    PLANE_NORM_BIRDNET = ('datasets/bv_kitti/annotations/train_coco_plane_norm.json','datasets/bv_kitti/annotations/valid_coco_plane_norm.json', 7)
    MEAN_BIRDNET = ('datasets/bv_kitti/annotations/train_coco_mean.json','datasets/bv_kitti/annotations/valid_coco_mean.json', 6)
    STACK_BIRDNET = ('datasets/bv_kitti/annotations/train_coco_stack_half.json','datasets/bv_kitti/annotations/valid_coco_stack_half.json', 6)
    STACK_ONE_BIRDNET = ('datasets/bv_kitti/annotations/train_coco_stack_one.json','datasets/bv_kitti/annotations/valid_coco_stack_one.json', 6)
    def __init__(self, train_json, test_json, channels):
        self.val = train_json, test_json, channels
    def __str__(self):
        return f"{self.name}"
    @staticmethod
    def get_type(t):
        return {str(member): member for member in DatasetTypes}[t.upper()]


class SparseTypes(Enum):
    HEIGHT_THRESH_4 = ('height')
    COUNT_THRESH_3 = ('count')
    DENSITY_THRESH_8 = ('density')
    NONE = ('NONE')
    def __init__(self, val):
        self.val = (val,)
    def __str__(self):
        return f"{self.name}"
    @staticmethod
    def get_type(t):
        return {str(member): member for member in SparseTypes}[t.upper()]

def main(args, out_name):
    # TODO : adjust config file to match birdnet, run overnight...
    # TODO : make system that catches divirgence and retries with samller
    # ARGS: 
    #    type of sparsity
    #    data set
    #    whether to use WEIGHTS
    #    learning rate
    sparse_block_size = args.sparse_block_size
    sparse_type = SparseTypes.get_type(args.sparse_type)
    train_dataset_json, test_dataset_json, num_channels = DatasetTypes.get_type(args.bev_dataset).val
    print("HITTING DATASET:", train_dataset_json, test_dataset_json)
    use_weights = args.use_weights
    bev_lr = args.bev_lr

    # cfg = setup(args)
    # configuration load
    cfg = get_cfg()
    cfg.merge_from_file(
        "./configs/Base-RCNN-FPN.yaml"
    )
    cfg.DATASETS.TRAIN = ("birdnet2_train",)
    cfg.DATASETS.TEST = ("birdnet2_val",) # ("base_birdnet2_val", )
    cfg.MODEL.PIXEL_MEAN = [0.0] * num_channels
    cfg.MODEL.PIXEL_STD = [1.0] * num_channels
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl" if use_weights else ""
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    
    cfg.INPUT.SPARSE.ENABLED = sparse_type != SparseTypes.NONE
    cfg.INPUT.SPARSE.TYPE = sparse_type.val[0] # SHOULD MATCH WITH COCO FILE DICT
    cfg.INPUT.SPARSE.BLOCK_SIZE = sparse_block_size
    cfg.INPUT.SPARSE.BLOCK_TYPE = "count_2"
    cfg.SOLVER.BASE_LR = bev_lr * (1.0 if use_weights else 0.25)

    cfg.MODEL.META_ARCHITECTURE = "SparseGeneralizedRCNN" if cfg.INPUT.SPARSE.ENABLED else "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_sparse_mini_fpn_backbone" if cfg.INPUT.SPARSE.ENABLED else "build_resnet_fpn_backbone"

    file_name = f"OUTPUT_sparse_{args.sparse_block_size}_{args.sparse_type}_{args.bev_dataset}_"
    file_name += f"{'weights' if args.use_weights else 'no_weights'}_{args.bev_lr}.pth"
    cfg.OUTPUT_DIR = "output" + out_name

    print("Registering", cfg.DATASETS.TRAIN[0], "with", train_dataset_json)
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, train_dataset_json, "")

    if cfg.DATASETS.TRAIN[0] != cfg.DATASETS.TEST[0]:
        print("Registering", cfg.DATASETS.TEST[0], "with", test_dataset_json)
        register_coco_instances(cfg.DATASETS.TEST[0], {}, test_dataset_json, "")
    
    cfg.freeze()
    default_setup(cfg, None)
    print(cfg)
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    train_st_time = time.time()
    trainer.resume_or_load(resume=False)
    trainer.train()
    trained_model = trainer.model
    train_time = time.time() - train_st_time

    test_st_time = time.time()
    res = trainer.test(cfg, trained_model)
    test_time = time.time() - test_st_time

    with open(cfg.OUTPUT_DIR + "/out.log", 'w') as f:
        json.dump({"bbox": res.get('bbox'), "times": {"test_time": test_time, "train_time": train_time}}, f)
    
    wandb.login()
    wandb.init(project='BEV_PROJECT', name=name, config=vars(args))
    wandb.log(res.get('bbox'))
    wandb.log({"test_time": test_time, "train_time": train_time})
    return res

import wandb

if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--sparse_block_size', type=int, default=64, help="Size of sparse block, must be power 2")
    args.add_argument('--sparse_type', type=str, default="NONE", help="HEIGHT_THRESH_4, COUNT_THRESH_3, DENSITY_THRESH_8")
    args.add_argument('--bev_dataset', type=str, default="BASE_BIRDNET", help="BASE_BIRDNET, PLANE_BIRDNET, MEAN_BIRDNET, STACK_BIRDNET")
    args.add_argument('--use_weights', type=bool, default=True, help="Use pretrained weights")
    args.add_argument('--bev_lr', type=float, default=0.02, help="Model training LR")
    args = args.parse_args()

    name = '_'.join(["bevdataset", str(args.bev_dataset), "bevlr", str(args.bev_lr)])
    name += '_'.join(["sparse_type", str(args.sparse_type), "sparse_block_size", str(args.sparse_block_size)])

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,name),
    )
