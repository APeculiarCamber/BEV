KITTI DATASET:
    - Download the KITTI LiDAR dataset (https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
    - Place the labels_2 files into datasets/bv_kitti/label
        - Such that it has sub-directory /training/label_2/
    - Place the velodyne_3d_points files into datasets/bv_kitti/velodyne_3d_points
        - Such that it has sub-directories /testing/ and /training/
    - Place the calib files into datasets/bv_kitti/velodyne_3d_points
        - Such that it has sub-directories /testing/ and /training/
    - Download the BirdNet BEV train-test splits (https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz, from https://github.com/AlejandroBarrera/birdnet2)
    - Place the splits .txt files into datasets/bv_kitti/lists

BEV Generation:
    - Run python point_cloud_to_bev.py
        - To generate ALL encodings, this code take some time
    - Run python convert_kitti_to_coco.py
        - To generate all coco annotation json files for each

Training:
    - The base training CONFIG file used is configs/Base-RCNN-FPN.yaml
        - Modify this file as needed
    - To run training, call 'python birdnet/train.py' from the 'detectron2' directory
        - Before running, note the commmand line arguments in the if __name__ == "__main__" BLOCK_SIZE



Primary modified or created files:
    - birdnet/train.py : custom training code using DefaultTrainer
    - detectron2/config/default.py : created new CFG nodes for block sparsity
    - detectron2/data/dataset_mapper.py : Refactored to allow for loading Coco datasets with any number of feature channels using multiple PNGs
    - detectron2/data/datasets/coco.py : Refactored to accomodate sparse mask data in Coco JSON files and variable channel data in Coco JSON files
    - detectron2/modeling/backbone/fpn.py : Implementation of Block Sparse FPN
    - detectron2/modeling/meta_arch/rcnn.py : Implementation of Block Sparse RCNN
    - root/points_cloud_to_bev.py : converts LiDAR data at datasets/bv_kitti/velodyne_3d_points to BEV images
    - root/convert_kitti_to_coco.py : Generates COCO-style JSON file for BEV image datasets