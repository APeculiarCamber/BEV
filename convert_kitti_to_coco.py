# BEV-PROJECT : CREATED, ADAPTED FROM BIRDNET

import json

from PIL import Image, ImageDraw
import os
import math
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import math as m

class bbox(object):
    ''' 3d object label '''
    def __init__(self, xmin, ymin, xmax, ymax):
        self.x_offset = xmin
        self.y_offset = ymin
        self.width = xmax-xmin
        self.height = ymax-ymin

class location(object):
    ''' 3d object label '''
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        # extract label, truncated, occluded
        self.kind_name = data[0]  # 'Car', 'Pedestrian', ...
        self.truncated = data[1]  # truncated pixel ratio [0..1]
        self.occluded = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]
        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.bbox = bbox(self.xmin, self.ymin, self.xmax, self.ymax) #np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        # extract 3d bounding box information
        self.height = data[8]  # box height
        self.width = data[9]  # box width
        self.length = data[10]  # box length (in meters)
        self.location = location(data[11], data[12], data[13])  # location (x,y,z)
        self.yaw = data[14]  # yaw angle
        if len(data) > 15:
            self.score = data[15]
    def print_object(self):
        print('kind_name, truncated, occluded, alpha: %s, %d, %d, %f' % \
              (self.kind_name, self.truncated, self.occluded, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox height,width,length: %f, %f, %f' % \
              (self.height, self.width, self.length))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.location.x, self.location.y, self.location.z, self.yaw))

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                    = K * [1|t]
        image2 coord:
            ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    '''
    def __init__(self, calib_filepath, scaling=None):
        calibs = self.read_calib_file(calib_filepath)

        eye = np.eye(4,4)
        self.p0_mat = np.eye(4, 4)
        self.p0_mat[:3, :4] = calibs["P0"].reshape(3, 4)
        self.r0_rect_mat = np.eye(4, 4)
        self.r0_rect_mat[:3, :3] = calibs["R0_rect"].reshape(3, 3)
        self.velo_to_cam_mat = np.eye(4, 4)
        self.velo_to_cam_mat[:3, :4] = calibs["Tr_velo_to_cam"].reshape(3, 4)
        
        #print(p0_mat)
        #print(r0_rect_mat)
        #print(velo_to_cam_mat)

        # Compute the inverse of ABC
        self.velo_to_cam = np.matmul(np.matmul(eye, self.r0_rect_mat), self.velo_to_cam_mat)
        self.cam_to_velo = np.linalg.inv(self.velo_to_cam)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    
    def cam2velo(self, points):
        points = self.cart2hom(points)
        return np.matmul(points, np.transpose(self.cam_to_velo))[:, :3]
    
    def camrect2velo(self, points):
        points = self.cart2hom(points)
        mat = np.matmul(np.matmul(self.p0_mat, self.r0_rect_mat), self.velo_to_cam_mat)
        mat = np.linalg.inv(self.velo_to_cam)

        return np.matmul(points, np.transpose(mat))[:, :3]

def draw_rotated_box(img, draw, center_x, center_y, width, height, rotation, im_bounds):

    # Calculate the coordinates of the four corners of the rotated rectangle
    angle_rad = rotation
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    half_width = width / 2
    half_height = height / 2

    # Calculate the rotated coordinates
    x1 = center_x + (cos_theta * half_width) - (sin_theta * half_height)
    y1 = center_y + (sin_theta * half_width) + (cos_theta * half_height)

    x2 = center_x - (cos_theta * half_width) - (sin_theta * half_height)
    y2 = center_y - (sin_theta * half_width) + (cos_theta * half_height)

    x3 = 2 * center_x - x1
    y3 = 2 * center_y - y1

    x4 = 2 * center_x - x2
    y4 = 2 * center_y - y2

    minx, maxx = min(x1, x2, x3, x4), max(x1,x2,x3,x4)
    miny, maxy = min(y1, y2, y3, y4), max(y1,y2,y3,y4)

    # Draw the rotated rectangle
    draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline='red', width=3)
    draw.rectangle([(minx, miny), (maxx, maxy)], outline='green', width=3)

    in_bounds = minx < im_bounds[0] and miny < im_bounds[1]
    in_bounds = maxx > 0 and maxy > 0 and in_bounds

    return [minx, miny, maxx-minx, maxy-miny], [x1, y1, x2, y2, x3, y3, x4, y4], in_bounds


def draw_circle(img, draw, x, y, radius, color):

    # Define the bounding box for the circle
    bounding_box = [(x - radius, y - radius), (x + radius, y + radius)]

    # Draw the circle
    draw.ellipse(bounding_box, outline=color, width=3)







# ********** PARAMS *************************************************************
cell_size=0.05
map_size=cell_size*1024.0
grid_size = map_size/cell_size














def make_coco_ann(id, image_id, cat_id, bbox, seg):
    ann = dict()
    ann['id'] = id
    ann['image_id'] = image_id
    ann['category_id'] = cat_id
    ann['bbox'] = bbox
    ann['iscrowd'] = 0
    ann['area'] = bbox[2] * bbox[3]
    ann['segmentation'] = [seg]
    return ann

def make_coco_image(image_id, im_w, im_h, channels, im_path, sp_paths):
    ann = dict()
    ann['id'] = image_id
    ann['width'] = im_w
    ann['height'] = im_h
    ann['channels'] = channels
    ann['file_name'] = im_path
    ann['sparse_names'] = sp_paths
    ann['bbox_mode'] = 1
    return ann

failed = 0
def convert_split_file(splitfile, channels, image_id, anno_id, im_dir, label_dir, calib_dir, sp_dirs):
    global failed
    name_to_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    full_coco = {'images': [], 'annotations': [],
                'categories': [{"id": 0, "name": "Car"}, {"id": 1, "name": "Pedestrian"}, {"id": 2, "name": "Cyclist"}]}

    with open(splitfile) as f:
        for line in f:
            i = line.strip()

            # IMAGES
            i = str(i).zfill(6)
            if channels <= 3:
                im_path = [os.path.abspath(f"{im_dir}/{i}_1.png")]
            else:
                im_path = [os.path.abspath(f"{im_dir}/{i}_{p}.png") for p in range(1, int(m.ceil(channels/3)) + 1)]
            
            im = Image.open(im_path[0])

            sp_paths = [(sp_name, os.path.abspath(f"{sp_dir}/{i}.png")) for (sp_name, sp_dir) in sp_dirs.items()]
            for _, sp_path in sp_paths: Image.open(sp_path).close()
            sp_paths = dict(sp_paths)

            im_w, im_h = im.size
            coco_im = make_coco_image(image_id, im_w, im_h, channels, im_path, sp_paths)

            # CALIB
            calib = Calibration(f"{calib_dir}/{i}.txt")

            # ANNOTATIONS
            old_anno_id = anno_id
            img = im
            draw = ImageDraw.Draw(img)
            with open(f"{label_dir}/{i}.txt", 'r') as label_file:
                for line in label_file:
                    splits = line.split()
                    if splits[0] not in name_to_id: continue

                    # LOAD AND CONVERT ANNOTATION
                    name = splits[0]
                    width, length, x, y, z, rot = [float(x) for x in splits[9:15]]
                    # CONVERT TO VELO SPACE

                    x, y, z = calib.cam2velo(np.array([[x, y, z]]))[0]
                    # print("VELO:", x, y, z)

                    # CONVERT TO PIXEL COORDS
                    x, y, z = [(v/cell_size) for v in (x,y,z)]
                    width,length = [(v/cell_size) for v in (width,length)]
                    xt, yt = -y, x
                    xt, yt = (grid_size/2)-xt, (grid_size/2)-yt
                    rot = (-rot - (math.pi / 2))

                    bbox, seg, in_bounds = draw_rotated_box(img, draw, xt, yt, width, length, rot, (grid_size, grid_size/2))
                    if not in_bounds: continue

                    # bbox, seg = convert_to_bbox_and_segmentation(x, y, width, length, -rot)
                    coco_ann = make_coco_ann(anno_id, image_id, name_to_id[name], bbox, seg)
                    full_coco['annotations'].append(coco_ann)
                    anno_id += 1
            
            if old_anno_id == anno_id: 
                img.save(f"out_boxed_images/_failed_{i}.png")
                failed += 1
                continue
            
            img.save(f"out_boxed_images/{image_id}.png")
            full_coco['images'].append(coco_im)
            image_id += 1
            if image_id % 100 == 0: print("At", image_id)


    return full_coco, image_id, anno_id

def convert_to_kitti(im_dir, channels, sp_dirs, label_dir, calib_dir, out_fn, trainsplit, valsplit, anno_type=""):

    os.makedirs(out_fn, exist_ok=True)

    image_id = 0
    anno_id = 0
    val_coco, image_id, anno_id = convert_split_file(valsplit, channels, image_id, anno_id, im_dir, label_dir, calib_dir, sp_dirs)
    train_coco, image_id, anno_id = convert_split_file(trainsplit, channels, image_id, anno_id, im_dir, label_dir, calib_dir, sp_dirs)

    with open(os.path.join(out_fn, f"train_coco{anno_type}.json"), mode='w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(out_fn, f"valid_coco{anno_type}.json"), mode='w') as f:
        json.dump(val_coco, f)


def main_convert(im_dir="detectron2/datasets/bv_kitti/image/", num_channels=3, anno_type="_base"):
    global failed
    failed = 0
    label_dir = "detectron2/datasets/bv_kitti/label/training/label_2/"
    calib_dir = "detectron2/datasets/bv_kitti/calib/training/calib/"
    tsplit_dir = "detectron2/datasets/bv_kitti/lists/trainsplit_chen.txt"
    vsplit_dir = "detectron2/datasets/bv_kitti/lists/valsplit_chen.txt"
    anno_dir = "detectron2/datasets/bv_kitti/annotations"
    sp_dirs = {"count": "detectron2/datasets/bv_kitti/sparse/count_3/", 
               "density": "detectron2/datasets/bv_kitti/sparse//density_0.08",
               "height": "detectron2/datasets/bv_kitti/sparse/height_0.4"}
    convert_to_kitti(im_dir, num_channels, sp_dirs, label_dir, calib_dir, anno_dir, tsplit_dir, vsplit_dir, anno_type=anno_type)

    print("Total failed was", failed)


if __name__ == "__main__":
    main_convert("detectron2/datasets/bv_kitti/image/stack_one", num_channels=6, anno_type="_stack_one")
    main_convert("detectron2/datasets/bv_kitti/image/birdnet", num_channels=3, anno_type="_base")
    main_convert("detectron2/datasets/bv_kitti/image/mean", num_channels=6, anno_type="_mean")
    main_convert("detectron2/datasets/bv_kitti/image/stack_half", num_channels=6, anno_type="_stack_half")
    main_convert("detectron2/datasets/bv_kitti/image/plane_floor", num_channels=7, anno_type="_plane_floor")
    main_convert("detectron2/datasets/bv_kitti/image/plane_norm", num_channels=7, anno_type="_plane_norm")
