
import torch.multiprocessing as multiprocessing
import torch
import time
import matplotlib.pyplot as plt


import torch
import matplotlib.pyplot as plt
import numpy as np
import time

import point_cloud_dataset
def output_max_map_debug(grid, circle_radius, intersects1, intersects2):
    x = grid[:, 0, 0]
    y = grid[0, :, 1]

    # Define the circle parameters
    circle_center = (0, 0)

    # Create a Matplotlib figure
    plt.figure(figsize=(8, 8))

    # Plot the grid
    for xi in x:
        plt.plot([xi, xi], [y[0], y[-1]], color='gray', linestyle='--')
    for yi in y:
        plt.plot([x[0], x[-1]], [yi, yi], color='gray', linestyle='--')

    # Plot the circle
    circle = plt.Circle(circle_center, circle_radius, color='blue', fill=False, label='Circle')
    plt.gca().add_patch(circle)

    # Plot the points of intersection
    plt.plot(intersects1[:, 0], intersects1[:, 1], 'ro', markersize=5, label='Intersection Points 1')
    plt.plot(intersects2[:, 0], intersects2[:, 1], 'bo', markersize=3, label='Intersection Points 2')

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Set the aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the plot
    plt.grid(True)
    plt.title('Grid, Circle, and Intersection Points')
    plt.show()


def util_clamp_vec_mags(t, max_mag):
    mags = torch.norm(t, dim=1)
    exceed = mags > max_mag

    t_new = t.clone() # TODO : THIS IS COPY!
    t_new[exceed] = t[exceed] * (max_mag / mags[exceed].unsqueeze(1))
    return t_new

# ASSUMES plane_res and plane_angles are in DERGEES!!!
def make_max_map(grid_size, cell_res, h_top, h_ground, h_velo, velo_plane_res, plane_angles, h_slices=[]):
    dev = torch.device("cuda")

    x = (torch.linspace(0, grid_size - 1, grid_size, dtype=torch.float, device=dev) * cell_res)
    y = (torch.linspace(0, grid_size - 1, grid_size, dtype=torch.float, device=dev) * cell_res)
    x, y = torch.meshgrid(x, y, indexing='xy')
    cell_pos = torch.stack((x, y), dim=2)
    # TODO : assumes lidar center is 0,0,h_velo

    # compute ground and top intersection points with lidar planes, TODO : ensure that ANGLEs are correct
    plane_radians = torch.deg2rad(-plane_angles)
    clamped_radians = torch.clamp(plane_radians, min=0.00001)
    ground_radius = (h_velo - h_ground) / torch.tan(clamped_radians)
    top_radius = (h_top - h_velo) / torch.tan(clamped_radians)
    rad_select_ground = plane_radians >= 0.0
    plane_radii = torch.where(rad_select_ground, ground_radius, top_radius)
    del plane_radians, clamped_radians, ground_radius, top_radius, rad_select_ground
    print(plane_radii)

    # determine if the plane angle causes a collision with each plane
    cell_dists = cell_pos[:,:,0]**2 + cell_pos[:,:,1]**2

    # !! Computing 2 ANGLES
    # Do the computes that can be done globally now
    ray_dir1 = torch.tensor([0, 1], device=dev)
    ray_dir2 = torch.tensor([1, 0], device=dev) # TODO TODO TODO

    a1 = ray_dir1[0] ** 2 + ray_dir1[1] ** 2
    a2 = ray_dir2[0] ** 2 + ray_dir2[1] ** 2
    # B = 2 * (ray_dir[0] * (ray_origin[0] - circle_center[0]) + ray_direction[1] * (ray_origin[1] - circle_center[1]))
    #print(cell_pos.shape)
    b1 = 2 * (ray_dir1 * cell_pos).sum(dim=-1)
    b2 = 2 * (ray_dir2 * cell_pos).sum(dim=-1)
    #print(b1.shape, b2.shape)

    max_contrib = torch.zeros(grid_size, grid_size, device=dev)

    # Create a function to update the plot for each frame
    for prad in plane_radii:
        print(prad*prad)
        plane_colls = (prad*prad) >= cell_dists
        #print("Total overlappers =", plane_colls.sum())

        cell_pos_pl = cell_pos[plane_colls]
        C_pl = (cell_pos_pl**2).sum(dim=-1) - (prad ** 2)

        # CELL EDGE 1 INTERSECTION
        b_pl = b1[plane_colls]
        #print(b_pl.shape, a1.shape, C_pl.shape)
        discriminant = b_pl ** 2 - 4 * a1 * C_pl
        #print(b_pl.shape, discriminant.shape, a1.shape)
        t = (-b_pl + torch.sqrt(discriminant)) / (2 * a1) # or - sqrt(discr)
        #print(cell_pos_pl.shape, ray_dir1.shape, t.shape)
        p1 = cell_pos_pl + util_clamp_vec_mags(ray_dir1 * t.unsqueeze(-1), cell_res)

        # CELL EDGE 2 INTERSECTION
        b_pl = b2[plane_colls]
        discriminant = b_pl ** 2 - 4 * a2 * C_pl
        t = (-b_pl + torch.sqrt(discriminant)) / (2 * a2) # or - sqrt(discr)
        p2 = cell_pos_pl + util_clamp_vec_mags(ray_dir2 * t.unsqueeze(-1), cell_res)

        # COMPUTE THE ANGLE BETWEEN THE POINTS AND ADD THE MAX CONTRIB MAP
        acoll1 = torch.atan2(p1[:, 1], p1[:, 0])
        acoll2 = torch.atan2(p2[:, 1], p2[:, 0])
        lidar_contrib = torch.ceil(torch.abs(acoll1 - acoll2) / torch.deg2rad(torch.tensor(velo_plane_res)))
        max_contrib[plane_colls] += lidar_contrib

    mir_max_contrib = torch.cat((max_contrib.flip(1), max_contrib), dim=1).flip(0)
    os.makedirs("max_map", exist_ok=True)
    save_as_mask(mir_max_contrib.unsqueeze(-1), "max_map", 0)
    return mir_max_contrib.contiguous()




import math as m

'''
Assumes points encoded as x,y,z,intensity. Where z is height.
'''
def convert_points_to_bev(points, map_size, cell_size, num_channels, gpu_bev_process, gpu_params, cpu_bev_process, cpu_params, reusable_pool : multiprocessing.Pool, last_pool_async=None, channel_type=torch.float):

    dev = torch.device("cuda")
    points = points.to(dev)

    grid_size = int(map_size / cell_size)
    grid_size = (grid_size // 2, grid_size)
    min_x, min_y = 0, -map_size / 2
    max_x, max_y = map_size / 2, map_size / 2

    # Assumes indexed with y MAJOR, x MINOR
    # We want index with x MAJOR, y MINOR
    normalized_points = (points[:,:2] - torch.tensor([[min_x, min_y]], device=dev)) / torch.tensor([[max_x - min_x, max_y - min_y]], device=dev)
    normalized_points[:, 0] = 1.0 - normalized_points[:, 0]
    grid_indices = (normalized_points * torch.tensor(grid_size, device=dev)).long()
    encoded_gi = (grid_size[1] * grid_indices[:,0]) + grid_indices[:,1]
    # Get a index list that sorts the cloud into a flattened BEV
    sorted_quantized_grid_pos, sorted_indices = torch.sort(encoded_gi)

    # Use the sort indices to sort the points, and their grid positions (quantized by col-major) TODO: see above todo
    sorted_points = points[sorted_indices]
    sorted_quantized_grid_pos = sorted_quantized_grid_pos.to(torch.float)


    # This histogram bins on-lines to the higher bin (this works for our purpose but for safety we nudge grid positions a little)
    histogram = torch.histc(sorted_quantized_grid_pos + 0.1, bins=grid_size[0]*grid_size[1], min=0, max=grid_size[0]*grid_size[1])
    # Eliminate cells without points
    active_cell_indices = torch.sort(torch.nonzero(histogram).squeeze()).values
    active_histogram = histogram[active_cell_indices]
    # This cum sum represents the indices of points within a grid space (i.e. cum_hist[i]:cum_hist[i+1] are the points in cell i), we use active_cell_indices to index it
    cum_hist = torch.cat([torch.tensor([0], device=dev), torch.cumsum(active_histogram, dim=0).int()])


    # Make BEV image
    bev_flat = torch.zeros((grid_size[0]*grid_size[1], num_channels), device=dev, dtype=channel_type)

    # Apply one-shot GPU operations
    if gpu_bev_process is not None: gpu_bev_process(bev_flat, histogram, cum_hist, sorted_points, gpu_params)

    cum_hist = cum_hist.cpu().share_memory_()
    active_cell_indices = active_cell_indices.cpu().share_memory_()
    cpu_sorted_points = sorted_points.cpu().share_memory_()
    bev_flat = bev_flat.cpu().share_memory_()

    assert cum_hist.is_shared() and active_cell_indices.is_shared() and cpu_sorted_points.is_shared() and bev_flat.is_shared()
    
    # Apply multi-proc CPU operations
    num_works = reusable_pool._processes * 4
    w = len(active_cell_indices)
    work_per = m.ceil(w / num_works)
    work = [(min(workid * work_per, w), min((workid+1) * work_per, w), active_cell_indices, cum_hist, cpu_sorted_points, bev_flat, cpu_params) for workid in range(num_works)]

    pool_async = None
    if cpu_bev_process is not None:
        if last_pool_async is not None:
            last_pool_async.wait()
        pool_async = reusable_pool.starmap_async(cpu_bev_process, work)

    return bev_flat, pool_async


import os


def points_within_angle(direction, angle, points):

    angles = torch.atan2(points[:, 1], points[:, 0])

    angles = (angles - direction) % (2 * torch.pi)
    angles = (angles + torch.pi) % (2 * torch.pi) - torch.pi

    within_angle = torch.abs(angles) <= angle / 2.0

    return within_angle


def filter_points(points, min_height, max_height, min_x, max_x, min_y, max_y, cam_forward=0.0, cam_degree=torch.pi/2):
    valid_h_points = torch.logical_and(points[:,2] >= min_height, points[:,2] <= max_height) # clip OOB points
    valid_xy_points = torch.logical_and(
        torch.logical_and(points[:,0] >= min_x, points[:,0] <= max_x),
        torch.logical_and(points[:,1] >= min_y, points[:,1] <= max_y))
    valid_a_points = points_within_angle(cam_forward, cam_degree, points[:,:2])
    valid = torch.logical_and(torch.logical_and(valid_xy_points, valid_h_points), valid_a_points)
    return points[valid,:]


def create_reusable_pool(num_workers=None):
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    start_method = 'spawn'
    if multiprocessing.get_start_method() != start_method:
        multiprocessing.set_start_method(start_method, force=True)

    ctx = multiprocessing.get_context(start_method)
    print(f"Making {num_workers} processes pool using {ctx.get_start_method()}")
    pool = ctx.Pool(num_workers)

    return pool

from PIL import Image

def save_as_image(t, dst_dir, im_id):
    numpy_array = t.cpu().numpy()
    numpy_array = np.round((numpy_array * 255)).astype(np.uint8)
    for i in range(0, numpy_array.shape[-1], 3):
        s = 0
        e = min(3, numpy_array.shape[-1] - i)
        sub_im = np.zeros((numpy_array.shape[0], numpy_array.shape[1], 3), dtype=np.uint8)
        sub_im[:,:,s:e] = numpy_array[:,:,i:i+(e-s)]
        image = Image.fromarray(sub_im, mode="RGB")
        image.save(f"{dst_dir}/{im_id:06d}_{(i//3)+1}.png")

def save_as_mask(t, dst_dir, i):
    if len(t.shape) != 3 or t.shape[2] != 1:
        raise ValueError("Input tensor shape should be (N x M x 1)")

    int_tensor = t.to(torch.uint8) * 255
    image = Image.fromarray(int_tensor.squeeze().cpu().numpy(), mode='L')
    image.save(f"{dst_dir}/{i:06d}.png")

time_tracker = dict()

def run_point_cloud_conversion(dataset, h_offset, h_min, h_max, map_size, grid_size, num_channels, 
    gpu_process, gpu_params, cpu_process, cpu_params, post_process, post_params, dst_dir, channel_type=torch.float):
    global time_tracker
    assert grid_size % 2 == 0

    os.makedirs(dst_dir, exist_ok=True)

    cell_size = map_size / grid_size

    pool_async = None
    bev_im, last_bev_im = None, None
    reusable_pool = create_reusable_pool(16)
    st_time = time.time()
    full_time = time.time()

    for i in range(0, len(dataset)):
        cloud = dataset[i].cuda()
        cloud[:,2] += h_offset
        cloud = filter_points(cloud, h_min, h_max, 0, map_size / 2, -map_size / 2, map_size / 2)

        last_bev_im = bev_im
        bev_im, pool_async = convert_points_to_bev(cloud, map_size, cell_size, num_channels, gpu_process, gpu_params, cpu_process, cpu_params, reusable_pool, pool_async, channel_type=channel_type)
        
        # Because convert_points_to_bev starts an async starmap, we need to wait for the next iteration...
        if last_bev_im is not None:
            last_bev_im = last_bev_im.cpu().reshape(grid_size//2, grid_size, -1)
            if post_process is not None:
                last_bev_im = post_process(last_bev_im, post_params)
            if last_bev_im.dtype != torch.bool:
                save_as_image(last_bev_im, dst_dir, i-1)
            else:
                save_as_mask(last_bev_im, dst_dir, i-1)

        if i % 100 == 0:
            print("That took", time.time() - st_time, "for 100 images;", i, "/", len(train_3d_ds))
            print("--- At this rate it will take", (time.time() - st_time) * ((len(train_3d_ds) - i) / 100) / (60*60), "hours")
            st_time = time.time()

    # wait for the last image to complete
    if pool_async is not None:
        pool_async.wait()
    
    last_bev_im = bev_im.cpu().reshape(grid_size//2, grid_size, -1)
    if post_process is not None:
        last_bev_im = post_process(last_bev_im, post_params)
    if last_bev_im.dtype != torch.bool:
        save_as_image(last_bev_im, dst_dir, i)
    else:
        save_as_mask(last_bev_im, dst_dir, i)

    time_tracker[dst_dir] = (full_time - time.time()) / len(dataset)


def _post_process(t, params):
    t[:,:,2] = torch.clamp((t[:,:,2] - params[0]) / (params[1] - params[0]), min=0.0, max=1.0)
    return t

def _post_process_stack(t, params):
    t[:,:,2] = torch.clamp((t[:,:,2] - params[0]) / (params[1] - params[0]), min=0.0, max=1.0)
    t[:,:,5] = torch.clamp((t[:,:,5] - params[1]) / (params[2] - params[1]), min=0.0, max=1.0)
    return t


def multiproc_process_active_cells_birdnet2(_start, _end, active_cell_indices, cum_hist, sorted_points, bev_flat, params):
    for i in range(_start,_end):
        cell_ind = active_cell_indices[i]
        start = cum_hist[i]
        end = cum_hist[i+1]
        bev_flat[cell_ind,0] = torch.mean(sorted_points[start:end, 3])
        bev_flat[cell_ind,2] = torch.max(sorted_points[start:end, 2])

def gpu_process_active_cells_birdnet2(bev_flat, histogram, cum_hist, sorted_points, params):
    max_map = params[0]
    density = (histogram / max_map.flatten())
    bev_flat[:,1] = density

def perform_birdnet2_bev_conversion(dataset, dst_dir, parameters_dict):
    map_size = parameters_dict["map_size"]
    cell_size = parameters_dict["cell_size"]
    min_height = parameters_dict["min_height"]
    max_height = parameters_dict["max_height"]
    h_slices = parameters_dict["h_slices"]
    num_planes = parameters_dict["num_planes"]
    velo_minangle = parameters_dict["velo_minangle"]
    velo_hres = parameters_dict["velo_hres"]
    velo_vres = parameters_dict["velo_vres"]
    velo_height = parameters_dict["velo_height"]
    
    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    

    gpu_params = [max_map]
    gpu_process = gpu_process_active_cells_birdnet2
    cpu_params = None
    cpu_process = multiproc_process_active_cells_birdnet2
    post_params = [min_height, max_height]
    run_point_cloud_conversion(dataset, velo_height, min_height, max_height, map_size, grid_size, 3, 
        gpu_process, gpu_params, cpu_process, cpu_params, _post_process, post_params, dst_dir)


def multiproc_process_active_cells_birdnet2_with_mean(_start, _end, active_cell_indices, cum_hist, sorted_points, bev_flat, params):
    x_len, cell_size = params
    y_len = x_len // 2

    for i in range(_start,_end):
        cell_ind = active_cell_indices[i]
        cell_x, cell_y = cell_ind % x_len, cell_ind // x_len
        y_bottom = ((y_len-1) - cell_y) * cell_size
        x_bottom = (cell_x  - (x_len//2))*cell_size

        start = cum_hist[i]
        end = cum_hist[i+1]
        bev_flat[cell_ind,0] = torch.mean(sorted_points[start:end, 3])
        bev_flat[cell_ind,2] = torch.max(sorted_points[start:end, 2])
        bev_flat[cell_ind,3:] = torch.mean(sorted_points[start:end, :3], dim=0)
        bev_flat[cell_ind,3] -= y_bottom
        bev_flat[cell_ind,4] -= x_bottom
        bev_flat[cell_ind,3:5] /= cell_size


def perform_birdnet2_bev_with_mean_conversion(dataset, dst_dir, parameters_dict):
    map_size = parameters_dict["map_size"]
    cell_size = parameters_dict["cell_size"]
    min_height = parameters_dict["min_height"]
    max_height = parameters_dict["max_height"]
    h_slices = parameters_dict["h_slices"]
    num_planes = parameters_dict["num_planes"]
    velo_minangle = parameters_dict["velo_minangle"]
    velo_hres = parameters_dict["velo_hres"]
    velo_vres = parameters_dict["velo_vres"]
    velo_height = parameters_dict["velo_height"]
    
    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    

    gpu_params = [max_map]
    gpu_process = gpu_process_active_cells_birdnet2
    cpu_params = [grid_size, cell_size]
    cpu_process = multiproc_process_active_cells_birdnet2_with_mean
    post_params = [min_height, max_height]
    run_point_cloud_conversion(dataset, velo_height, min_height, max_height, map_size, grid_size, 6, 
        gpu_process, gpu_params, cpu_process, cpu_params, _post_process, post_params, dst_dir)


def fit_plane_to_points(points):
    '''
    points: tensor of shape (N, 3) where N is the number of points
    '''
    A = torch.cat((points, torch.ones(points.shape[0], 1)), dim=1)
    U, S, V = torch.linalg.svd(A)
    plane_coefficients = V[-1, :]
    plane_coefficients /= torch.norm(plane_coefficients[:3])
    return plane_coefficients

def fit_plane_to_points_with_add(points):
    '''
    TODO TOOD TODO
    points: tensor of shape (N, 3) where N is the number of points
    '''
    A = torch.cat((points, torch.ones(points.shape[0], 1)), dim=1)
    U, S, V = torch.linalg.svd(A)
    plane_coefficients = V[-1, :]
    plane_coefficients /= torch.norm(plane_coefficients[:3])
    return plane_coefficients

def multiproc_process_active_cells_birdnet2_with_plane(_start, _end, active_cell_indices, cum_hist, sorted_points, bev_flat, params):
    x_len, cell_size = params
    y_len = x_len // 2

    for i in range(_start,_end):
        cell_ind = active_cell_indices[i]
        cell_x, cell_y = cell_ind % x_len, cell_ind // x_len
        y_bottom = ((y_len-1) - cell_y) * cell_size
        x_bottom = (cell_x  - (x_len//2))*cell_size

        start = cum_hist[i]
        end = cum_hist[i+1]
        bev_flat[cell_ind,0] = torch.mean(sorted_points[start:end, 3])
        bev_flat[cell_ind,2] = torch.max(sorted_points[start:end, 2])

        if end - start >= 3:
            norm_points = sorted_points[start:end, :3]
            norm_points[:, 0] -= y_bottom
            norm_points[:, 1] -= x_bottom
            norm_points[:,0:2] /= cell_size

            plane = fit_plane_to_points(norm_points)
            if plane[1] < 0.0: plane *= -1.0
            plane = torch.clamp((plane + 1.0) / 2.0, min=0.0, max=1.0)

            bev_flat[cell_ind,3] = plane[0]
            bev_flat[cell_ind,4] = plane[1]
            bev_flat[cell_ind,5] = plane[2]
            bev_flat[cell_ind,6] = plane[3]
        else:
            bev_flat[cell_ind,3] = 0.0
            bev_flat[cell_ind,4] = 1.0
            bev_flat[cell_ind,5] = 0.0
            bev_flat[cell_ind,6] = 0.0



def multiproc_process_active_cells_birdnet2_with_plane_with_floor_point(_start, _end, active_cell_indices, cum_hist, sorted_points, bev_flat, params):
    x_len, cell_size = params
    y_len = x_len // 2

    floor_point = torch.tensor([[0.5, 0.5, 0.0]])

    for i in range(_start,_end):
        cell_ind = active_cell_indices[i]
        cell_x, cell_y = cell_ind % x_len, cell_ind // x_len
        y_bottom = ((y_len-1) - cell_y) * cell_size
        x_bottom = (cell_x  - (x_len//2))*cell_size

        start = cum_hist[i]
        end = cum_hist[i+1]
        bev_flat[cell_ind,0] = torch.mean(sorted_points[start:end, 3])
        bev_flat[cell_ind,2] = torch.max(sorted_points[start:end, 2])

        if end - start >= 3:
            norm_points = sorted_points[start:end, :3]
            norm_points[:, 0] -= y_bottom
            norm_points[:, 1] -= x_bottom
            norm_points[:,0:2] /= cell_size
            norm_points = torch.cat([norm_points, floor_point], dim=0)

            plane = fit_plane_to_points(norm_points)
            if plane[1] < 0.0: plane *= -1.0
            plane = torch.clamp((plane + 1.0) / 2.0, min=0.0, max=1.0)

            bev_flat[cell_ind,3] = plane[0]
            bev_flat[cell_ind,4] = plane[1]
            bev_flat[cell_ind,5] = plane[2]
            bev_flat[cell_ind,6] = plane[3]
        else:
            bev_flat[cell_ind,3] = 0.0
            bev_flat[cell_ind,4] = 1.0
            bev_flat[cell_ind,5] = 0.0
            bev_flat[cell_ind,6] = 0.0



def perform_birdnet2_bev_with_plane_conversion(dataset, dst_dir, parameters_dict, cpu_process=multiproc_process_active_cells_birdnet2_with_plane):
    print("Making Planes!")
    map_size = parameters_dict["map_size"]
    cell_size = parameters_dict["cell_size"]
    min_height = parameters_dict["min_height"]
    max_height = parameters_dict["max_height"]
    h_slices = parameters_dict["h_slices"]
    num_planes = parameters_dict["num_planes"]
    velo_minangle = parameters_dict["velo_minangle"]
    velo_hres = parameters_dict["velo_hres"]
    velo_vres = parameters_dict["velo_vres"]
    velo_height = parameters_dict["velo_height"]

    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    

    gpu_params = [max_map]
    gpu_process = gpu_process_active_cells_birdnet2
    cpu_params = [grid_size, cell_size]
    post_params = [min_height, max_height]
    run_point_cloud_conversion(dataset, velo_height, min_height, max_height, map_size, grid_size, 7, 
        gpu_process, gpu_params, cpu_process, cpu_params, _post_process, post_params, dst_dir)



def multiproc_process_active_cells_birdnet2_stack(_start, _end, active_cell_indices, cum_hist, sorted_points, bev_flat, params):
    max_map, h_mid = params
    for i in range(_start,_end):
        cell_ind = active_cell_indices[i]
        start = cum_hist[i]
        end = cum_hist[i+1]
        points = sorted_points[start:end]
        above = points[:, 2] >= h_mid
        above_points = points[above]
        below_points = points[~above]
        below_num = len(below_points)
        above_num = len(above_points)

        # Below
        bev_flat[cell_ind,0] = torch.mean(below_points[:, 3]) if below_num > 0 else 0.0
        bev_flat[cell_ind,1] = below_num / max_map[cell_ind]
        bev_flat[cell_ind,2] = torch.max(below_points[:, 2]) if below_num > 0 else 0.0
        # Above
        bev_flat[cell_ind,3] = torch.mean(above_points[:, 3]) if above_num > 0 else 0.0
        bev_flat[cell_ind,4] = above_num / max_map[cell_ind]
        bev_flat[cell_ind,5] = torch.max(above_points[:, 2]) if above_num > 0 else h_mid

def perform_birdnet2_bev_with_stack_conversion(dataset, dst_dir, parameters_dict, h_mid):
    map_size = parameters_dict["map_size"]
    cell_size = parameters_dict["cell_size"]
    min_height = parameters_dict["min_height"]
    max_height = parameters_dict["max_height"]
    h_slices = parameters_dict["h_slices"]
    num_planes = parameters_dict["num_planes"]
    velo_minangle = parameters_dict["velo_minangle"]
    velo_hres = parameters_dict["velo_hres"]
    velo_vres = parameters_dict["velo_vres"]
    velo_height = parameters_dict["velo_height"]
    
    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    

    gpu_params = [max_map]
    gpu_process = None
    cpu_params = [max_map.flatten().cpu(), h_mid]
    cpu_process = multiproc_process_active_cells_birdnet2_stack
    post_params = [min_height, h_mid, max_height]
    run_point_cloud_conversion(dataset, velo_height, min_height, max_height, map_size, grid_size, 6, 
        gpu_process, gpu_params, cpu_process, cpu_params, _post_process_stack, post_params, dst_dir)

















''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def gpu_process_count_thresh_sparse(bev_flat, histogram, cum_hist, sorted_points, params):
    bev_flat[:,0] = histogram >= params[0]


def perform_birdnet2_bev_sparse_with_count_thresh(dataset, dst_dir, parameters_dict, count=3):
    map_size = parameters_dict["map_size"]
    cell_size = parameters_dict["cell_size"]
    min_height = parameters_dict["min_height"]
    max_height = parameters_dict["max_height"]
    h_slices = parameters_dict["h_slices"]
    num_planes = parameters_dict["num_planes"]
    velo_minangle = parameters_dict["velo_minangle"]
    velo_hres = parameters_dict["velo_hres"]
    velo_vres = parameters_dict["velo_vres"]
    velo_height = parameters_dict["velo_height"]

    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    

    gpu_params = [count]
    gpu_process = gpu_process_count_thresh_sparse
    cpu_params = None
    cpu_process = None
    post_params = [min_height, max_height]
    run_point_cloud_conversion(dataset, velo_height, min_height, max_height, map_size, grid_size, 1, 
        gpu_process, gpu_params, cpu_process, cpu_params, None, post_params, dst_dir, channel_type=torch.bool)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def gpu_process_density_thresh_sparse(bev_flat, histogram, cum_hist, sorted_points, params):
    max_map = params[0]
    density = (histogram / max_map.flatten())
    bev_flat[:,0] = density >= params[1]

def perform_birdnet2_bev_sparse_with_density_thresh(dataset, dst_dir, parameters_dict, density_thresh=0.02):
    map_size = parameters_dict["map_size"]
    cell_size = parameters_dict["cell_size"]
    min_height = parameters_dict["min_height"]
    max_height = parameters_dict["max_height"]
    h_slices = parameters_dict["h_slices"]
    num_planes = parameters_dict["num_planes"]
    velo_minangle = parameters_dict["velo_minangle"]
    velo_hres = parameters_dict["velo_hres"]
    velo_vres = parameters_dict["velo_vres"]
    velo_height = parameters_dict["velo_height"]

    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    

    gpu_params = [max_map, density_thresh]
    gpu_process = gpu_process_density_thresh_sparse
    cpu_params = None
    cpu_process = None
    post_params = [min_height, max_height]
    run_point_cloud_conversion(dataset, velo_height, min_height, max_height, map_size, grid_size, 1, 
        gpu_process, gpu_params, cpu_process, cpu_params, None, post_params, dst_dir, channel_type=torch.bool)


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def gpu_process_height_thresh_sparse(bev_flat, histogram, cum_hist, sorted_points, params):
    bev_flat[:, 0] = histogram > 0

def perform_birdnet2_bev_sparse_with_height_thresh(dataset, dst_dir, parameters_dict, min_height=0.5):
    map_size = parameters_dict["map_size"]
    cell_size = parameters_dict["cell_size"]
    # min_height = parameters_dict["min_height"]
    max_height = parameters_dict["max_height"]
    h_slices = parameters_dict["h_slices"]
    num_planes = parameters_dict["num_planes"]
    velo_minangle = parameters_dict["velo_minangle"]
    velo_hres = parameters_dict["velo_hres"]
    velo_vres = parameters_dict["velo_vres"]
    velo_height = parameters_dict["velo_height"]

    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    

    gpu_params = [max_map, 0.1] # max_map, density threshold
    gpu_process = gpu_process_height_thresh_sparse
    cpu_params = None
    cpu_process = None
    post_params = [min_height, max_height]
    run_point_cloud_conversion(dataset, velo_height, min_height, max_height, map_size, grid_size, 1, 
        gpu_process, gpu_params, cpu_process, cpu_params, None, post_params, dst_dir, channel_type=torch.bool)




cell_size=0.05
map_size=cell_size*1024.0
min_height=0.0
max_height=3
h_slices=[] # Number of slices in Z, for now this is 1 only
num_planes=64
velo_minangle=-24.9 # TODO : might need reversing in make_max_map?
velo_hres=0.2
velo_vres=0.4
velo_height=1.73
parameters_dict = {
    "map_size": map_size,
    "cell_size": cell_size,
    "min_height": min_height,
    "max_height": max_height,
    "h_slices": h_slices,
    "num_planes": num_planes,
    "velo_minangle": velo_minangle,
    "velo_hres": velo_hres,
    "velo_vres": velo_vres,
    "velo_height": velo_height
}

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    src_dir = "detectron2/datasets/bv_kitti/velodyne_3d_points"
    src_train_dir = f"{src_dir}/training/velodyne/"

    im_dst_dir = "detectron2/datasets/bv_kitti/image/"
    sp_dst_dir = "detectron2/datasets/bv_kitti/sparse/"

    train_3d_ds = point_cloud_dataset.PointCloudData(src_train_dir)

    grid_size = int(map_size / cell_size)
    plane_angles = velo_minangle + (torch.arange(0, num_planes,dtype=torch.float) * velo_vres)
    max_map = make_max_map(grid_size // 2, cell_size, max_height, min_height, velo_height, velo_hres, plane_angles, h_slices)    
    torch.save(max_map, "custom_max_map.pt")

    perform_birdnet2_bev_with_stack_conversion(train_3d_ds, f'{im_dst_dir}/stack_half', parameters_dict, h_mid=velo_height/2) # 72 for 100

    perform_birdnet2_bev_with_plane_conversion(train_3d_ds, f'{im_dst_dir}/plane_norm', parameters_dict, multiproc_process_active_cells_birdnet2_with_plane)
    perform_birdnet2_bev_with_plane_conversion(train_3d_ds, f'{im_dst_dir}/plane_floor', parameters_dict, multiproc_process_active_cells_birdnet2_with_plane_with_floor_point)
    
    perform_birdnet2_bev_conversion(train_3d_ds, f'{im_dst_dir}/birdnet', parameters_dict) # ???
    perform_birdnet2_bev_with_mean_conversion(train_3d_ds, f'{im_dst_dir}/mean', parameters_dict) # ???
    perform_birdnet2_bev_with_stack_conversion(train_3d_ds, f'{im_dst_dir}/stack_one', parameters_dict, h_mid=1.0) # 72 for 100

    perform_birdnet2_bev_sparse_with_count_thresh(train_3d_ds, f'{sp_dst_dir}/count_1', parameters_dict, count=1)
    perform_birdnet2_bev_sparse_with_count_thresh(train_3d_ds, f'{sp_dst_dir}/count_2', parameters_dict, count=2)
    perform_birdnet2_bev_sparse_with_count_thresh(train_3d_ds, f'{sp_dst_dir}/count_3', parameters_dict, count=3)
    perform_birdnet2_bev_sparse_with_count_thresh(train_3d_ds, f'{sp_dst_dir}/count_4', parameters_dict, count=4)

    perform_birdnet2_bev_sparse_with_density_thresh(train_3d_ds, f'{sp_dst_dir}/density_0.01', parameters_dict, 0.01)
    perform_birdnet2_bev_sparse_with_density_thresh(train_3d_ds, f'{sp_dst_dir}/density_0.02', parameters_dict, 0.02)
    perform_birdnet2_bev_sparse_with_density_thresh(train_3d_ds, f'{sp_dst_dir}/density_0.04', parameters_dict, 0.04)
    perform_birdnet2_bev_sparse_with_density_thresh(train_3d_ds, f'{sp_dst_dir}/density_0.08', parameters_dict, 0.08)

    perform_birdnet2_bev_sparse_with_height_thresh(train_3d_ds, f'{sp_dst_dir}/height_0.0', parameters_dict, 0.0)
    perform_birdnet2_bev_sparse_with_height_thresh(train_3d_ds, f'{sp_dst_dir}/height_0.2', parameters_dict, 0.2)
    perform_birdnet2_bev_sparse_with_height_thresh(train_3d_ds, f'{sp_dst_dir}/height_0.4', parameters_dict, 0.4)
    perform_birdnet2_bev_sparse_with_height_thresh(train_3d_ds, f'{sp_dst_dir}/height_0.6', parameters_dict, 0.6)

    print(time_tracker)
    print()
    print(time_tracker.keys())
    print()
    print(time_tracker.values())


{'detectron2/datasets/bv_kitti/image//plane_floor': -0.04657586968019993, 'detectron2/datasets/bv_kitti/image//birdnet': -0.07699675426781456, 'detectron2/datasets/bv_kitti/image//mean': -0.16768011420889503, 'detectron2/datasets/bv_kitti/image//stack_half': -0.04829737917335734, 'detectron2/datasets/bv_kitti/image//stack_one': -0.04830121911156864, 'detectron2/datasets/bv_kitti/image//plane_norm': -0.04928944104815084, 'detectron2/datasets/bv_kitti/sparse//count_1': -0.012126762429400412, 'detectron2/datasets/bv_kitti/sparse//count_2': -0.010238076735110131, 'detectron2/datasets/bv_kitti/sparse//count_3': -0.008746687445852818, 'detectron2/datasets/bv_kitti/sparse//count_4': -0.00786990061942117, 'detectron2/datasets/bv_kitti/sparse//density_0.01': -0.012073667462760314, 'detectron2/datasets/bv_kitti/sparse//density_0.02': -0.011519942527753659, 'detectron2/datasets/bv_kitti/sparse//density_0.04': -0.010041991078173185, 'detectron2/datasets/bv_kitti/sparse//density_0.08': -0.008821688716925652, 'detectron2/datasets/bv_kitti/sparse//height_0.0': -0.012226427960660332, 'detectron2/datasets/bv_kitti/sparse//height_0.2': -0.010253951823929705, 'detectron2/datasets/bv_kitti/sparse//height_0.4': -0.009597974128860822, 'detectron2/datasets/bv_kitti/sparse//height_0.6': -0.009200925161583254}

#dict_keys(['detectron2/datasets/bv_kitti/image//plane_floor', 'detectron2/datasets/bv_kitti/image//birdnet', 'detectron2/datasets/bv_kitti/image//mean', 'detectron2/datasets/bv_kitti/image//stack_half', 'detectron2/datasets/bv_kitti/image//stack_one', 'detectron2/datasets/bv_kitti/image//plane_norm', 'detectron2/datasets/bv_kitti/sparse//count_1', 'detectron2/datasets/bv_kitti/sparse//count_2', 'detectron2/datasets/bv_kitti/sparse//count_3', 'detectron2/datasets/bv_kitti/sparse//count_4', 'detectron2/datasets/bv_kitti/sparse//density_0.01', 'detectron2/datasets/bv_kitti/sparse//density_0.02', 'detectron2/datasets/bv_kitti/sparse//density_0.04', 'detectron2/datasets/bv_kitti/sparse//density_0.08', 'detectron2/datasets/bv_kitti/sparse//height_0.0', 'detectron2/datasets/bv_kitti/sparse//height_0.2', 'detectron2/datasets/bv_kitti/sparse//height_0.4', 'detectron2/datasets/bv_kitti/sparse//height_0.6'])

#dict_values([-0.04657586968019993, -0.07699675426781456, -0.16768011420889503, -0.04829737917335734, -0.04830121911156864, -0.04928944104815084, -0.012126762429400412, -0.010238076735110131, -0.008746687445852818, -0.00786990061942117, -0.012073667462760314, -0.011519942527753659, -0.010041991078173185, -0.008821688716925652, -0.012226427960660332, -0.010253951823929705, -0.009597974128860822, -0.009200925161583254])