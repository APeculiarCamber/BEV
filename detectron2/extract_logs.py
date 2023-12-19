import os
import shutil

def extract_out_logs(dir_list, output_dir):
    for directory in dir_list:
        for root, dirs, files in os.walk(directory):
            if 'out.log' in files:
                log_file_path = os.path.join(root, 'out.log')
                # Get the old parent directory name
                parent_dir_name = os.path.basename(os.path.dirname(log_file_path))
                # Create a new name for the file by concatenating old parent directory and file name
                new_file_name = f"{parent_dir_name}_{os.path.basename(log_file_path)}"
                # Define the destination path in the output directory
                destination_path = os.path.join(output_dir, new_file_name)
                # Copy the file to the output directory with the new name
                shutil.copy(log_file_path, destination_path)

directories = ["outputbevdataset_BASE_BIRDNET_bevlr_0.01",
"outputbevdataset_BASE_BIRDNET_bevlr_0.02",
"outputbevdataset_BASE_BIRDNET_bevlr_0.02",
"outputbevdataset_MEAN_BIRDNET_bevlr_0.01",
"outputbevdataset_MEAN_BIRDNET_bevlr_0.02",
"outputbevdataset_PLANE_BIRDNET_bevlr_0.01",
"outputbevdataset_PLANE_BIRDNET_bevlr_0.02",
"outputbevdataset_STACK_BIRDNET_bevlr_0.01",
"outputbevdataset_STACK_BIRDNET_bevlr_0.02",]

directories += ["outputbevdataset_BASE_BIRDNET_bevlr_0.02sparse_type_HEIGHT_THRESH_4_sparse_block_size_32",
"outputbevdataset_BASE_BIRDNET_bevlr_0.02sparse_type_HEIGHT_THRESH_4_sparse_block_size_64",
"outputbevdataset_BASE_BIRDNET_bevlr_0.02sparse_type_HEIGHT_THRESH_4_sparse_block_size_128",
"outputbevdataset_BASE_BIRDNET_bevlr_0.02sparse_type_DENSITY_THRESH_8_sparse_block_size_32",
"outputbevdataset_BASE_BIRDNET_bevlr_0.02sparse_type_DENSITY_THRESH_8_sparse_block_size_64",
"outputbevdataset_BASE_BIRDNET_bevlr_0.02sparse_type_DENSITY_THRESH_8_sparse_block_size_128",]
output_directory = 'log_out'  # Replace with your desired output directory
os.makedirs(output_directory, exist_ok=True)
extract_out_logs(directories, output_directory)
