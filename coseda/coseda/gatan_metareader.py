import hyperspy.api as hs
import re
import json
import os
import glob
import os
import shutil
import platform
import ctypes
import h5py
from coseda.io import log_start, log_result, handle_input, parse_config, shoutout, get_free_space_unix, get_free_space_windows, get_dir_size

def parse_dm4_metadata(dm4file):
    # Initialize variables
    tree = {}
    path = []  # To keep track of the current position in the hierarchy

    file = hs.load(dm4file, lazy=True)
    metadata_str = str(file.original_metadata)
    
    # Process each line
    lines = metadata_str.splitlines()
    
    for line in lines:
        # Count leading special characters to determine depth
        depth = line.count('\u2502')
        
        # Adjust current path according to the depth
        path = path[:depth]
        
        # Use regex to remove leading special characters, including specific prefixes, and split key and value
        cleaned_line = re.sub(r'^[\u2502\u251c\u2500\s]*', '', line)  # Remove leading tree characters
        cleaned_line = re.sub(r'^[└──╠══╚══\s]*', '', cleaned_line)  # Remove specific leading characters for keys and list indices
        
        parts = cleaned_line.split(' = ')
        key = parts[0].strip()

        # Handle case with no value
        value = parts[1].strip() if len(parts) > 1 else None
        
        # Navigate to the correct place in the tree
        current = tree
        for step in path:
            current = current.setdefault(step, {})
        
        # Update the tree and path
        if value:
            current[key] = value
        else:
            path.append(key)
    
    return tree

def get_timestamp_from_path(path):
    match = re.search(r'Hour_(\d+)/Minute_(\d+)/Second_(\d+)/\d+_Hour_(\d+)_Minute_(\d+)_Second_(\d+)_Frame_(\d+).dm4', path)
    if match:
        # Generate a sortable timestamp string
        return tuple(map(int, match.groups()))
    return (0, 0, 0, 0, 0, 0, 0)  # Fallback for non-matching paths

def find_all_frames_ordered(parent_folder):
    all_frames = []
    
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".dm4"):
                all_frames.append(os.path.join(root, file))
                
    # Sort frames based on the extracted timestamp
    all_frames.sort(key=get_timestamp_from_path)
    
    return all_frames

def extract_info_from_gatan_metadata(input_path):
    all_files = find_all_frames_ordered(input_path)
    first_file = all_files[0]
    last_file = all_files[-1]

    metadata = parse_dm4_metadata(first_file)

    exposure_time = metadata['ImageList']['ImageTags']['Acquisition']['Detector']['exposure (s)']
    resolution = metadata['ImageList']['ImageTags']['Acquisition']['Device']['Active Size (pixels)']
    resolution_values = resolution.strip("()").split(", ")
    resolution_width = int(resolution_values[0])
    resolution_height = int(resolution_values[1])
    binning_info = metadata['ImageList']['ImageTags']['Acquisition']['Frame']['Area']['Binning']
    binning_values = binning_info.strip("()").split(", ")
    binning_width = int(float(binning_values[0]))
    binning_height = int(float(binning_values[1]))
    acceleration_voltage = int(float(metadata['ImageList']['ImageTags']['Microscope Info']['Voltage']))
    pixel_size = metadata['ImageList']['ImageTags']['Acquisition']['Device']['CCD']['Pixel Size (um)']
    pixel_size_values = pixel_size.strip("()").split(", ")
    pixel_width = float(pixel_size_values[0])
    pixel_height = float(pixel_size_values[1])
    pixel_unit = 'um'
    camera_length = metadata['ImageList']['ImageTags']['Microscope Info']['STEM Camera Length']
    detector_used = metadata['ImageList']['ImageTags']['Acquisition']['Device']['Source']
    instrument_id = metadata['ImageList']['ImageTags']['Acquisition']['Device']['Source ID']
    instrument_model = metadata['ImageList']['ImageTags']['Microscope']
    instrument_manufacturer = metadata['ImageList']['ImageTags']['Microscope Info']['Name']
    acquisition_start = metadata['ImageList']['ImageTags']['DataBar']['Acquisition Time (OS)']

    metadata_last = parse_dm4_metadata(last_file)

    acquisition_end = metadata['ImageList']['ImageTags']['DataBar']['Acquisition Time (OS)']


    return exposure_time, resolution_height, resolution_width, binning_height, binning_width, acceleration_voltage, pixel_height, pixel_width, pixel_unit, camera_length, detector_used, instrument_id, instrument_manufacturer, instrument_model, acquisition_start, acquisition_end

def extract_frame_timestamp_from_gatan_metadata(framefile_path):
    metadata = parse_dm4_metadata(framefile_path)
    timestamp = metadata['ImageList']['ImageTags']['DataBar']['Acquisition Time (OS)']
    return timestamp

def dm4_folder_conversion(dm4parentfolder, outputfolder, logfile_path, extract_timestamp=True, chunk_size=(1000, 1024, 1024)):
    try:
        h5file = outputfolder  + ".h5"
        parentfolderpath = os.path.dirname(logfile_path)
        h5file_path = os.path.join(parentfolderpath,h5file)

        all_files = find_all_frames_ordered(dm4parentfolder)

        new_framepath = 'entry/data/images'
        new_indexpath = 'entry/data/index'
        timestamppath = 'entry/data/timestamp_image'

        # Check if we have enough free disk space
        
        # Get the size of the file
        file_size = get_dir_size(dm4parentfolder)

        # Get free disk space on the drive where the file is located
        if platform.system() == 'Windows':
            free_space = get_free_space_windows(os.path.dirname(dm4parentfolder))
        else:
            free_space = get_free_space_unix(os.path.dirname(dm4parentfolder))

        # Check if the free space is greater than or equal to the file size
        if free_space >= 1.1 * file_size:
            print("sufficient disk space for conversion available")
        else:
            errormessage = f"insufficient disk space for file conversion, please ensure at least {1.01 * file_size} of free disk space is available"
            return errormessage, None

        with h5py.File(h5file_path, 'w') as new_file:
            log_start(logfile_path, "hdf5 file created")

            initial_shape = (0, chunk_size[1], chunk_size[2])  # No images at start, height, width
            maxshape = (None, chunk_size[1], chunk_size[2])
            new_dataset = new_file.create_dataset(new_framepath, shape=initial_shape, maxshape=maxshape, chunks=chunk_size, dtype='int16')

            index_dataset = new_file.create_dataset(new_indexpath, shape=(len(all_files),), dtype='i4')  # create dataset for indices

            if extract_timestamp is True:
                timestamp_dataset = new_file.create_dataset(timestamppath, shape=(len(all_files),), dtype='f8')  # create dataset for timestamps if needed

            for i, file in enumerate(all_files):
                if extract_timestamp is True:
                    timestamp = extract_frame_timestamp_from_gatan_metadata(file)
                    timestamp_dataset[i] = float(timestamp)

                #frame = hs.load(file, lazy=True)
                frame = hs.load(file).data  # Load the frame and get the data as a NumPy array

                # Resize the dataset to accommodate the new frame
                new_dataset.resize(new_dataset.shape[0] + 1, axis=0)
                new_dataset[-1] = frame  # Append the new frame to the dataset

                index_dataset[i] = i

            log_start(logfile_path, "chunked dataset created and data copied")
        return None, h5file
    except Exception as e:
        errormessage = f"an error occured during file conversion: {str(e)}"
        return errormessage, None
    

