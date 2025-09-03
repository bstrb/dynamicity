import h5py
import json
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import shutil
import platform
from PIL import Image
from coseda.io import log_start, log_result, handle_input, parse_config, shoutout, get_free_space_windows, get_free_space_unix, config_to_paths, read_config


def velox_batch(input_folder):
    input_paths = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".emd"):  # Check for HDF5 files
            input_paths.append(os.path.join(input_folder, filename))
    
    for input_path in input_paths:
        velox_process(input_path)

    return True

def velox_process(input_path):
    try:
        framepath, framelookuptablepath, datapath = find_dataset_paths(input_path)
        #print(framepath, datapath)

        print('Extracting stage positions from metadata')
        #extract_pos_data(input_path, datapath)

        print('Refining stage positions')
        #refine_stagepos_x(input_path)

        print('Remapping frame dataset')
        #remap_framestack(input_path, framepath)

        calculate_mean_intensities(input_path)
    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")
        return False

    return True

def find_dataset_paths(h5file_path):
    framepath, framelookuptablepath, metadatapath = None, None, None
    with h5py.File(h5file_path, 'r') as workingfile:
        result = find_path_in_h5(workingfile)
        if result:
            framepath, framelookuptablepath, metadatapath = result
            log_result(None, f'Frame path identified as {str(framepath)}', None)
            log_result(None, f'Framelookuptable path identified as {str(framelookuptablepath)}', None)
            log_result(None, f'Metadata path identified as {str(metadatapath)}', None)
    return framepath, framelookuptablepath, metadatapath

def find_path_in_h5(h5file):
    base_path = '/Data/Image/'
    if base_path in h5file:
        image_group = h5file[base_path]
        for subfolder in image_group:
            data_path = f'{base_path}{subfolder}/Data'
            framlookuptable_path = f'{base_path}{subfolder}/FrameLookupTable'
            metadata_path = f'{base_path}{subfolder}/Metadata'
            if data_path in h5file and metadata_path in h5file and framlookuptable_path in h5file:
                return data_path, framlookuptable_path, metadata_path
    return None

def decode_column(metadata, column_index, limit=None):
    byte_str = metadata[:limit, column_index].tobytes() if limit else metadata[:, column_index].tobytes()
    return byte_str.decode('ascii', errors='ignore').strip()

def extract_instrument_data(input_type, originalfile_path):
    if input_type == 'emd' or input_type == '.emd':
        _, _, metadatapath = find_dataset_paths(originalfile_path)

        if not metadatapath:
            print('Metadata path not found.')
            return None

        print(f'trying to locate metadata')

        with h5py.File(originalfile_path, 'r+') as workingfile:
            print(f'extracting metadata from {metadatapath}')
            metadata = workingfile[metadatapath]

            # Decode the metadata column
            decoded_metadata = decode_column(metadata, 0)
            
            try:
                data = json.loads(decoded_metadata)
                keys = data.keys()
                return data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")

                print(f"attempting to fix this by trimming the JSON string")

                # Find the position of the error
                error_position = e.pos

                # Cut the string at the position of the error
                cleaned_data = decoded_metadata[:error_position]
    
            try:
                    # Directly accessing dictionary keys
                    data = json.loads(cleaned_data)

                    extracted_data = {
                    'InstrumentManufacturer': data.get('Instrument', {}).get('Manufacturer', None),
                    'InstrumentModel': data.get('Instrument', {}).get('InstrumentModel', None),
                    'InstrumentId': data.get('Instrument', {}).get('InstrumentId', None),
                    'DetectorUsed': next(iter(data.get('Detectors', {}).values()), {}).get('DetectorName', None),
                    'AccelerationVoltage': data.get('Optics', {}).get('AccelerationVoltage', None),
                    'PixelWidth': data.get('BinaryResult', {}).get('PixelSize', {}).get('width', None),
                    'PixelHeight': data.get('BinaryResult', {}).get('PixelSize', {}).get('height', None),
                    'BinningWidth': next(iter(data.get('Detectors', {}).values()), {}).get('Binning', {}).get('width', None),
                    'BinningHeight': next(iter(data.get('Detectors', {}).values()), {}).get('Binning', {}).get('height', None),
                    'Resolution': (data.get('BinaryResult', {}).get('ImageSize', {}).get('width', None),
                                   data.get('BinaryResult', {}).get('ImageSize', {}).get('height', None)),
                    'ExposureTime': next(iter(data.get('Detectors', {}).values()), {}).get('ExposureTime', None),
                    'PixelUnit': data.get('BinaryResult', {}).get('PixelUnitX', None),
                    'PixelOffsetX': data.get('BinaryResult', {}).get('Offset', {}).get('x', None),
                    'PixelOffsetY': data.get('BinaryResult', {}).get('Offset', {}).get('y', None),
                    'AcquisitionStart': data.get('Acquisition', {}).get('AcquisitionStartDatetime', {}).get('DateTime', None),
                    'AcquisitionEnd': data.get('Acquisition', {}).get('AcquisitionDatetime', {}).get('DateTime', None),
                    'CameraLength': data.get('Optics', {}).get('CameraLength', None)
                }
                    pixels_per_meter = None
                    if extracted_data['DetectorUsed'] == 'BM-Ceta' and extracted_data['BinningWidth'] is not None:
                        if extracted_data['BinningWidth'] is not None:
                            pixels_per_meter = 1/(14e-6*float(extracted_data['BinningWidth']))
                        else:
                            pixels_per_meter = None

                    extracted_data.update({
                        'pixels_per_meter': pixels_per_meter
                    })
                        
                    print("fixed JSON parsing error")
                    return extracted_data
            
            except Exception as e:
                    print(f"JSON parsing error: {e}")
                    return None
    else:
        return None

def extract_stage_pos(decoded_metadata):
    try:
        metadata_json = json.loads(decoded_metadata)
        stage_pos_x = float(metadata_json["Stage"]["Position"]["x"])
        stage_pos_y = float(metadata_json["Stage"]["Position"]["y"])
        stage_pos_z = float(metadata_json["Stage"]["Position"]["z"])
        alphatilt = float(metadata_json["Stage"]["AlphaTilt"])
        betatilt = float(metadata_json["Stage"]["BetaTilt"])
        return stage_pos_x, stage_pos_y, stage_pos_z, alphatilt, betatilt, None
    except json.JSONDecodeError as e:
        return None, None, None, None, None, e.pos

def write_stage_pos_to_hdf5(stage_positions, workingfile):
    if 'entry' in workingfile:
        entry_grp = workingfile['entry']
    else:
        entry_grp = workingfile.create_group('entry')

    if 'data' in entry_grp:
        data_grp = entry_grp['data']
    else:
        data_grp = entry_grp.create_group('data')


    # Function to create or overwrite dataset
    def create_or_overwrite_dataset(group, name, data):
        if name in group:
            del group[name]
        group.create_dataset(name, data=data)

    # Create or overwrite datasets
    create_or_overwrite_dataset(data_grp, 'stagepos_x', [pos[0] for pos in stage_positions])
    create_or_overwrite_dataset(data_grp, 'stagepos_y', [pos[1] for pos in stage_positions])
    create_or_overwrite_dataset(data_grp, 'stagepos_z', [pos[2] for pos in stage_positions])
    create_or_overwrite_dataset(data_grp, 'alphatilt', [pos[3] for pos in stage_positions])
    create_or_overwrite_dataset(data_grp, 'betatilt', [pos[4] for pos in stage_positions])

def extract_pos_data(h5file_path, datapath):
    try:
        with h5py.File(h5file_path, 'r+') as workingfile:
            metadata = workingfile[datapath]
            num_frames = metadata.shape[1]
            
            stage_positions = []

            for i in range(num_frames):
                decoded_metadata = decode_column(metadata, i)
                stage_pos_x, stage_pos_y, stage_pos_z, alphatilt, betatilt, error_pos = extract_stage_pos(decoded_metadata)

                if error_pos is not None:
                    decoded_metadata = decode_column(metadata, i, limit=error_pos - 1)
                    stage_pos_x, stage_pos_y, stage_pos_z, alphatilt, betatilt, _ = extract_stage_pos(decoded_metadata)

                if None not in [stage_pos_x, stage_pos_y, stage_pos_z, alphatilt, betatilt]:
                    stage_positions.append((stage_pos_x, stage_pos_y, stage_pos_z, alphatilt, betatilt))
                else:
                    print(f"skipping column {i} due to decoding error.")

            #print(stage_positions)

            write_stage_pos_to_hdf5(stage_positions, workingfile)
        return None
    except Exception as e:
        return str(e)

def refine_stagepos_x(h5file_path):
    try:
        with h5py.File(h5file_path, 'r+') as file:
            stage_pos_x = np.array(file['entry/data/stagepos_x'])
            original_stage_pos_x = stage_pos_x.copy()
            prev_value = None

            # Replace all values that are similar to previous value by NaN
            for index, value in enumerate(stage_pos_x):
                if value == prev_value:
                    stage_pos_x[index] = np.nan
                prev_value = value
            
            # Identify indices where NaN values are present
            nan_indices = np.where(np.isnan(stage_pos_x))[0]

            # Identify indices where NaN values are NOT present 
            non_nan_indices = np.where(~np.isnan(stage_pos_x))[0]

            # Perform linear interpolation
            stage_pos_x[nan_indices] = np.interp(nan_indices, non_nan_indices, stage_pos_x[non_nan_indices])


            # Plot for debugging
            # plt.figure(figsize=(10, 6))
            # plt.plot(original_stage_pos_x, label='Original Data')
            # plt.plot(stage_pos_x, label='Interpolated Data', linestyle='--')
            # plt.legend()
            # plt.title('Comparison of Original and Interpolated Data')
            # plt.xlabel('Index')
            # plt.ylabel('Stage Position X')
            # plt.show()
            
            # Write or overwrite the refined data
            refined_dataset_path = 'entry/data/stagepos_x_refined'
            if refined_dataset_path in file:
                # Overwrite existing dataset
                file[refined_dataset_path][...] = stage_pos_x
            else:
                # Create a new dataset
                file.create_dataset(refined_dataset_path, data=stage_pos_x)
        return None
    except Exception as e:
        return str(e)

def remap_framestack(h5file_path, framepath, chunked=True, chunk_size=(1000, 1024, 1024)):
    try:
        with h5py.File(h5file_path, 'a') as workingfile:
            dataset = workingfile[framepath]
            x_dim, y_dim, z_dim = dataset.shape

            if chunked:
                # Create a new chunked dataset with the correct shape
                new_dataset_name = 'entry/data/images'
                chunked_dataset = workingfile.create_dataset(new_dataset_name, 
                                                             shape=(z_dim, y_dim, x_dim), 
                                                             dtype=dataset.dtype, 
                                                             chunks=chunk_size)

                # Reshape and copy data from the original dataset to the new chunked dataset
                for z in range(z_dim):
                    chunked_dataset[z, :, :] = dataset[:, :, z].transpose()

                # Optionally, replace the original dataset with the new one
                # This step is destructive. Ensure you have a backup of your data
                del workingfile[framepath]

                workingfile[framepath] = chunked_dataset

            else:
                # Original virtual dataset remapping
                vlayout = h5py.VirtualLayout(shape=(z_dim, y_dim, x_dim), dtype='int16')
                vsource = h5py.VirtualSource(h5file_path, framepath, shape=(x_dim, y_dim, z_dim))
                for z in range(z_dim):
                    vlayout[z, :, :] = vsource[:, :, z]
                workingfile.create_virtual_dataset('entry/data/images', vlayout, fillvalue=0)

        return None
    except Exception as e:
        return str(e)

def velox_true_conversion(h5file_path, chunk_size=(1000, 1024, 1024)):
    backup_file_path = h5file_path + ".backup"
    new_file_path = h5file_path
    framepath, framelookuptablepath, metadata = find_dataset_paths(h5file_path)
    print(f'original datset located in: {framepath}')
    new_framepath = 'entry/data/images'

    # Check if we have enough free disk space
    
    # Get the size of the file
    file_size = os.path.getsize(h5file_path)

    # Get free disk space on the drive where the file is located
    if platform.system() == 'Windows':
        free_space = get_free_space_windows(os.path.dirname(h5file_path))
    else:
        free_space = get_free_space_unix(os.path.dirname(h5file_path))

    # Check if the free space is greater than or equal to the file size
    if free_space >= 1.1 * file_size:
        print("sufficient disk space for conversion available")
    else:
        print(f"insufficient disk space, please ensure at least {1.1 * file_size} of free disk space is available")
        return 'Insufficient disk space for conversion'

    try:
        # Step 1: Rename the original file
        shutil.move(h5file_path, backup_file_path)
        print("original file renamed to backup")

        # Step 2: Create a new file with the original name
        with h5py.File(new_file_path, 'w') as new_file, h5py.File(backup_file_path, 'r') as backup_file:
            print("new file created with the original name")

            # Step 3: Copy data, specifically handle 'Data' group
            for group_name in backup_file:
                if group_name != 'Data':
                    backup_file.copy(group_name, new_file)
                    print(f"copied {group_name} from backup to new file")

            if metadata in backup_file:
                # Split the metadata path to get the parent group and dataset name
                path_parts = metadata.split('/')
                dataset_name = path_parts[-1]
                parent_group_path = '/'.join(path_parts[:-1])

                parent_group = backup_file[parent_group_path]
                parent_group.copy(dataset_name, new_file[parent_group_path] if parent_group_path in new_file else new_file)
                print(f"copied {metadata} from backup to new file")

            if framelookuptablepath in backup_file:
                # Split the metadata path to get the parent group and dataset name
                path_parts = framelookuptablepath.split('/')
                dataset_name = path_parts[-1]
                parent_group_path = '/'.join(path_parts[:-1])

                parent_group = backup_file[parent_group_path]
                parent_group.copy(dataset_name, new_file[parent_group_path] if parent_group_path in new_file else new_file)
                print(f"copied {framelookuptablepath} from backup to new file")
                

            # Delete the framepath in the new dataset, we don't need that but I'm too lazy to find a way excluding it from copying
            if new_framepath in new_file:
                del new_file[new_framepath]
                print(f"deleted {new_framepath} from new file")

            # Step 4 & 5: Chunk and write the frames
            print(f"started copying image data")    
            dataset = backup_file[framepath]
            x_dim, y_dim, z_dim = dataset.shape
            new_dataset_name = new_framepath

            chunked_dataset = new_file.create_dataset(new_dataset_name, shape=(z_dim, y_dim, x_dim),dtype=dataset.dtype, chunks=chunk_size)
            for z in range(z_dim):
                chunked_dataset[z, :, :] = dataset[:, :, z]
            
            print("chunked dataset created and data copied")

        # Step 6: Delete the original file
        os.remove(backup_file_path)
        print("backup file deleted")

        return None
    except Exception as e:
        # In case of an exception, revert the renaming
        if os.path.exists(backup_file_path):
            shutil.move(backup_file_path, h5file_path)
        return f"{str(e)}"



def calculate_mean_intensities(h5file_path):
    try:
        with h5py.File(h5file_path, 'r+') as file:
            intensities_dataset = 'entry/data/mean_intensities'
            images = file['entry/data/images'][:]
            intensities = images.mean(axis=(1, 2))
                  
            if intensities_dataset in file:
                # Overwrite existing dataset
                file[intensities_dataset][...] = intensities
            else:
                # Create a new dataset
                file.create_dataset(intensities_dataset, data=intensities)
        return None
    except Exception as e:
        return str(e)


def calculate_mean_intensities_chunked(h5file_path, batch_size=1000):
    try:
        with h5py.File(h5file_path, 'r+') as file:
            intensities_dataset = 'entry/data/mean_intensities'
            images_dataset = file['entry/data/images']

            num_images = images_dataset.shape[0]
            mean_intensities = np.zeros(num_images)

            # Process images in batches
            for i in range(0, num_images, batch_size):
                batch = images_dataset[i:i+batch_size]
                mean_intensities[i:i+batch_size] = batch.mean(axis=(1, 2))

            # Save the results
            if intensities_dataset in file:
                file[intensities_dataset][...] = mean_intensities
            else:
                file.create_dataset(intensities_dataset, data=mean_intensities)
        return None
    except Exception as e:
        return str(e)

def calculate_total_intensities(h5file_path):
    try:
        with h5py.File(h5file_path, 'r+') as file:
            intensities_dataset = 'entry/data/total_intensities'
            images = file['entry/data/images'][:]
            intensities = images.mean(axis=(1, 2))
                  
            if intensities_dataset in file:
                # Overwrite existing dataset
                file[intensities_dataset][...] = intensities
            else:
                # Create a new dataset
                file.create_dataset(intensities_dataset, data=intensities)
        return None
    except Exception as e:
        return str(e)
    

def calculate_total_intensities_chunked(h5file_path, batch_size=1000):
    try:
        with h5py.File(h5file_path, 'r+') as file:
            intensities_dataset = 'entry/data/total_intensities'
            images_dataset = file['entry/data/images']

            num_images = images_dataset.shape[0]
            total_intensities = np.zeros(num_images)

            # Process images in batches
            for i in range(0, num_images, batch_size):
                batch = images_dataset[i:i+batch_size]
                total_intensities[i:i+batch_size] = batch.sum(axis=(1, 2))

            # Save the results
            if intensities_dataset in file:
                file[intensities_dataset][...] = total_intensities
            else:
                file.create_dataset(intensities_dataset, data=total_intensities)
        return None
    except Exception as e:
        return str(e)

def plot_4Dstem(input_path):
    with h5py.File(input_path, 'r') as file:
        stagepos_x = file['entry/data/stagepos_x_refined'][:]
        stagepos_y = file['entry/data/stagepos_y'][:]
        total_intensities = file['entry/data/mean_intensities'][:]

    # Determine the scaling and translation factors for the coordinates
    x_min, x_max = stagepos_x.min(), stagepos_x.max()
    y_min, y_max = stagepos_y.min(), stagepos_y.max()

    print('x range: ' + str(x_max - x_min))
    print(x_min, x_max)
    print('y range: ' + str(y_max - y_min))
    print(y_min, y_max)

    # Normalize and visualize
    plt.figure()
    plt.scatter(stagepos_x, stagepos_y, c=total_intensities, cmap='inferno_r', marker='.', s=200)
    plt.colorbar(label='Normalized Total Intensity')
    plt.title('Intensity Mapping')
    plt.axis('equal')  # Ensure equal scaling of the x and y axes
    plt.savefig('intensity_map.png')
    plt.show()

def plot_crystamorphus(input_path):
    with h5py.File(input_path, 'r') as file:
        stagepos_x = file['entry/data/stagepos_x_refined'][:]
        stagepos_y = file['entry/data/stagepos_y'][:]
        total_intensities = file['entry/data/mean_intensities'][:]
        npeaks = file['entry/data/nPeaks'][:]

    # Determine the scaling and translation factors for the coordinates
    x_min, x_max = stagepos_x.min(), stagepos_x.max()
    y_min, y_max = stagepos_y.min(), stagepos_y.max()

    print('x range: ' + str(x_max - x_min))
    print(x_min, x_max)
    print('y range: ' + str(y_max - y_min))
    print(y_min, y_max)

    # Normalize and visualize
    plt.figure()
    plt.scatter(stagepos_x, stagepos_y, c=total_intensities, cmap='inferno_r', marker='.', s=200)
    plt.colorbar(label='Normalized Total Intensity')
    plt.title('Intensity Mapping')
    plt.axis('equal')  # Ensure equal scaling of the x and y axes
    plt.savefig('intensity_map.png')
    plt.show()

def add_intesities_batch(input_path):
    configfiles, _ = handle_input(input_path)

    for configfile in configfiles:
        outputfolder, outputfolder_path, logfile, logfile_path, h5file, h5file_path = config_to_paths(configfile)
        
        print(f"Working with {os.path.basename(configfile)}")

        calculate_mean_intensities_chunked(h5file_path)

def save_4Dstem(input_path):
    from coseda.initialize import handle_input, parse_config

    configfiles, input_path = handle_input(input_path)
    
    for configfile in configfiles:
        config, outputfolder, originalfile, logfile, path, outputfolder_path, originalfile_path, logfile_path, framepath, h5file, h5file_path = parse_config(configfile)

        with h5py.File(h5file_path, 'r') as file:
            stagepos_x = file['entry/data/stagepos_x_refined'][:]
            stagepos_y = file['entry/data/stagepos_y'][:]
            total_intensities = file['entry/data/mean_intensities'][:]
            alphatilt = file['entry/data/alphatilt'][:]

        # Determine the scaling and translation factors for the coordinates
        x_min, x_max = stagepos_x.min(), stagepos_x.max()
        y_min, y_max = stagepos_y.min(), stagepos_y.max()

        print('x range: ' + str(x_max - x_min))
        print(x_min, x_max)
        print('y range: ' + str(y_max - y_min))
        print(y_min, y_max)

        # Convert alpha tilt from radians to degrees
        alphatilt_degrees = np.degrees(alphatilt)

        # Check if alpha tilt is similar for all frames
        if np.allclose(alphatilt_degrees, alphatilt_degrees[0], atol=1e-6):
            # If alpha tilt is similar, add it as text to the title
            title = f'Intensity Mapping (Alpha Tilt: {alphatilt_degrees[0]:.2f} degrees)'
        else:
            title = 'Intensity Mapping'

        # Normalize and visualize
        plt.figure()
        plt.scatter(stagepos_x, stagepos_y, c=total_intensities, cmap='inferno_r', marker='.', s=200)
        plt.colorbar(label='Normalized Total Intensity')
        plt.title(title)
        plt.axis('equal')  # Ensure equal scaling of the x and y axes

        filename = outputfolder + '.png'
        plt.savefig(os.path.join(outputfolder_path, filename))

def get_image_mode(dtype):
    if dtype == np.uint8:
        return 'L'
    elif dtype == np.uint16:
        return 'I;16'
    elif dtype == np.int16:
        return 'I;16'  # Treat as 16-bit unsigned for Pillow
    elif dtype == np.float32 or dtype == np.float64:
        return 'F'
    else:
        raise ValueError(f"Unsupported image data type: {dtype}")

def export_hdf5_images_to_tiff(hdf5_path, output_dir):
    framepath, framelookuptablepath, metadatapath = find_dataset_paths(hdf5_path)
    
    if not framepath:
        raise ValueError("Frame path not found in the HDF5 file.")
    
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        dataset = f[framepath]
        width, height, num_images = dataset.shape
        num_digits = len(str(num_images))
        os.makedirs(output_dir, exist_ok=True)
        mode = get_image_mode(dataset.dtype)  # Get mode from the dataset's dtype

        print(f"Dataset shape: (num_images={num_images}, height={height}, width={width})")
        print(f"Image mode: {mode}")

        for i in range(num_images):
            img_array = dataset[:, :, i]  # Read the image slice
            if dataset.dtype == np.int16:
                # Convert signed int16 to unsigned int16 by adding 32768
                img_array = (img_array + 32768).astype(np.uint16)
            img = Image.fromarray(img_array, mode=mode)
            filename = os.path.join(output_dir, f"{i+1:0{num_digits}d}.tiff")
            img.save(filename)

            # Progress output
            if (i + 1) % 1000 == 0:  # Print every 100 frames
                print(f"Exported {i + 1} of {num_images} frames...")
