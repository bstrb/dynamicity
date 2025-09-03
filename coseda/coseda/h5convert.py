import h5py
import hyperspy.api as hs
import os
import gc
import numpy as np
import configparser
import datetime
import shutil
import mrcfile
from coseda.import_velox import find_dataset_paths, extract_pos_data, refine_stagepos_x, remap_framestack, calculate_mean_intensities, calculate_mean_intensities_chunked, calculate_total_intensities, calculate_total_intensities_chunked, velox_true_conversion
from coseda.gatan_metareader import dm4_folder_conversion
from coseda.io import log_start, log_result, handle_input, parse_config, shoutout, config_to_paths, read_config

def bin_chunk(chunk, bin_factor):
    """
    Bin the image data.
    :param chunk: numpy array of image data
    :param bin_factor: integer binning factor
    :return: binned numpy array
    """
    n_frames, height, width = chunk.shape
    new_height = height // bin_factor
    new_width = width // bin_factor

    # Binning the chunk by reshaping and averaging
    chunk_binned = chunk.reshape(n_frames, new_height, bin_factor, new_width, bin_factor).mean(axis=(2, 4))
    return chunk_binned

def emi_to_h5(input_path):
    configfiles, input_path = handle_input(input_path)
    
    for configfile in configfiles:

        _, outputfolder_path, _, _, _, _ = config_to_paths(configfile)

        base = os.path.basename(outputfolder_path)
        print(f'Processing {base}...')

        logfile_path = os.path.join(outputfolder_path, f'{base}.log')
        new_file_name = f'{base}.h5'
        new_file_path = os.path.join(outputfolder_path, f'{base}.h5')

        config = read_config(configfile)

        originalfile_path = config.get('Paths', 'originalfile')
        originalfile = os.path.basename(originalfile_path)
        framepath = config.get('Paths', 'framepath')

        log_start(logfile_path, f'processing {originalfile_path}')

        # Check if necessary parameters are defined
        for param in ['h5conversion_bin_factor']:
            if not config.has_option('Parameters', param):
                with open(f'{logfile_path}', 'a') as file:
                    file.write(f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}; Error: {param.split("_", 1)[1]} not defined, peak finding interrupted\n')
                raise Exception(f'{param.split("_", 1)[1]} not defined')

        # Load parameters
        limit_frames = False # todo: limit frames option needs to be removed
        bin_factor = int(config.get('Parameters','h5conversion_bin_factor'))

        # Definition of functions
        def bin_chunk(chunk, bin_factor):
            n_frames, height, width = chunk.shape
            new_height = height // bin_factor
            new_width = width // bin_factor

            chunk_binned = chunk.reshape(n_frames, new_height, bin_factor, new_width, bin_factor).mean(axis=(2, 4))
            return chunk_binned

        def process_and_save_h5(originalfile_path, output_h5_file, bin_factor, limit_frames, framepath):
            with open(f'{logfile_path}', 'a') as file:
                file.write(f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}; start HDF5 conversion of {originalfile}, limit_frames = {limit_frames}, bin_factor = {bin_factor}\n')
            
            starttime = datetime.datetime.now()
            
            emi_data = hs.load(originalfile_path, only_valid_data=True)
            
            n_frames, height, width = emi_data.data.shape

            if limit_frames:
                n_frames = min(1000, n_frames)

            if bin_factor > 1:
                new_height = height // bin_factor
                new_width = width // bin_factor
            else:
                new_height, new_width = height, width

            chunk_size = 20

            with h5py.File(output_h5_file, 'w') as h5f:
                dset = h5f.create_dataset(framepath, shape=(n_frames, new_height, new_width), dtype=np.float32)
                

                for i in range(0, n_frames, chunk_size):
                    end = min(i + chunk_size, n_frames)
                    chunk_data = emi_data.data[i:end, :, :].astype(np.float32)
                    
                    if bin_factor > 1:
                        chunk_data = bin_chunk(chunk_data, bin_factor)

                    dset[i:end, :, :] = chunk_data
            
            del emi_data
            gc.collect()
            runtime = datetime.datetime.now() - starttime
            # Log
            with open(f'{logfile_path}', 'a') as file:
                file.write(f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}; HDF5 conversion successful, written to {outputfile}, {n_frames} frames, {height} by {width}px, finished in {runtime}\n')

            return n_frames, height, width, runtime

        outputfile = f"{outputfolder}.h5"if not limit_frames else f"{outputfolder}_for_FIJI.h5"
        output_h5_file = os.path.join(outputfolder_path,outputfile)
        config.set('Paths', 'h5file', f'{outputfile}')
        with open(configfile, 'w') as current_configfile:
            config.write(current_configfile)
                
        n_frames, height, width, runtime = process_and_save_h5(originalfile_path, output_h5_file, bin_factor, limit_frames, framepath)

    return True

def emd_to_h5(input_path):
    configfiles, input_path = handle_input(input_path)
    
    for configfile in configfiles:

        _, outputfolder_path, _, _, _, _ = config_to_paths(configfile)

        base = os.path.basename(outputfolder_path)
        print(f'Processing {base}...')

        logfile_path = os.path.join(outputfolder_path, f'{base}.log')
        new_file_name = f'{base}.h5'
        new_file_path = os.path.join(outputfolder_path, f'{base}.h5')

        config = read_config(configfile)

        originalfile_path = config.get('Paths', 'originalfile')

        log_start(logfile_path, f'Processing {originalfile_path}')

        # Check if the .h5 file already exists
        h5_exists = os.path.exists(new_file_path)

        # Flag to determine if we need to process the initial steps
        need_initial_processing = True

        # If the .h5 file exists, check if the 'refined_stage_positions' dataset exists
        if h5_exists:
            with h5py.File(new_file_path, 'r') as h5f:
                if 'entry/data/stagepos_x_refined' in h5f:
                    log_start(logfile_path, 'Refined stage positions dataset exists. Resuming from frame dataset conversion.')
                    need_initial_processing = False
                else:
                    log_start(logfile_path, 'Refined stage positions dataset not found. Resuming from metadata extraction.')

        if need_initial_processing:
            # Create output folder if it doesn't exist
            if not os.path.exists(outputfolder_path):
                os.makedirs(outputfolder_path)

            # Move and rename the file if it hasn't been moved yet
            if not h5_exists:
                # Move and rename the original .emd file to .h5
                shutil.move(originalfile_path, new_file_path)
                log_start(logfile_path, '.emd file renamed to .h5 and moved to output folder')

            with open(configfile, 'w') as current_configfile:
                config.write(current_configfile)

            # Find paths of datasets in the original file
            framepath, _, datapath = find_dataset_paths(new_file_path)
            log_start(logfile_path, f'Path of frames in original file: {framepath}')
            log_start(logfile_path, f'Path of metadata in original file: {datapath}')

            # Extract stage positions from metadata
            log_start(logfile_path, 'Extracting stage positions from Velox metadata')
            result = extract_pos_data(new_file_path, datapath)
            log_result(logfile_path, 'Stage positions written to new dataset', result)

            # Refine stage positions
            log_start(logfile_path, 'Refining stage positions from Velox')
            result = refine_stagepos_x(new_file_path)
            log_result(logfile_path, 'Stage positions refined', result)
        else:
            # If we skipped initial processing, ensure paths are updated
            h5file_path = new_file_path
            # If framepath is needed later, make sure to retrieve it
            framepath, _, _ = find_dataset_paths(new_file_path)

        # Proceed with frame dataset conversion
        with h5py.File(new_file_path, 'r') as h5f:
            if framepath in h5f:
                frame_dataset = h5f[framepath]
                frame_height, frame_width, n_frames = frame_dataset.shape
            else:
                raise KeyError(f"Frame dataset '{framepath}' not found in '{new_file_path}'.")

        # Determine chunk size based on dataset length
        if n_frames < 1000:
            chunk_size = (n_frames, frame_height, frame_width)
        else:
            chunk_size = (1000, frame_height, frame_width)

        log_start(logfile_path, f'Attempting to rewrite {os.path.basename(configfile)} with chunked frame stack')
        result = velox_true_conversion(new_file_path, chunk_size)
        log_result(logfile_path, 'Conversion successful', result)

        if result is not None:
            return result

def velox_true_conversion_batch(input_path):
    configfiles, input_path = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config, outputfolder, originalfile, logfile, path, outputfolder_path, originalfile_path, logfile_path, framepath, h5file, h5file_path = parse_config(configfile)
        
        log_start(logfile_path, f'attempting to rewrite {os.path.basename(configfile)} with chunked framstack')
        result = velox_true_conversion(h5file_path)
        log_result(logfile_path, 'conversion successful', result)

def dm4_folder_conversion_batch(input_path):
    configfiles, input_path = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config, outputfolder, originalfile, logfile, path, outputfolder_path, originalfile_path, logfile_path, framepath, _, _ = parse_config(configfile)
        extract_timestamp=True
        chunk_size = (1000, 1024, 1024)

        log_start(logfile_path, f'attempting to rewrite {os.path.basename(configfile)} with chunked framstack')
        result, h5file = dm4_folder_conversion(originalfile_path, outputfolder, logfile_path, extract_timestamp, chunk_size)

        # write the name of the newly created h5file to the config file if conversion was successful
        if result is None:
            config = configparser.ConfigParser()
            config.read(configfile)
            config.set('Paths', 'h5file', h5file)

        # Open log file, add some structure and basic information
        log_result(logfile_path, 'conversion successful', result)


def calculate_mean_intensity_per_frame_batch(input_path):
    configfiles, input_path = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config, outputfolder, originalfile, logfile, path, outputfolder_path, originalfile_path, logfile_path, framepath, h5file, h5file_path = parse_config(configfile)
        
        log_start(logfile_path, f'calculating mean intensity per frame {os.path.basename(configfile)}')
        result = calculate_mean_intensities_chunked(h5file_path)
        log_result(logfile_path, 'mean intensities written to h5 file', result)
    

def calculate_total_intensity_per_frame_batch(input_path):
    configfiles, input_path = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config, outputfolder, originalfile, logfile, path, outputfolder_path, originalfile_path, logfile_path, framepath, h5file, h5file_path = parse_config(configfile)


        log_start(logfile_path, f'calculating total intensity per frame {os.path.basename(configfile)}')
        result = calculate_total_intensities_chunked(h5file_path)
        log_result(logfile_path, 'total intensities written to h5 file', result)

        