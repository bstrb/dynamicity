import os
import configparser
import datetime
import platform
import psutil
import sys
import json
import h5py
from coseda.h5convert import find_dataset_paths
from coseda.import_velox import extract_instrument_data
from coseda.io import log_start, log_result, handle_input, parse_config, shoutout, is_parent_config_file, read_config, config_to_paths
from coseda.gatan_metareader import extract_info_from_gatan_metadata

def find_configfiles(input_path):
    configfiles, _ = handle_input(input_path)
    return configfiles

def create_insfiles(input_path, input_type, target=None):
    if input_type == 'dm4_folder':
        create_insfiles_gatan(input_path)
    elif input_type == 'emd' or input_type == '.emd':
        result = create_insfiles_emd(input_path, target)
        return result
    else:
        create_insfiles_emd(input_path, target)

def create_insfiles_emd(input_path, target):
        '''
        # Check if input_path is a string (folder path) or a list (specific file paths)
        if isinstance(input_path, str):
            # List all .emd files in the input folder
            originalfiles = [file for file in os.listdir(input_path) if file.endswith(input_type)]
            originalfiles = [os.path.join(input_path, file) for file in originalfiles]
        elif isinstance(input_path, list):
            # Filter out only .emd files from the list
            originalfiles = [file for file in input_path if file.endswith(input_type)]
            print(originalfiles)
        else:
            print("Invalid input. Please provide a folder path or a list of file paths.")
            return False'''

        input_type = 'emd'

        if isinstance(input_path, str):
            if os.path.isdir(input_path):
                # Input is a folder path, collect all .emd files in the folder
                originalfiles = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(f'.{input_type}')]
            elif os.path.isfile(input_path) and input_path.endswith(f'.{input_type}'):
                # Input is a single .emd file
                originalfiles = [input_path]
            else:
                raise ValueError(f"The input must be a valid .{input_type} file path or a folder containing .{input_type} files.")
        elif isinstance(input_path, list):
            # Input is a list of .emd files, validate all paths
            originalfiles = [file for file in input_path if file.endswith(f'.{input_type}') and os.path.isfile(file)]
        else:
            raise ValueError("Invalid input. Please provide a folder path, a single file path, or a list of file paths.")

        if not originalfiles:
            raise ValueError(f"No .{input_type} files found in the provided input.")

        for file_path in originalfiles:
            if ' ' in os.path.basename(file_path):
                print("You're trying to use files with spaces in their names. Considering you might need to work on this data in a Unix system, this is stupid and I refuse to do this! Remove the spaces and try again.")
                return False

        newfiles = []
        
        for originalfile_path in originalfiles:
            # Create the configfile and folder structure
            directory = os.path.dirname(originalfile_path)
            filename, _ = os.path.splitext(os.path.basename(originalfile_path))
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Create output project folder for HDF5 and log
            outputfolder = f"{filename}_run_{timestamp}"
            outputfolder_path = os.path.join(directory, outputfolder)
            os.makedirs(outputfolder_path, exist_ok=True)
            os.makedirs(os.path.join(outputfolder_path, "plots"), exist_ok=True)

            # Define INS file inside the project folder
            configfile = f"{filename}_run_{timestamp}.ini"
            configfile_path = os.path.join(outputfolder_path, configfile)
            framepath = 'entry/data/images'

            print('configuration file created for ' + originalfile_path)

            instrument_data = extract_instrument_data(input_type, originalfile_path)

            with open(configfile_path, 'w') as configfile_f:
                pass

            print(f"configuration will be written to {configfile_path}")

            newfiles.append(configfile_path)

            print(f"results will be saved in {outputfolder_path}")

            # Create log file
            logfile = f"{filename}_run_{timestamp}.log"
            logfile_path = os.path.join(outputfolder_path, logfile)

            # Open log file, add some structure and basic information
            config = read_config(configfile_path)

            if not config.has_section('Paths'):
                config.add_section('Paths')
                
            config.set('Paths', 'framepath', framepath)
            config.set('Paths', 'originalfile', originalfile_path)

            log_start(logfile_path, f'original file is {originalfile_path}')

            if instrument_data is not None:
                if not config.has_section('AcquisitionDetails'):
                    config.add_section('AcquisitionDetails')

                config.set('AcquisitionDetails', 'acquisition_start', str(instrument_data.get('AcquisitionStart', '')))
                config.set('AcquisitionDetails', 'acquisition_end', str(instrument_data.get('AcquisitionEnd', '')))
                config.set('AcquisitionDetails', 'instrument_manufacturer', str(instrument_data.get('InstrumentManufacturer', '')))
                config.set('AcquisitionDetails', 'instrument_model', str(instrument_data.get('InstrumentModel', '')))
                config.set('AcquisitionDetails', 'instrument_id', str(instrument_data.get('InstrumentId', '')))
                config.set('AcquisitionDetails', 'detector_used', str(instrument_data.get('DetectorUsed', '')))
                config.set('AcquisitionDetails', 'acceleration_voltage', str(instrument_data.get('AccelerationVoltage', '')))
                config.set('AcquisitionDetails', 'camera_length', str(instrument_data.get('CameraLength', '')))
                config.set('AcquisitionDetails', 'pixel_width', str(instrument_data.get('PixelWidth', '')))
                config.set('AcquisitionDetails', 'pixel_height', str(instrument_data.get('PixelHeight', '')))
                config.set('AcquisitionDetails', 'binning_width', str(instrument_data.get('BinningWidth', '')))
                config.set('AcquisitionDetails', 'binning_height', str(instrument_data.get('BinningHeight', '')))
                resolution = instrument_data.get('Resolution', ('', ''))
                config.set('AcquisitionDetails', 'resolution_width', str(resolution[0]))
                config.set('AcquisitionDetails', 'resolution_height', str(resolution[1]))
                config.set('AcquisitionDetails', 'exposure_time', str(instrument_data.get('ExposureTime', '')))
                config.set('AcquisitionDetails', 'pixel_unit', str(instrument_data.get('PixelUnit', '')))
                config.set('AcquisitionDetails', 'pixel_offset_x', str(instrument_data.get('PixelOffsetX', '')))
                config.set('AcquisitionDetails', 'pixel_offset_y', str(instrument_data.get('PixelOffsetY', '')))

                print('successfully extracted acquisition details from original file')

            if not config.has_section('Parameters'):
                config.add_section('Parameters')

            if not config.has_section('Output'):
                config.add_section('Output')

            with open(configfile_path, 'w') as cfgfile:
                config.write(cfgfile)

            log_start(logfile_path, '.ini file created')
            log_start(logfile_path, f'system: {platform.system()} {platform.architecture()}, version {platform.version()}')
            log_start(logfile_path, f'hardware: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores, {round(psutil.virtual_memory().total / (1024 ** 3))} GB total memory')
            log_start(logfile_path, f'python version {sys.version}')
            log_start(logfile_path, f'data acquisition details: {str(instrument_data)}')
        return newfiles

def create_plain_insfile(hdf5_file_path, output_folder, acquisition_details, parameters):
    """
    Create a plain .ins file for an existing HDF5 file with manually provided inputs.

    Parameters:
    - hdf5_file_path (str): Path to the existing HDF5 file.
    - output_folder (str): Path to the folder where the .ins file will be saved.
    - acquisition_details (dict): Dictionary containing the acquisition details.
    - parameters (dict): Dictionary containing the parameters.

    Returns:
    - str: Path to the created .ins file.
    """
    if not os.path.isfile(hdf5_file_path):
        raise ValueError(f"The provided HDF5 file path '{hdf5_file_path}' does not exist or is not a file.")

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(hdf5_file_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output folder structure
    outputfolder = f"{filename}_run_{timestamp}"
    outputfolder_path = os.path.join(output_folder, outputfolder)
    os.makedirs(outputfolder_path)
    os.makedirs(os.path.join(outputfolder_path, "plots"))

    # Define the .ins file path inside the project folder
    insfile_path = os.path.join(outputfolder_path, f"{filename}_run_{timestamp}.ini")

    # Create the .ins file
    config = configparser.ConfigParser()

    # Add the Paths section
    config['Paths'] = {
        'h5file': f"{filename}_run_{timestamp}.h5",
        'originalfile_path': hdf5_file_path,
        'outputfolder': outputfolder
    }
    # Record the INI path in the INI
    config.set('Paths', 'configfile_path', insfile_path)

    # Add the provided acquisition details
    if not config.has_section('AcquisitionDetails'):
        config.add_section('AcquisitionDetails')
    for key, value in acquisition_details.items():
        config.set('AcquisitionDetails', key, str(value))

    # Add the provided parameters
    if not config.has_section('Parameters'):
        config.add_section('Parameters')
    for key, value in parameters.items():
        config.set('Parameters', key, str(value))

    # Create log file
    logfile = f"{filename}_run_{timestamp}.log"
    logfile_path = os.path.join(outputfolder_path, logfile)

    # Add log file path to the config
    config.set('Paths', 'logfile', logfile)
    config.set('Paths', 'framepath', 'entry/data/images')

    # Write the .ins file
    with open(insfile_path, 'w') as cfgfile:
        config.write(cfgfile)

    # Log the creation of the .ins file
    with open(logfile_path, 'w') as log:
        log.write(f".ins file created at: {insfile_path}\n")
        log.write(f"System: {platform.system()} {platform.architecture()}, version {platform.version()}\n")
        log.write(f"Hardware: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores, {round(psutil.virtual_memory().total / (1024 ** 3))} GB total memory\n")
        log.write(f"Python version: {sys.version}\n")
        log.write(f"Acquisition details: {acquisition_details}\n")
        log.write(f"Parameters: {parameters}\n")

    print(f"Plain .ins file created at: {insfile_path}")
    return insfile_path

def create_insfiles_gatan(input_path, target=None):
    filename = os.path.basename(input_path)
    directory = os.path.dirname(input_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create project folder
    outputfolder = f"{filename}_run_{timestamp}"
    if target is None:
        outputfolder_path = os.path.join(directory, outputfolder)
    else:
        outputfolder_path = os.path.join(target, outputfolder)
    os.makedirs(outputfolder_path, exist_ok=True)
    os.makedirs(os.path.join(outputfolder_path, "plots"), exist_ok=True)

    # Define INS file inside the project folder
    configfile = f"{filename}_run_{timestamp}.ini"
    configfile_path = os.path.join(outputfolder_path, configfile)
    framepath = 'entry/data/images'

    newfiles = configfile_path

    print('configuration file created for ' + input_path)

    with open(configfile_path, 'w') as configfile:
            pass

    print(f"configuration will be written to {configfile_path}")

    print(f"results will be saved in {outputfolder_path}")

    # Create log file
    logfile = f"{filename}_run_{timestamp}.log"
    logfile_path = os.path.join(outputfolder_path, logfile)

    # Open log file, add some structure and basic information
    config = read_config(configfile_path)

    if not config.has_section('Paths'):
        config.add_section('Paths')
    config.set('Paths', 'framepath', framepath)

    if not config.has_section('AcquisitionDetails'):
        config.add_section('AcquisitionDetails')

    try:
        exposure_time, resolution_height, resolution_width, binning_height, binning_width, acceleration_voltage, pixel_height, pixel_width, pixel_unit, camera_length, detector_used, instrument_id, instrument_manufacturer, instrument_model, acquisition_start, acquisition_end = extract_info_from_gatan_metadata(input_path)
    except:
        print('No valid metadata found in file')
    
    config.set('AcquisitionDetails', 'acquisition_start', str(acquisition_start))
    config.set('AcquisitionDetails', 'acquisition_end', str(acquisition_end))
    config.set('AcquisitionDetails', 'instrument_manufacturer', str(instrument_manufacturer))
    config.set('AcquisitionDetails', 'instrument_model', str(instrument_model))
    config.set('AcquisitionDetails', 'instrument_id', str(instrument_id))
    config.set('AcquisitionDetails', 'detector_used', str(detector_used))
    config.set('AcquisitionDetails', 'acceleration_voltage', str(acceleration_voltage))
    config.set('AcquisitionDetails', 'camera_length', str(camera_length))
    config.set('AcquisitionDetails', 'pixel_width', str(pixel_width))
    config.set('AcquisitionDetails', 'pixel_height', str(pixel_height))
    config.set('AcquisitionDetails', 'binning_width', str(binning_width))
    config.set('AcquisitionDetails', 'binning_height', str(binning_height))
    config.set('AcquisitionDetails', 'resolution_width', str(resolution_width))
    config.set('AcquisitionDetails', 'resolution_height', str(resolution_height))
    config.set('AcquisitionDetails', 'exposure_time', str(exposure_time))
    config.set('AcquisitionDetails', 'pixel_unit', str(pixel_unit))

    if not config.has_section('Parameters'):
            config.add_section('Parameters')

    if not config.has_section('Output'):
        config.add_section('Output')

    with open(configfile_path, 'w') as cfgfile:
        config.write(cfgfile)

    log_start(logfile_path, '.ini file created')
    log_start(logfile_path, f'system: {platform.system()} {platform.architecture()}, version {platform.version()}')
    log_start(logfile_path, f'hardware: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores, {round(psutil.virtual_memory().total / (1024 ** 3))} GB total memory')
    log_start(logfile_path, f'python version {sys.version}')
    log_start(logfile_path, f'data acquisition details: InstrumentManufacturer: {str(instrument_manufacturer)}, InstrumentModel: {str(instrument_model)}, InstrumentId: {str(instrument_id)}, DetectorUsed: {str(detector_used)}, AccelerationVoltage: {str(acceleration_voltage)}, CameraLength: {camera_length}, PixelWidth: {pixel_width} PixelHeight: {pixel_height}, BinningWidth: {binning_width}, BinningHeight: {binning_height}, Resolution: ({resolution_width}, {resolution_height}), ExposureTime: {exposure_time}, PixelUnit: {pixel_unit}, AcquisitionStart: {acquisition_start}, AcquisitionEnd: {acquisition_end}')
    
    return True, newfiles

def resume_processing(input_path):
    configfiles, _ = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config = read_config(configfile)

        config.set('Paths', 'configfile_path', configfile)

        with open(configfile, 'w', encoding='utf-8') as cfgfile:
            config.write(cfgfile)

        _, _, _, logfile_path, _, _ =  config_to_paths(configfile)
        log_start(logfile_path, f'resuming processing, location of .ini file updated')
        log_start(logfile_path, f'system: {platform.system()} {platform.architecture()}, version {platform.version()}')
        log_start(logfile_path, f'hardware: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores, {round(psutil.virtual_memory().total / (1024 ** 3))} GB total memory')
        log_start(logfile_path, f'python version {sys.version}')


def write_h5conversionsettings(input_path, limit_frames, bin_factor, sample_class):
    configfiles, _ = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config = read_config(configfile)

        # Set parameters
        config.set('Parameters', 'sample_class', f'{sample_class}')
        config.set('Parameters', 'h5conversion_limit_frames', f'{limit_frames}')
        config.set('Parameters', 'h5conversion_bin_factor', f'{bin_factor}')

        with open(configfile, 'w') as cfgfile:
            config.write(cfgfile)

        _, _, _, logfile_path, _, _ =  config_to_paths(configfile)

        log_start(logfile_path, f'sample class set to: {sample_class}')
        log_start(logfile_path, f'HDF5 conversion parameters set, limit_frames = {limit_frames}, bin_factor = {bin_factor}')
       
def write_peakfindersettings( input_path, threshold, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res, x0=None, y0=None, num_threads=None):
    configfiles, _ = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config = read_config(configfile)

        # Set parameters
        parameters_to_set = {
            'peakfinding_threshold': threshold,
            'peakfinding_min_snr': min_snr,
            'peakfinding_min_pix_count': min_pix_count,
            'peakfinding_max_pix_count': max_pix_count,
            'peakfinding_local_bg_radius': local_bg_radius,
            'peakfinding_min_res': min_res,
            'peakfinding_max_res': max_res
        }

        # Add 'num_threads' if provided
        if num_threads is not None:
            parameters_to_set['peakfinding_num_threads'] = num_threads

        # Set x0 and y0, defaulting to 'None' if not provided
        parameters_to_set['peakfinding_x0'] = x0 if x0 is not None else 'None'
        parameters_to_set['peakfinding_y0'] = y0 if y0 is not None else 'None'

        for param, value in parameters_to_set.items():
            config.set('Parameters', param, str(value))

        # Write configfile and log
        with open(configfile, 'w') as cfgfile:
            config.write(cfgfile)

        _, _, _, logfile_path, _, _ =  config_to_paths(configfile)

        # Prepare log messages
        log_message = (
            f'peakfinder settings set, threshold = {threshold}, '
            f'min_snr = {min_snr}, min_pix_count = {min_pix_count}, '
            f'max_pix_count = {max_pix_count}, local_bg_radius = {local_bg_radius}, '
            f'min_res = {min_res}, max_res = {max_res}'
        )

        if num_threads is not None:
            log_message += f', num_threads = {num_threads}'
        else:
            log_message += ', num_threads not set'

        log_start(logfile_path, log_message)

        geometry_message = (
            f'detector geometry set, x0 = {x0 if x0 is not None else "None"}, '
            f'y0 = {y0 if y0 is not None else "None"}'
        )
        log_start(logfile_path, geometry_message)

        # Parameter sensibility checks
        if max_pix_count <= 2 * min_pix_count:
            log_start(
                logfile_path,
                'Warning: max_pix_count is close or equal to min_pix_count, '
                'attempting to find very specific peak sizes may lead to a small number of identified peaks.'
            )

        if max_res <= 2 * min_res:
            log_start(
                logfile_path,
                'Warning: max_res is close or equal to min_res, this might prevent peakfinder8 from identifying any peaks.'
            )

def write_centerfindersettings(input_path, tolerance, min_peaks, resolution_limit, min_samples_fraction, force_linear_fit=False, x0=None, y0=None):
    configfiles, input_path = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config = read_config(configfile)
        
        # Set parameters
        config.set('Parameters', 'centerfinding_tolerance', f'{tolerance}')
        config.set('Parameters', 'centerfinding_min_peaks', f'{min_peaks}')
        config.set('Parameters', 'centerfinding_resolution_limit', f'{resolution_limit}')
        config.set('Parameters', 'centerfinding_min_samples_fraction', f'{min_samples_fraction}')
        config.set('Parameters', 'centerfinding_x0', f'{x0}')
        config.set('Parameters', 'centerfinding_y0', f'{y0}')
        config.set('Parameters', 'centerfinding_force_linear_fit', str(force_linear_fit))

        # Write the file with 'utf-8' encoding
        with open(configfile, 'w') as cfgfile:
            config.write(cfgfile)

        _, _, _, logfile_path, _, _ =  config_to_paths(configfile)

        log_start(logfile_path, f'parameters set for center finding, tolerance = {tolerance}, min_peaks = {min_peaks}, resolution_limit = {resolution_limit}, min_samples_fraction = {min_samples_fraction}, force_linear_fit = {force_linear_fit}, x0 = {x0}, y0 = {y0}')        

def write_centerrefinementsettings(input_path, tolerance, min_peaks, resolution_limit, min_samples_fraction, max_iterations, convergence_threshold):
    configfiles, input_path = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config = read_config(configfile)

        # Set parameters
        config.set('Parameters', 'centerrefinement_tolerance', f'{tolerance}')
        config.set('Parameters', 'centerrefinement_min_peaks', f'{min_peaks}')
        config.set('Parameters', 'centerrefinement_resolution_limit', f'{resolution_limit}')
        config.set('Parameters', 'centerrefinement_min_samples_fraction', f'{min_samples_fraction}')
        config.set('Parameters', 'centerrefinement_max_iterations', f'{max_iterations}')
        config.set('Parameters', 'centerrefinement_convergence_threshold', f'{convergence_threshold}')

        # Write the file with 'utf-8' encoding
        with open(configfile, 'w') as cfgfile:
            config.write(cfgfile)

        _, _, _, logfile_path, _, _ =  config_to_paths(configfile)

        log_start(logfile_path, f'parameters set for refinement of center, tolerance = {tolerance}, min_peaks = {min_peaks}, resolution_limit = {resolution_limit}, min_samples_fraction = {min_samples_fraction}, max_iterations = {max_iterations}, convergence_threshold = {convergence_threshold}')

def get_parent_ini_file(input_path):
    """
    Accepts an input path which is either a parent INI file or a folder containing one parent INI file.
    Returns the path to the parent INI file.
    """
    if os.path.isfile(input_path):
        # Input is a file
        if input_path.endswith('.ini'):
            if is_parent_config_file(input_path):
                return input_path
            else:
                raise ValueError(f"The file '{input_path}' is not a parent INI file.")
        else:
            raise ValueError(f"The file '{input_path}' is not an INI file.")
    elif os.path.isdir(input_path):
        # Input is a directory
        ini_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.ini')]
        parent_ini_files = [f for f in ini_files if is_parent_config_file(f)]
        if len(parent_ini_files) == 1:
            return parent_ini_files[0]
        elif len(parent_ini_files) == 0:
            raise ValueError(f"No parent INI file found in the directory '{input_path}'.")
        else:
            raise ValueError(f"Multiple parent INI files found in the directory '{input_path}'. Please specify one.")
    else:
        raise ValueError(f"The input path '{input_path}' is neither a file nor a directory.")

def write_gandalfiteratorsettings(
    configfile,
    geomfile_path=None,
    cellfile_path=None,
    output_file_base=None,
    threads=None,
    max_radius=None,
    step=None,
    peakfinder_method=None,
    peakfinder_params=None,
    min_peaks=None,
    cell_tolerance=None,
    sampling_pitch=None,
    grad_desc_iterations=None,
    xgandalf_tolerance=None,
    int_radius=None,
    other_flags=None
):
    """
    Update the parent INI file with Gandalf-iterator settings under [Parameters].
    Does NOT require input folder (list file is used instead).
    """
    shoutout(configfile)
    outputfolder, outputfolder_path, logfile, logfile_path, h5file, h5file_path =  config_to_paths(configfile)
    config = read_config(configfile)

    # Ensure 'Paths' section exists and set geometry, cell, output_base
    if not config.has_section('Paths'):
        config.add_section('Paths')
    if geomfile_path is not None:
        config.set('Paths', 'geomfile_path', geomfile_path)
    if cellfile_path is not None:
        config.set('Paths', 'cellfile_path', cellfile_path)
    if output_file_base is not None:
        config.set('Paths', 'output_file_base', output_file_base)

    # Prepare parameters to set under [Parameters]
    params = {
        'gandalfiterations_threads': threads,
        'gandalfiterations_max_radius': max_radius,
        'gandalfiterations_step': step,
        'gandalfiterations_peakfinder_method': peakfinder_method,
        'gandalfiterations_min_peaks': min_peaks,
        'gandalfiterations_cell_tolerance': cell_tolerance,
        'gandalfiterations_sampling_pitch': sampling_pitch,
        'gandalfiterations_grad_desc_iterations': grad_desc_iterations,
        'gandalfiterations_xgandalf_tolerance': xgandalf_tolerance,
        'gandalfiterations_int_radius': int_radius
    }

    # Add peakfinder_params and other_flags as multiline if provided
    if peakfinder_params is not None:
        params['gandalfiterations_peakfinder_params'] = "\n".join(peakfinder_params)
    if other_flags is not None:
        params['gandalfiterations_other_flags'] = "\n".join(other_flags)

    if not config.has_section('Parameters'):
        config.add_section('Parameters')

    # Write all parameters, only writing those that are provided (not None)
    for key, value in params.items():
        if value is not None:
            config.set('Parameters', key, str(value))
        # If value is None, skip writing the key (leave any existing value untouched)

    # Write the updated config file
    with open(configfile, 'w', encoding='utf-8') as cfgfile:
        config.write(cfgfile)

    # Log the settings updated
    log_message = 'Gandalf iterator settings updated: ' + ', '.join(
        f'{k} = {config.get("Parameters", k)}' for k in params.keys()
    )
    log_start(logfile_path, log_message)

    # Log Paths settings for clarity
    paths_keys = ['geomfile_path', 'cellfile_path', 'output_file_base']
    paths_settings = [
        f'{key} = {config.get("Paths", key)}' for key in paths_keys if config.has_option('Paths', key)
    ]
    if paths_settings:
        paths_message = 'Gandalf iterator paths updated: ' + ', '.join(paths_settings)
        log_start(logfile_path, paths_message)

def write_stripsettings(input, strip_threshold, strip_force):
    configfiles, _ = handle_input(input)
    num_configfiles = len(configfiles)

    # Check if strip_force is True and multiple config files are provided
    if strip_force and num_configfiles > 1:
        raise ValueError("I refuse to set the force flag for all files. Please review your peakfinding results for each file and set the flag manually for each file if necessay.")

    for configfile in configfiles:
        config = read_config(configfile)
        shoutout(configfile)

        # Ensure 'Parameters' section exists
        if not config.has_section('Parameters'):
            config.add_section('Parameters')

        # Set parameters
        config.set('Parameters', 'strip_threshold', f'{strip_threshold}')
        config.set('Parameters', 'strip_force', f'{strip_force}')

        # Write the updated config file with 'utf-8' encoding
        with open(configfile, 'w', encoding='utf-8') as f:
            config.write(f)

        outputfolder, outputfolder_path, logfile, logfile_path, h5file, h5file_path =  config_to_paths(configfile)

        log_start(logfile_path, f"Parameters set for stripping: strip_threshold = {strip_threshold}, strip_force = {strip_force}")


def write_index_file(input_path):
    configfiles, _ = handle_input(input_path)
    print(input_path)

    h5list = []

    for configfile in configfiles:
        shoutout(configfile)
        config = read_config(configfile)

        outputfolder, outputfolder_path, logfile, logfile_path, h5file, h5file_path = config_to_paths(configfile)

        h5list.append(h5file_path)

    outputpath = os.path.join(outputfolder_path, 'template.list')

    with open(outputpath, 'w') as file:
        for h5 in h5list:
            file.write(h5 + '\n')

def write_mergingsettings(
    input_path,
    hkl_filename,
    pointgroup,
    min_res,
    iterations,
    threads=None,
    model="offset",
    polarisation="none",
    min_measurements=2,
    max_adu="inf",
    push_res="inf",
    no_Bscale=True,
    output_every_cycle=True,
    unique_axis=None,
    streamfile=None
):
    """
    Write merging parameters into the INI file.
    'hkl_filename' is saved under [Paths], all other parameters under [Parameters]
    with keys prefixed by 'merging_'.
    """
    # Determine list of config files (INI) to update
    configfiles, base_path = handle_input(input_path)

    for configfile in configfiles:
        shoutout(configfile)
        config = read_config(configfile)

        # Ensure [Paths] section exists and set HKL filename
        if not config.has_section('Paths'):
            config.add_section('Paths')
        config.set('Paths', 'hkl_file', hkl_filename)
        # Write streamfile if provided
        if streamfile is not None:
            config.set('Paths', 'streamfile', streamfile)

        # Prepare merging parameters under [Parameters]
        params = {
            'merging_pointgroup': pointgroup.replace(" ", ""),
            'merging_min_res': min_res,
            'merging_iterations': iterations,
            'merging_model': model,
            'merging_polarisation': polarisation,
            'merging_min_measurements': min_measurements,
            'merging_max_adu': max_adu,
            'merging_push_res': push_res,
            'merging_no_Bscale': str(no_Bscale),
            'merging_output_every_cycle': str(output_every_cycle)
        }
        if threads is not None:
            params['merging_threads'] = threads
        if unique_axis is not None:
            params['merging_unique_axis'] = unique_axis

        # Ensure [Parameters] section exists
        if not config.has_section('Parameters'):
            config.add_section('Parameters')

        # Write each merging parameter if provided
        for key, value in params.items():
            if value is not None:
                config.set('Parameters', key, str(value))

        # Save back to INI file
        with open(configfile, 'w', encoding='utf-8') as cfgfile:
            config.write(cfgfile)

        _, _, _, logfile_path, _, _ = config_to_paths(configfile)

        log_start(logfile_path, log_message)

        log_message = (
            f"Merging settings set: hkl_file = {hkl_filename}, "
            f"pointgroup = {pointgroup}, min_res = {min_res}, "
            f"iterations = {iterations}, model = {model}, polarisation = {polarisation}, "
            f"min_measurements = {min_measurements}, max_adu = {max_adu}, "
            f"push_res = {push_res}, no_Bscale = {no_Bscale}, "
            f"output_every_cycle = {output_every_cycle}"
        )
        if threads is not None:
            log_message += f", threads = {threads}"
        if unique_axis is not None:
            log_message += f", unique_axis = {unique_axis}"

        log_start(logfile_path, log_message)


def generate_parent_ini_file(folder_path, parent_ini_filename=None):
    # Determine the default parent INI filename if not provided
    if parent_ini_filename is None:
        # Get the absolute path to handle cases where folder_path ends with '/'
        folder_path = os.path.abspath(folder_path)
        # Get the name of the parent folder
        parent_folder_name = os.path.basename(folder_path)
        # Construct the parent INI filename
        parent_ini_filename = f"{parent_folder_name}.ini"

    # Use the handle_input function to get all non-parent INI files
    configfiles, _ = handle_input(folder_path)

    if not configfiles:
        message = "No valid subset INI files found to include in the parent INI file."
        log_start(None, message)
        return None

    # Create the parent INI file
    parent_config = configparser.ConfigParser()

    # Add the General section with config_type = parent
    parent_config['General'] = {'config_type': 'parent'}

    # Add the Subsets section
    parent_config['Subsets'] = {}

    # Add each child INI file under the Subsets section
    parent_ini_path = os.path.join(folder_path, parent_ini_filename)
    parent_dir = os.path.dirname(parent_ini_path)

    for idx, cfg in enumerate(configfiles, start=1):
        # Get the relative path to the child INI file
        relative_path = os.path.relpath(cfg, parent_dir)
        key_name = f'subset{idx}'
        parent_config['Subsets'][key_name] = relative_path

    # Add the Paths section
    parent_config['Paths'] = {}

    # Set configfile_path to the absolute path of the parent INI file
    parent_config['Paths']['configfile_path'] = parent_ini_path

    # Set logfile to have the same name as the INI file but with .log extension
    log_filename = os.path.splitext(parent_ini_filename)[0] + '.log'
    parent_config['Paths']['logfile'] = log_filename

    # Set framepath (assuming it's a fixed value)
    parent_config['Paths']['framepath'] = 'entry/data/images'

    # Write the parent INI file
    with open(parent_ini_path, 'w') as parent_ini_file:
        parent_config.write(parent_ini_file)

    # Construct the log file path
    log_file_path = os.path.join(folder_path, log_filename)

    # Use log_start to create the log file and write the initial message
    if not os.path.exists(log_file_path):
        initial_message = f"Log file created for {parent_ini_filename}"
        log_start(log_file_path, initial_message)
    else:
        message = f"Log file '{log_filename}' already exists."
        log_start(log_file_path, message)

    # Log the creation of the parent INI file
    message = f"Parent INI file generated at: {parent_ini_path}"
    log_start(log_file_path, message)

    # Log the list of child INI files included
    child_files = ', '.join([os.path.basename(cfg) for cfg in configfiles])
    message = f"Subset INI files included: {child_files}"
    log_start(log_file_path, message)

    return parent_ini_path

def write_xgandalfsettings(input_path, tolerance, sampling_pitch, min_lattice_vector_length, max_lattice_vector_length, grad_desc_iterations, tolerance_5d, fix_profile_radius):
    """
    Writes Xgandalf settings to the [Parameters] section of each config file determined by handle_input.
    Logs the new settings if [Paths] logfile is present.
    """
    configfiles, _ = handle_input(input_path)

    for configfile in configfiles:
        config = read_config(configfile)

        if not config.has_section('Parameters'):
            config.add_section('Parameters')

        config.set('Parameters', 'xgandalf_tolerance', str(tolerance))
        config.set('Parameters', 'xgandalf_sampling_pitch', str(sampling_pitch))
        config.set('Parameters', 'xgandalf_min_lattice_vector_length', str(min_lattice_vector_length))
        config.set('Parameters', 'xgandalf_max_lattice_vector_length', str(max_lattice_vector_length))
        config.set('Parameters', 'xgandalf_grad_desc_iterations', str(grad_desc_iterations))
        config.set('Parameters', 'tolerance_5d', str(tolerance_5d))
        config.set('Parameters', 'fix_profile_radius', str(fix_profile_radius))

        with open(configfile, 'w', encoding='utf-8') as cfgfile:
            config.write(cfgfile)

        # Resolve logfile path from project folder
        _, _, _, logfile_path, _, _ = config_to_paths(configfile)
        log_start(
            logfile_path,
            f'Xgandalf settings set: tolerance={tolerance}, sampling_pitch={sampling_pitch}, min_lattice_vector_length={min_lattice_vector_length}, max_lattice_vector_length={max_lattice_vector_length}, grad_desc_iterations={grad_desc_iterations}, tolerance_5d={tolerance_5d}, fix_profile_radius={fix_profile_radius}'
        )