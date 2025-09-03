import os
import configparser
import datetime
import ctypes

def log_start(logfile_path, message):
    print(message)
    if logfile_path is not None:
        with open(f'{logfile_path}', 'a') as file:
            file.write(f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}; {message}\n')

def log_result(logfile_path, message, error):
    if error is None:
        print(message)
        if logfile_path is not None:
            with open(f'{logfile_path}', 'a') as file:
                file.write(f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}; {message}\n')
    else:
        print(error)
        print("+++ This error might already be fixed. Did you pull the latest version? +++")
        if logfile_path is not None:
            with open(f'{logfile_path}', 'a') as file:
                file.write(f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}; Error: {error}\n')

def shoutout(configfile_path):
    print(f'working with {configfile_path}')

def is_parent_config_file(file_path):
    # Check if the given INI file has a [General] section with config_type = parent
    config = configparser.ConfigParser()
    config.read(file_path)
    if config.has_section('General') and config.has_option('General', 'config_type'):
        config_type = config.get('General', 'config_type').strip().lower()
        if config_type == 'parent':
            return True
    return False

def handle_input(input_path):
    configfiles = []
    new_input_path = input_path

    # Check if the input is a list of files
    if isinstance(input_path, list):
        # Check if all files are in the same directory and are .ini files
        directory = None
        valid_files = []
        for path in input_path:
            if not os.path.isfile(path) or not path.endswith('.ini') or os.path.basename(path).startswith('.'):
                raise ValueError(f"Each item in the list must be a non-hidden .ini file. Invalid path: {path}")

            if is_parent_config_file(path):
                continue  # Skip files with config_type = parent

            current_directory, _ = os.path.split(path)
            if directory is None:
                directory = current_directory
            elif directory != current_directory:
                raise ValueError("All files in the list must be in the same directory.")

            valid_files.append(path)

        configfiles = valid_files
        new_input_path = directory

    elif os.path.isdir(input_path):
        # List all non-hidden .ini files in the input folder, excluding macOS resource fork files
        all_ini_files = [
            os.path.join(input_path, file) for file in os.listdir(input_path)
            if file.endswith('.ini') and not file.startswith('.') and not file.startswith('._')
        ]
        # Filter out files with config_type = parent
        configfiles = (file_path for file_path in all_ini_files if not is_parent_config_file(file_path))
        

    elif os.path.isfile(input_path) and input_path.endswith('.ini') and not os.path.basename(input_path).startswith('.'):
        # If it's a single non-hidden .ini file
        if is_parent_config_file(input_path):
            raise ValueError(f"The file '{input_path}' is a parent config file and cannot be processed.")
        else:
            configfiles = [input_path]
            new_input_path, _ = os.path.split(input_path)

    else:
        # Handle cases where the input is neither a folder, a valid file, nor a list
        raise ValueError("Input must be a folder, a non-hidden .ini file, or a list of non-hidden .ini files")

    # Check if any valid config files are found
    if not configfiles:
        raise ValueError("No valid configuration files found after filtering out parent config files.")

    return configfiles, new_input_path

def parse_config(configfile):
    config = configparser.ConfigParser()

    # Try reading the file with 'utf-8' encoding first
    try:
        with open(configfile, 'r', encoding='utf-8') as f:
            config.read_file(f)
    except UnicodeDecodeError:
        # If 'utf-8' fails, try 'latin-1' encoding
        try:
            with open(configfile, 'r', encoding='latin-1') as f:
                config.read_file(f)
        except UnicodeDecodeError:
            # If both 'utf-8' and 'latin-1' fail, raise an error
            raise UnicodeDecodeError(f"Cannot decode file {configfile}. Please check its encoding.")

    # Required parameters
    outputfolder = config.get('Paths', 'outputfolder')

    if config.has_option('Paths', 'originalfile'):
            originalfile = config.get('Paths', 'originalfile')
    elif config.has_option('Paths', 'originalfolder'):
            originalfile = config.get('Paths', 'originalfolder')

    logfile = config.get('Paths', 'logfile')
    path, _ = os.path.split(configfile)
    outputfolder_path = os.path.join(path, outputfolder)
    originalfile_path = os.path.join(path, originalfile)
    logfile_path = os.path.join(outputfolder_path, logfile)
    
    # Optional parameters
    framepath = config.get('Paths','framepath', fallback=None)
    h5file = config.get('Paths', 'h5file', fallback=None)
    h5file_path = os.path.join(outputfolder_path, h5file) if h5file else None

    return config, outputfolder, originalfile, logfile, path, outputfolder_path, originalfile_path, logfile_path, framepath, h5file, h5file_path

def get_free_space_windows(path):
    free_bytes = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes))
    return free_bytes.value

def get_free_space_unix(path):
    statvfs = os.statvfs(path)
    return statvfs.f_frsize * statvfs.f_bavail

def get_dir_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def config_to_paths(configfile: str):
    """
    Given the path to a .ini configfile, determine and return:
    outputfolder, outputfolder_path, logfile, logfile_path, h5file, h5file_path

    Path A: if <basename>.h5 exists in the same directory as the .ini:
      - outputfolder: basename of the ini (without extension)
      - outputfolder_path: same directory as ini
      - logfile: <basename>.log
      - logfile_path: same directory as ini
      - h5file: <basename>.h5
      - h5file_path: same directory as ini
    
    Path B: otherwise, read the [Paths] section of the INI:
      - outputfolder: config.get('Paths','outputfolder')
      - outputfolder_path: parent_dir/outputfolder
      - logfile: config.get('Paths','logfile')
      - logfile_path: parent_dir/logfile
      - h5file: config.get('Paths','h5file')
      - h5file_path: outputfolder_path/h5file
    """
    parent_dir = os.path.dirname(os.path.abspath(configfile))
    basename = os.path.splitext(os.path.basename(configfile))[0]

    # Path A: check for HDF5 alongside INI
    candidate_h5 = os.path.join(parent_dir, f"{basename}.h5")
    if os.path.isfile(candidate_h5):
        outputfolder = basename
        outputfolder_path = parent_dir
        logfile = f"{basename}.log"
        logfile_path = os.path.join(parent_dir, logfile)
        h5file = f"{basename}.h5"
        h5file_path = candidate_h5
        return outputfolder, outputfolder_path, logfile, logfile_path, h5file, h5file_path

    # Path B: read from INI’s [Paths] section
    config = configparser.ConfigParser()
    config.read(configfile)
    section = 'Paths'
    outputfolder = config.get(section, 'outputfolder', fallback=None)
    logfile = config.get(section, 'logfile', fallback=None)
    h5file = config.get(section, 'h5file', fallback=None)

    outputfolder_path = os.path.join(parent_dir, outputfolder) if outputfolder else parent_dir
    logfile_path = os.path.join(parent_dir, logfile) if logfile else None
    h5file_path = os.path.join(outputfolder_path, h5file) if outputfolder and h5file else None

    return outputfolder, outputfolder_path, logfile, logfile_path, h5file, h5file_path

def read_config(configfile):
    """
    Reads a configuration file and returns the config object.
    """
    config = configparser.ConfigParser()

    # Try reading the file with 'utf-8' encoding first
    try:
        with open(configfile, 'r', encoding='utf-8') as f:
            config.read_file(f)
    except UnicodeDecodeError:
        # If 'utf-8' fails, try 'latin-1' encoding
        try:
            with open(configfile, 'r', encoding='latin-1') as f:
                config.read_file(f)
        except UnicodeDecodeError:
            # If both 'utf-8' and 'latin-1' fail, raise an error
            raise UnicodeDecodeError(f"Cannot decode file {configfile}. Please check its encoding.")
        
    return config

def update_paths_old(configfile_path):        # Update paths
    config = configparser.ConfigParser()
    config.read(configfile_path)

    outputfolder = config.get('Paths', 'outputfolder')
    logfile = config.get('Paths', 'logfile')
    path, _ = os.path.split(configfile_path)
    logfile_path = os.path.join(path, outputfolder, logfile)
