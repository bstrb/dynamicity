import os

def list_h5_files(input_path):
    """
    Creates or replaces a 'list.lst' file in the specified input_path directory.
    The file contains the full paths of all files ending with '.h5' in the directory,
    sorted alphabetically.
    
    Args:
        input_path (str): The directory path where '.h5' files are located.
    """
    # Path to the list file
    listfile_path = os.path.join(input_path, 'list.lst')
    
    try:
        # List all .h5 files
        h5_files = [file for file in os.listdir(input_path) if file.endswith('.h5') and os.path.isfile(os.path.join(input_path, file))]
        
        # Sort the list alphabetically
        h5_files_sorted = sorted(h5_files, key=lambda x: x.lower())  # Case-insensitive sorting
        
        # Open the list file in write mode to overwrite if it exists
        with open(listfile_path, 'w') as list_file:
            for file in h5_files_sorted:
                full_path = os.path.join(input_path, file)
                list_file.write(full_path + '\n')
        
        print(f"'list.lst' has been created with {len(h5_files_sorted)} entries at {listfile_path}")
    
    except FileNotFoundError:
        print(f"The directory '{input_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied when accessing '{input_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return listfile_path
        
# Example usage:
if __name__ == "__main__":
    directory = "/path/to/your/h5/files"
    list_h5_files(directory)
