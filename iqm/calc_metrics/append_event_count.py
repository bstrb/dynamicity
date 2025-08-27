#!/usr/bin/env python3
import os
import glob
import re

def append_event_count(stream_file_folder):
    """
    Processes all .stream files in the specified folder by appending an occurrence counter
    to event lines that haven't yet been processed.

    For each file, the function scans for the first occurrence of a line starting with "Event: //".
    If that line already shows a counter (i.e. it matches "Event: //NUMBER-digit"), the file is skipped.
    Otherwise, every event line in the file is processed: the first unprocessed occurrence of
    an event number is modified by appending "-1", the next by appending "-2", and so on.

    Args:
        stream_file_folder (str): Path to the folder containing .stream files.
    
    Raises:
        ValueError: If the provided folder path does not exist.
    """
    if not os.path.isdir(stream_file_folder):
        raise ValueError(f"'{stream_file_folder}' is not a valid directory.")

    # Match all files ending with .stream in the provided folder.
    file_pattern = os.path.join(stream_file_folder, "*.stream")
    stream_files = glob.glob(file_pattern)

    if not stream_files:
        print(f"No .stream files found in '{stream_file_folder}'.")
        return

    for filepath in stream_files:
        process_file(filepath)

def process_file(filepath):
    """
    Processes a single .stream file by appending a counter to event lines that are unprocessed.

    A file is considered already processed if its first event line (a line starting with "Event: //")
    already has a counter appended (i.e. the portion immediately following the event number starts with a dash).
    In that case, the file is skipped entirely.
    
    For unprocessed files, each line that begins with "Event: //NUMBER" is modified to have a counter appended.
    For example, the first occurrence "Event: //1" becomes "Event: //1-1", the next "Event: //1" becomes "Event: //1-2", etc.

    Args:
        filepath (str): The path to the .stream file to process.
    """ 
    # Pattern to capture:
    #   Group 1: "Event:" followed by optional spaces and then "//"
    #   Group 2: The event number (one or more digits)
    #   Group 3: The rest of the line (which may or may not start with a dash)
    pattern = re.compile(r'^(Event:\s*//)(\d+)(.*)$')
    
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Check the first event line. If it already has a counter (i.e. rest starts with '-'),
    # we assume the file has already been processed and skip it.
    for line in lines:
        match = pattern.match(line)
        if match:
            prefix, number, rest = match.groups()
            if rest.lstrip().startswith('-'):
                # print(f"Skipping {filepath} as it appears to have been processed already.")
                return
            # Found an event line without a counter; no need to check further.
            break

    # Process the file: append counters to unprocessed event lines.
    counts = {}  # To track the number of times each event number appears.
    new_lines = []
    for line in lines:
        match = pattern.match(line)
        if match:
            prefix, number, rest = match.groups()
            counts[number] = counts.get(number, 0) + 1
            # Append the counter only if the line is unprocessed.
            cleaned_rest = rest.rstrip('\n')
            new_line = f"{prefix}{number}-{counts[number]}{cleaned_rest}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    with open(filepath, 'w') as file:
        file.writelines(new_lines)

if __name__ == "__main__":
    # Replace with the path to your folder containing .stream files
    stream_files_folder = "/Users/xiaodong/Desktop/UOX-simulations/simulation-5-triple"
    
    append_event_count(stream_files_folder)
