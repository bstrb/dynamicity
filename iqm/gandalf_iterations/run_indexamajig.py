import subprocess

def run_indexamajig(geomfile_path, listfile_path, cellfile_path, output_path, num_threads, extra_flags=None):

    if extra_flags is None:
        extra_flags = []

    # Create a list of command parts
    command_parts = [
        "indexamajig",
        "-g", geomfile_path,
        "-i", listfile_path,
        "-o", output_path,
        "-p", cellfile_path,
        "-j", str(num_threads)
    ]

    # Append any extra flags provided by the user.
    command_parts.extend(extra_flags)

    # Join the parts into a single command string.
    base_command = " ".join(command_parts)
    subprocess.run(base_command, shell=True, check=True)
