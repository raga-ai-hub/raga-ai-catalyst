import os
import hashlib
import zipfile
import re
import tempfile

def zip_list_of_unique_files(filepaths):
    """
    Generate a unique hash ID for the contents of the files (ignoring whitespace and comments),
    create a .zip of the files, and save it in a temporary folder with the filename <hash_id>.zip.

    Args:
        filepaths (list): List of file paths to process.

    Returns:
        tuple: A tuple containing the hash ID (str) and the path to the saved .zip file (str).
    """
    def clean_file_content(filepath):
        """Read a file and clean its content by ignoring whitespace and comments."""
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            # Remove comments (e.g., # for Python, // for JavaScript)
            content = re.sub(r'#.*|//.*', '', content)
            # Remove all whitespaces
            content = re.sub(r'\s+', '', content)
        return content

    # Sort file paths to ensure order does not affect the hash
    sorted_filepaths = sorted(filepaths)

    # Concatenate cleaned content of all files
    concatenated_content = ''
    for filepath in sorted_filepaths:
        concatenated_content += clean_file_content(filepath)

    # Generate a unique hash ID for the concatenated content
    hash_id = hashlib.sha256(concatenated_content.encode()).hexdigest()

    # Create a temporary directory for the zip file
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, f'{hash_id}.zip')
        
        # Create the zip file
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filepath in sorted_filepaths:
                zipf.write(filepath, os.path.basename(filepath))

        # Move the zip file to a permanent location
        permanent_zip_path = os.path.join(os.getcwd(), f'{hash_id}.zip')
        os.rename(zip_path, permanent_zip_path)

    return hash_id, permanent_zip_path

if __name__ == "__main__":
    filepaths = ["script1.py", "script2.py"]
    zip_list_of_unique_files(filepaths)