import os
import hashlib
import zipfile
import re
import ast
import importlib.util
import json
from pathlib import Path

class TraceDependencyTracker:
    def __init__(self, output_dir=None, project_root=None):
        self.tracked_files = set()
        self.python_imports = set()
        self.output_dir = output_dir or os.getcwd()
        self.project_root = project_root or os.getcwd()
        
    def is_project_file(self, filepath):
        """Check if the file is within the project directory."""
        try:
            abs_path = os.path.abspath(filepath)
            return abs_path.startswith(self.project_root)
        except:
            return False

    def track_file_access(self, filepath):
        """Track a file that's been accessed, only if it's within the project."""
        if os.path.exists(filepath) and self.is_project_file(filepath):
            self.tracked_files.add(os.path.abspath(filepath))

    def find_config_files(self, content, base_path):
        """Find configuration files referenced in the content."""
        patterns = [
            r'(?:open|read|load|with\s+open)\s*\([\'"]([^\'"]*\.(?:json|yaml|yml|txt|cfg|config|ini))[\'"]',
            r'(?:config|cfg|conf|settings|file|path)(?:_file|_path)?\s*=\s*[\'"]([^\'"]*\.(?:json|yaml|yml|txt|cfg|config|ini))[\'"]',
            r'[\'"]([^\'"]*\.txt)[\'"]',
            r'[\'"]([^\'"]*\.(?:yaml|yml))[\'"]'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                filepath = match.group(1)
                if not os.path.isabs(filepath):
                    full_path = os.path.join(os.path.dirname(base_path), filepath)
                else:
                    full_path = filepath
                
                if os.path.exists(full_path) and self.is_project_file(full_path):
                    self.track_file_access(full_path)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            self.find_config_files(f.read(), full_path)
                    except (UnicodeDecodeError, IOError):
                        pass

    def analyze_python_imports(self, filepath):
        """Analyze Python file for local project imports only."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read(), filename=filepath)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    # Handle relative imports
                    module_dir = os.path.dirname(filepath)
                    for _ in range(node.level - 1):
                        module_dir = os.path.dirname(module_dir)
                    if node.module:
                        module_path = os.path.join(module_dir, *node.module.split('.'))
                    else:
                        module_path = module_dir
                    potential_file = module_path + '.py'
                    if os.path.exists(potential_file) and self.is_project_file(potential_file):
                        self.track_file_access(potential_file)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Handle absolute imports
                    module_names = []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module_names.append(node.module)
                    else:
                        for name in node.names:
                            module_names.append(name.name.split('.')[0])
                    
                    for module_name in module_names:
                        # Look for local module files in project directory
                        module_path = os.path.join(self.project_root, *module_name.split('.')) + '.py'
                        if os.path.exists(module_path) and self.is_project_file(module_path):
                            self.track_file_access(module_path)
                            
        except Exception as e:
            print(f"Warning: Could not analyze imports in {filepath}: {str(e)}")

    def create_zip(self, filepaths):
        """
        Process files and create a single zip with all local dependencies.
        
        Args:
            filepaths (list): List of file paths to process.
            
        Returns:
            tuple: A tuple containing the hash ID (str) and the path to the saved .zip file (str).
        """
        # Process all files and their dependencies
        for filepath in filepaths:
            abs_path = os.path.abspath(filepath)
            if not self.is_project_file(abs_path):
                print(f"Warning: Skipping {filepath} as it's outside the project root")
                continue
                
            self.track_file_access(abs_path)
            
            try:
                with open(abs_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                self.find_config_files(content, abs_path)
                
                if filepath.endswith('.py'):
                    self.analyze_python_imports(abs_path)
            except Exception as e:
                print(f"Warning: Could not process {filepath}: {str(e)}")
        
        if not self.tracked_files:
            raise ValueError("No project files were found to include in the zip")

        # Generate hash from all files
        hash_contents = []
        for filepath in sorted(self.tracked_files):
            try:
                with open(filepath, 'rb') as file:
                    content = file.read()
                    hash_contents.append(content)
            except Exception as e:
                print(f"Warning: Could not read {filepath} for hash calculation: {str(e)}")
                
        combined_content = b''.join(hash_contents)
        hash_id = hashlib.sha256(combined_content).hexdigest()

        # Create zip file
        zip_filename = os.path.join(self.output_dir, f'{hash_id}.zip')
        base_path = os.path.commonpath([os.path.abspath(p) for p in self.tracked_files])
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in sorted(self.tracked_files):
                try:
                    relative_path = os.path.relpath(filepath, base_path)
                    zipf.write(filepath, relative_path)
                    print(f"Added to zip: {relative_path}")
                except Exception as e:
                    print(f"Warning: Could not add {filepath} to zip: {str(e)}")

        return hash_id, zip_filename

def zip_list_of_unique_files(filepaths):
    """
    Enhanced version of the original function that tracks all dependencies.
    
    Args:
        filepaths (list): List of file paths to process.
        
    Returns:
        tuple: A tuple containing the hash ID (str) and the path to the saved .zip file (str).
    """
    tracker = TraceDependencyTracker()
    return tracker.create_zip(filepaths)

if __name__ == "__main__":
    filepaths = ["script1.py", "script2.py"]
    hash_id, zip_path = zip_list_of_unique_files(filepaths)
    print(f"Created zip file: {zip_path}")
    print(f"Hash ID: {hash_id}")