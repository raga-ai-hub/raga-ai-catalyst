import inspect
from functools import wraps

class TrackName:
    def __init__(self):
        self.files = set()  # To store unique filenames

    def trace_decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # frame = inspect.stack()[1]
            # file_name = frame.filename
            file_name = self._get_file_name()
            # print(f"Called from file: {frame.filename}")
            # print(f"Called from line: {frame.lineno}")
            # print(f"Called from function: {frame.function}")
            self.files.add(file_name)
            return func(*args, **kwargs)
        return wrapper

    def _get_file_name(self):
        # Check if running in a Jupyter notebook
        try:
            from IPython import get_ipython
            if 'IPKernelApp' in get_ipython().config:
                return self._get_notebook_name()
        except Exception:
            pass

        # Default to the filename from the stack
        frame = inspect.stack()[2]
        return frame.filename

    def _get_notebook_name(self):
        # Attempt to get the notebook name
        try:
            import ipynbname
            return ipynbname.name()  # This will return the notebook name
        except ImportError:
            return "Notebook name retrieval requires ipynbname package"
        except Exception as e:
            return f"Error retrieving notebook name: {e}"

        
    def get_unique_files(self):
        return list(self.files)