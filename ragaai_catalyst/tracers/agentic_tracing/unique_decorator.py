import hashlib
import inspect
import functools
import re
import tokenize
import io
import uuid

def normalize_source_code(source):
    """
    Advanced normalization of source code that:
    1. Preserves docstrings
    2. Removes comments
    3. Removes extra whitespace
    
    Args:
        source (str): Original source code
    
    Returns:
        str: Normalized source code
    """
    # Use tokenize to carefully parse the source code
    normalized_tokens = []
    
    try:
        # Convert source to a file-like object for tokenize
        token_source = io.StringIO(source).readline
        
        for token_type, token_string, _, _, _ in tokenize.generate_tokens(token_source):
            # Preserve strings (including docstrings)
            if token_type == tokenize.STRING:
                normalized_tokens.append(token_string.strip())
            
            # Preserve code tokens
            elif token_type in [
                tokenize.NAME, 
                tokenize.NUMBER, 
                tokenize.OP
            ]:
                normalized_tokens.append(token_string.strip())
    
    except tokenize.TokenError:
        # Fallback to a simpler method if tokenization fails
        normalized_tokens = re.findall(r'\w+|[^\w\s]', source)
    
    # Remove extra spaces and join
    normalized_source = ''.join(normalized_tokens)
    
    return normalized_source

# def generate_unique_hash(obj, *call_args, **call_kwargs):
#     print('#'*100,'hash id: ', '#'*100)
#     print(obj)
#     print(*call_args)
#     # print(**call_kwargs)
#     """
#     Generate a unique, deterministic hash for a given object.
    
#     Args:
#         obj: The object (function or class) to generate hash for
#         additional_salt: Optional additional salt to ensure uniqueness
    
#     Returns:
#         str: A unique hash_id meeting the specified requirements
#     """
#     # Handle different object types

#     if inspect.isclass(obj):
#         # For classes, use the class definition
#         try:
#             source = inspect.getsource(obj)
#         except (IOError, TypeError):
#             source = repr(obj)
        
#         # Use class name in hash generation
#         hash_input = f"{obj.__name__}{normalize_source_code(source)}"
    
#     else:
#         # For functions and methods
#         # Get full signature information
#         signature = inspect.signature(obj)
        
#         # Capture parameter names and their default values
#         params_info = []
#         for name, param in signature.parameters.items():
#             param_str = f"{name}:{param.kind}"
#             if param.default != inspect.Parameter.empty:
#                 param_str += f":default={param.default}"
#             params_info.append(param_str)
        
#         # Get source code
#         try:
#             source = inspect.getsource(obj)
#         except (IOError, TypeError):
#             source = repr(obj)
        
#         # Combine method name, parameters, and normalized source
#         hash_input = (
#             f"{obj.__name__}"  # Method name
#             f"{''.join(params_info)}"  # Parameter details
#             f"{normalize_source_code(source)}"  # Normalized source code
#         )
    
#     # Add optional salt
#         args_repr = str(call_args) + str(sorted(call_kwargs.items()))
#         hash_input += args_repr    
#     # Use SHA-256 for generating the hash
#     hash_object = hashlib.sha256(hash_input.encode('utf-8'))
    
#     # Generate hash and truncate to 32 characters
#     hash_id = hash_object.hexdigest()[:32]
    
#     # Ensure the hash starts with a letter
#     if not hash_id[0].isalpha():
#         hash_id = 'a' + hash_id[1:]
    
#     print(hash_id)
#     return hash_id



def generate_unique_hash(obj, *args, **kwargs):
    """Generate a unique hash based on the normalized function definition and its arguments"""
    if inspect.ismethod(obj) or inspect.isfunction(obj):
        # Get function name and source code
        func_name = obj.__name__
        try:
            # Get the source code and normalize it
            func_source = inspect.getsource(obj)
            normalized_source = normalize_source_code(func_source)
        except (IOError, TypeError):
            normalized_source = ""
        
        # Get function arguments
        if args and hasattr(args[0], '__class__'):
            # If it's a method, skip the 'self' argument
            args = args[1:]
        
        # Normalize argument values
        def normalize_arg(arg):
            if isinstance(arg, (str, int, float, bool)):
                return str(arg)
            elif isinstance(arg, (list, tuple, set)):
                return '_'.join(normalize_arg(x) for x in arg)
            elif isinstance(arg, dict):
                return '_'.join(f"{normalize_arg(k)}:{normalize_arg(v)}" 
                              for k, v in sorted(arg.items()))
            elif callable(arg):
                return arg.__name__
            else:
                return str(type(arg).__name__)

        # Create normalized strings of arguments
        args_str = '_'.join(normalize_arg(arg) for arg in args)
        kwargs_str = '_'.join(f"{k}:{normalize_arg(v)}" 
                            for k, v in sorted(kwargs.items()))
        
        # Combine all components
        hash_input = f"{func_name}_{normalized_source}_{args_str}_{kwargs_str}"
    
    elif inspect.isclass(obj):
        # For classes, normalize the class definition
        try:
            class_source = inspect.getsource(obj)
            normalized_source = normalize_source_code(class_source)
            hash_input = f"{obj.__name__}_{normalized_source}"
        except (IOError, TypeError):
            hash_input = f"{obj.__name__}_{str(obj)}"
    
    else:
        # For other objects, use their string representation
        hash_input = str(obj)

    # Create hash
    hash_obj = hashlib.md5(hash_input.encode('utf-8'))
    return hash_obj.hexdigest()


class UniqueIdentifier:
    _instance = None
    _hash_cache = {}  # Class-level cache for storing hashes

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, salt=None):
        # Initialize only once
        if not hasattr(self, 'salt'):
            self.salt = salt

    def __call__(self, obj):
        if inspect.isclass(obj):
            hash_id = generate_unique_hash(obj)
            setattr(obj, 'hash_id', hash_id)
            return obj
        
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            # Generate cache key based on function and arguments
            cache_key = (obj.__name__, str(args), str(kwargs))
            
            # Use cached hash if available, otherwise generate new one
            if cache_key not in self._hash_cache:
                self._hash_cache[cache_key] = generate_unique_hash(obj, *args, **kwargs)
            
            # Store hash_id on the wrapper function
            wrapper.hash_id = self._hash_cache[cache_key]
            
            return obj(*args, **kwargs)
        
        # Initialize hash_id
        initial_hash = generate_unique_hash(obj)
        wrapper.hash_id = initial_hash
        
        return wrapper

# Create a single instance to be used across all mixins
mydecorator = UniqueIdentifier()

