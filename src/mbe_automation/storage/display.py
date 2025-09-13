import os
import h5py

def tree(dataset: str):
    """
    Visualize the tree structure of a dataset file.
    """
    if not os.path.exists(dataset):
        print(f"Error: File '{dataset}' not found.")
        return

    def print_recursive(name, obj, indent="", is_last=True):
        """Recursively print the contents of a group or dataset."""
        connector = "└── " if is_last else "├── "
        
        if isinstance(obj, h5py.Dataset):
            info = f"shape={obj.shape}, dtype={obj.dtype}"
            print(f"{indent}{connector}{os.path.basename(name)} [{info}]")
        else:
            print(f"{indent}{connector}{os.path.basename(name)}/")
            
            # Prepare indentation for children
            child_indent = indent + ("    " if is_last else "│   ")
            
            # Print attributes first
            if obj.attrs:
                attr_prefix = "└── " if not list(obj.keys()) else "├── "
                print(f"{child_indent}{attr_prefix}@attrs")
                attr_indent = child_indent + ("    " if not list(obj.keys()) else "│   ")
                
                attr_items = list(obj.attrs.items())
                for i, (key, value) in enumerate(attr_items):
                    is_last_attr = (i == len(attr_items) - 1)
                    attr_connector = "└── " if is_last_attr else "├── "
                    print(f"{attr_indent}{attr_connector}{key}: {value}")
            
            # Print groups and datasets
            items = list(obj.items())
            for i, (key, item) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                print_recursive(key, item, child_indent, is_last_item)

    try:
        with h5py.File(dataset, "r") as f:
            print(f"{os.path.basename(dataset)}")
            items = list(f.items())
            for i, (name, obj) in enumerate(items):
                is_last = (i == len(items) - 1)
                print_recursive(name, obj, "", is_last)
    except Exception as e:
        print(f"Error reading dataset file: {e}")


