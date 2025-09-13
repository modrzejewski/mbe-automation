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
        display_name = name

        if isinstance(obj, h5py.Dataset):
            info = f"shape={obj.shape}, dtype={obj.dtype}"
            print(f"{indent}{connector}{display_name} [{info}]")
        else:
            print(f"{indent}{connector}{display_name}/")
            
            child_indent = indent + ("    " if is_last else "│   ")
            
            items = list(obj.items())
            
            if obj.attrs:
                is_last_attr_block = not items
                attr_connector = "└── " if is_last_attr_block else "├── "
                print(f"{child_indent}{attr_connector}@attrs")
                
                attr_indent = child_indent + ("    " if is_last_attr_block else "│   ")
                attr_items = list(obj.attrs.items())
                for i, (key, value) in enumerate(attr_items):
                    is_last_attr = (i == len(attr_items) - 1)
                    connector_attr = "└── " if is_last_attr else "├── "
                    print(f"{attr_indent}{connector_attr}{key}: {value}")
            
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


