import os
import json
import re


def check_last_file_name(directory, file, ext=".json") -> int:
    """Check the last file name in a directory.

    Args:
        directory (str): The directory to check.
        file (str): The pattern of the filename without the int at the end.
        ext (str, optional): The extension of the file. Defaults to ".json".

    Returns:
        int: The highest number found in the filenames.
    """
    highest_number = 0
    for filename in os.listdir(directory):
        match = re.search(file + r"(\d+)" + ext, filename)
        if match:
            number = int(match.group(1))
            if number > highest_number:
                highest_number = number
    return highest_number


def open_file(directory: str, file: str, ext: str, full: bool = False) -> list[str]:
    """Open a file and return the content as a list of strings.

    Args:
        directory (str): The directory of the file.
        file (str): The name of the file.
        ext (str): The extension of the file.

    Returns:
        list[str]: The content of the file as a list of strings.
    """
    out = []
    if ext == ".json":
        with open(directory + "/" + file + ext, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if full:
                    return data
                return list(data.keys())
            else:
                return []
    elif ext == ".txt":
        with open(directory + "/" + file + ext, "r") as f:
            for line in f:
                line = line.strip()
                words = line.split()
                out.append(words[1])
        return out
    elif ext == ".smi":
        with open(directory + "/" + file + ext, "r") as f:
            for line in f:
                line = line.strip()
                words = line.split()
                out.append(words[0])
        return out
    else:
        raise ValueError("Extension not supported")


def save_file(directory: str, filename: str, data, ext: str = ".json") -> None:
    """Save a file with the data as json.

    Args:
        directory (str): The directory of the file.
        filename (str): The name of the file.
        data (dict): The data to save.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    if ext not in [".json", ".txt", ".smi"]:
        raise ValueError("Extension not supported")

    if ext == ".txt":
        with open(directory + "/" + filename + ext, "w") as f:
            if isinstance(data, list):
                for item in data:
                    f.write(f"{filename} {item}\n")
            elif isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key} {value}\n")
            else:
                raise ValueError("Data must be a list or a dictionary")
        return

    if ext == ".smi":
        with open(directory + "/" + filename + ext, "w") as f:
            if isinstance(data, list):
                for item in data:
                    f.write(f"{item}\n")
            elif isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key} {value}\n")
            else:
                raise ValueError("Data must be a list or a dictionary")
        return

    # Default case for JSON
    if not filename.endswith(".json"):
        filename += ".json"

    # Save data as JSON
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if v is not None}
    else:
        raise ValueError("Data must be a dictionary for JSON format")
    with open(directory + "/" + filename + ".json", "w") as f:
        json.dump(data, f, indent=4)


def open_data(directory: str, filename: str) -> dict:
    """Open a data file and return the content as a dictionary.

    Args:
        directory (str): The directory of the file.
        filename (str): The name of the file.

    Returns:
        dict: The content of the file as a dictionary.
    """
    with open(directory + filename + ".json", "r") as f:
        data = json.load(f)
    return data


def ensure_nested_path(data_dict: dict, keys: list[str]) -> dict:
    """
    Ensures a nested path of dictionaries exists.
    If the path or parts of it do not exist, they are created.
    If the path already exists, it is not modified.

    Args:
        data_dict: The dictionary to modify.
        keys: A list of keys representing the path to ensure.

    Returns:
        The dictionary at the end of the specified path.
    """
    # Start at the root of the dictionary.
    current_level = data_dict

    # Iterate through every key in the provided path.
    for key in keys:
        current_level = current_level.setdefault(key, {})

    # Return the deepest dictionary in the path.
    return current_level
