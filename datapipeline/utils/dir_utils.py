from pathlib import Path

def create_dir(path):
    """Creates dir if not exists."""
    if not path.exists():
        path.mkdir(parents=True,exist_ok=True)
    return path
