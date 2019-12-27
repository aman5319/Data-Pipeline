from pathlib import Path

def create_dir(path):
    if not path.exists():
        path.mkdir(parents=True,exist_ok=True)
    return path
