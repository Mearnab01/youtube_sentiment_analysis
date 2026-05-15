import os


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/data/__init__.py",
    f"src/data/data_ingestion.py",
    f"src/data/data_preprocessing.py",
    f"src/model/__init__.py",
    f"src/model/model_building.py",
    f"src/model/model_evaluation.py",
    f"src/model/register_model.py",
    f"src/utils/__init__.py",
    f"src/utils/logger.py",
    "params.yaml",
    "dvc.yaml",
    "requirements.txt",
    "setup.py",
    "app.py",
    ".gitignore",
    ".dvcignore"
]

for filepath in list_of_files:
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Created directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # Create empty file
        print(f"📄 Created empty file: {filepath}")

print("\n✅ Your MLOps structure is ready ")