import yaml
import os
import glob
import sys

# Configuration
SOURCE_KB_PATH = "/Users/allison/Projects/neuro-omics-kb"
SOURCE_MODELS = os.path.join(SOURCE_KB_PATH, "kb/model_cards")
SOURCE_DATASETS = os.path.join(SOURCE_KB_PATH, "kb/datasets")

DEST_MODELS = "models"
DEST_DATASETS = "datasets"

def load_yaml(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"Saved {path}")

def import_models():
    print(f"Importing models from {SOURCE_MODELS}...")
    files = glob.glob(os.path.join(SOURCE_MODELS, "*.yaml"))
    
    for f in files:
        if "template.yaml" in f or "README.md" in f:
            continue
            
        source = load_yaml(f)
        if not source:
            continue
            
        # Map fields
        dest = {
            "model_id": source.get("id"),
            "name": source.get("name"),
            "modality": source.get("modality"),
            "upstream_repo": source.get("repo"),
            "notes": source.get("summary"),
            # Extra fields useful for linking
            "arch": source.get("arch", {}).get("type"),
            "params": source.get("arch", {}).get("parameters")
        }
        
        filename = os.path.basename(f)
        save_yaml(dest, os.path.join(DEST_MODELS, filename))

def import_datasets():
    print(f"Importing datasets from {SOURCE_DATASETS}...")
    files = glob.glob(os.path.join(SOURCE_DATASETS, "*.yaml"))
    
    for f in files:
        if "template.yaml" in f or "README.md" in f:
            continue
            
        source = load_yaml(f)
        if not source:
            continue
            
        # Map fields
        dest = {
            "dataset_id": source.get("id"),
            "name": source.get("name"),
            "source": "External/Restricted" if source.get("access") == "restricted" else "Public",
            "n_subjects": source.get("counts", {}).get("subjects", "Unknown"),
            "modalities": source.get("modalities", []),
            "notes": source.get("description")
        }
        
        filename = os.path.basename(f)
        save_yaml(dest, os.path.join(DEST_DATASETS, filename))

if __name__ == "__main__":
    if not os.path.exists(SOURCE_KB_PATH):
        print(f"Error: Source path {SOURCE_KB_PATH} does not exist.")
        sys.exit(1)
        
    os.makedirs(DEST_MODELS, exist_ok=True)
    os.makedirs(DEST_DATASETS, exist_ok=True)
    
    import_models()
    import_datasets()

