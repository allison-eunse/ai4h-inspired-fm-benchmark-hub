import importlib
from typing import Any, Dict, Optional

def load_model_from_config(config: Dict[str, Any]) -> Any:
    """
    Load and instantiate a model class based on a configuration dictionary.
    
    Expects config to have:
    - type: "python_class" (currently only supported type)
    - import_path: "module.submodule:ClassName"
    - init_kwargs: dict of arguments to pass to constructor
    """
    model_type = config.get("type")
    if model_type != "python_class":
        raise ValueError(f"Unsupported model type: {model_type}. Only 'python_class' is supported.")
        
    import_path = config.get("import_path")
    if not import_path or ":" not in import_path:
        raise ValueError(f"Invalid import_path: {import_path}. Must be in format 'module:ClassName'")
        
    module_path, class_name = import_path.split(":")
    
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load class {class_name} from {module_path}: {e}")
        
    init_kwargs = config.get("init_kwargs", {})
    try:
        model = cls(**init_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate {class_name} with kwargs {init_kwargs}: {e}")
        
    return model

