import importlib
from typing import Any, Dict, Optional


def load_model_from_config(config: Dict[str, Any]) -> Any:
    """
    Load and instantiate a model based on a configuration dictionary.
    
    Supported types:
    - "python_class": Direct Python class import
    - "adapter": Use model adapter from fmbench.model_adapters
    
    For python_class:
    - import_path: "module.submodule:ClassName"
    - init_kwargs: dict of arguments to pass to constructor
    
    For adapter:
    - adapter_name: name of adapter in ADAPTER_REGISTRY
    - (other kwargs passed to adapter)
    """
    model_type = config.get("type")
    
    if model_type == "adapter":
        return _load_adapter(config)
    elif model_type == "python_class":
        return _load_python_class(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'python_class' or 'adapter'.")


def _load_adapter(config: Dict[str, Any]) -> Any:
    """Load a model using the adapter system."""
    from .model_adapters import get_adapter
    
    adapter_name = config.get("adapter_name")
    if not adapter_name:
        raise ValueError("adapter config must contain 'adapter_name'")
    
    # Extract adapter kwargs (everything except type, adapter_name, model_id)
    adapter_kwargs = {
        k: v for k, v in config.items() 
        if k not in ("type", "adapter_name", "model_id")
    }
    
    adapter = get_adapter(adapter_name, **adapter_kwargs)
    adapter.load()
    return adapter


def _load_python_class(config: Dict[str, Any]) -> Any:
    """Load a model via direct Python class import."""
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
