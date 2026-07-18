"""Check available packages."""
import importlib
for mod in ['pandas', 'sklearn', 'lightgbm', 'joblib', 'numpy']:
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"{mod}: NOT FOUND")
