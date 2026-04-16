from .maze_manager import MazeEnvManager

try:
    from .webshop_manager import WebShopEnvManager
except ModuleNotFoundError:  # Optional dependency for WebShop-only environments.
    WebShopEnvManager = None

__all__ = ["MazeEnvManager", "WebShopEnvManager"]
