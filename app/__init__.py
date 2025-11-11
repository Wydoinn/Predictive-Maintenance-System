"""
Application package initialization.
"""

__version__ = '1.0.0'
__author__ = 'wydoinn'

# Import main modules for easier access
from . import app
from . import dashboard

__all__ = [
    'app',
    'dashboard',
]
