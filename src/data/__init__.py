# src/data/__init__.py

# Initialize the data package
from .mnist_loader import load_mnist
# from .cifar10_loader import load_cifar10
# from .cifar100_loader import load_cifar100
# from .imdb_loader import load_imdb
# from .agnews_loader import load_agnews
# from .adult_loader import load_adult
# from .winequality_loader import load_winequality

__all__ = ['load_mnist']

# __all__ = ['load_mnist', 'load_cifar10', 'load_cifar100', 'load_imdb', 'load_agnews', 'load_adult', 'load_winequality']
