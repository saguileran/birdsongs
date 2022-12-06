"""Version info"""

import sys, importlib

short_version = "0.1"
version = "0.1.beta"


def __get_mod_version(modname):

    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
        try:
            return mod.__version__
        except AttributeError:
            return "installed, no version number available"

    except ImportError:
        return None


def show_versions():
    """Return the version information for all librosa dependencies."""

    core_deps = [
        "librosa",
        "lmfit",
        "scipy",
        "sympy",
        "numpy",
        "pandas",
        "matplotlib",
        "playsound",
        "PeakUtils",
        "mpl_pan_zoom",
        "mpl_point_clicker",
        "scikit_learn",
        "scikit_maad",
        "setuptools",
        "ipython",
        "lazy_loader",
    ]


    print("INSTALLED VERSIONS")
    print("------------------")
    print("python: {}\n".format(sys.version))
    print("librosa: {}\n".format(version))
    [print("{}: {}".format(dep, __get_mod_version(dep))) for dep in core_deps];