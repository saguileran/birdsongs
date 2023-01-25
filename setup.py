import os
#from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages, Command
from importlib.machinery import SourceFileLoader

version = SourceFileLoader('birdsongs.version',
                           'birdsongs/version.py').load_module()

with open('README.md', 'r', encoding="utf8") as fdesc:
    long_description = fdesc.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root.
    Deletes directories ./build, ./dist and ./*.egg-info
    From the terminal type:
        > python setup.py clean
    """
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.egg-info')

#if __name__ == '__main__':
#packages=['birdsongs'],

setup(
        name='birdsongs',
        version = version.__version__,
        packages = find_packages(),
        author = 'Sebastian Aguilera Novoa',
        maintainer = 'Sebastian Aguilera Novoa',
        author_email = 'saguileran@unal.edu.co',
        url = 'https://github.com/saguileran/birdsongs',
        license = 'GPL-3.0',
        keywords = ['numerical-optimization', 'bioacustics', 'syrinx', 'signal processing'],
        description = 'A python package for analyzing, visualizing and generating synthetic bird songs from recorded audio.',
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        license_file = 'LICENSE',
        cmdclass= {'clean': CleanCommand},
        python_requires = '>=3.5',
        install_requires = [  
                            "librosa>0.9",
                            "lmfit",
                            "scipy",
                            "sympy",
                            "numpy<1.24",
                            "pandas>1.5",
                            "matplotlib",
                            "playsound",
                            "PeakUtils",
                            "mpl_pan_zoom",
                            "mpl_point_clicker",
                            "scikit_learn",
                            "scikit_maad",
                            "setuptools",
                            "ipython",
                            "ffmpeg"
                            ]
     )

# Allow editable install into user site directory.
# See https://github.com/pypa/pip/issues/7953.
#site.ENABLE_USER_SITE = "--user" in sys.argv[1:]