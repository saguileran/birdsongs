from distutils.sysconfig import get_python_lib
from setuptools import setup


if __name__ == '__main__':
    setup(
    name='birdsongs',
    version='0.1.0',
    author='Sebastian Aguilera Novoa',
    author_email='saguileran@unal.edu.co',
    packages=['birdsongs'],
    url='https://github.com/saguileran/birdsongs',
    license='LICENSE.txt',
    description='A python package to analyze and visualize birdsongs',
    long_description=open('README.md').read(),
    install_requires=[
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
    ],
        )
    
    # Allow editable install into user site directory.
    # See https://github.com/pypa/pip/issues/7953.
    #site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

