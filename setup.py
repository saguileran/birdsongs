from distutils.sysconfig import get_python_lib
from setuptools import setup


if __name__ == '__main__':
    setup()
    
    # Allow editable install into user site directory.
    # See https://github.com/pypa/pip/issues/7953.
    #site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

