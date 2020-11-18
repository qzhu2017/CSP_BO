from distutils.core import setup
import setuptools  # noqa
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

exec(open("cspbo/version.py").read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cspbo",
    version=__version__,
    description="Python code for prediction of crystal structures based on symmetry constraints.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "cspbo",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'cffi>=1.0.0',
        "pyxtal>=0.1.2",
        "pyxtal_ff>=0.1.2",
    ],

    # Check the differece
    setup_requires = [
    'cffi>=1.0.0',
    ],

    python_requires=">=3.6.1",
    license="MIT",
)
