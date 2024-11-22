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
    description="GPR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "cspbo",
        "cspbo.kernels",
    ],

    package_data={
        "cspbo.kernels": ["*.cpp"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'cffi>=1.0.0',
        'mpi4py>=3.0.3',
        "pyxtal>=1.0.4",
    ],
    python_requires=">=3.8.1",
    license="MIT",
    setup_requires=['cffi>=1.0.0'],
    cffi_modules=[
        "cspbo/kernels/libdot_builder.py:ffibuilder",
        "cspbo/kernels/librbf_builder.py:ffibuilder"
    ],
)
