import sys, os

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

hashgrid_cmake_toolchain_file = os.environ.get("HASHGRID_CMAKE_TOOLCHAIN_FILE", "")
hashgrid_drjit_cmake_dir = os.environ.get("HASHGRID_DRJIT_CMAKE_DIR", "")

VERSION = "0.0.1"

setup(
    name="hashgrid",
    version=VERSION,
    description="Hash Grid Implementation",
    author="Christian Döring",
    author_email="christian.doering@tum.de",
    license="BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    install_requires=["drjit"],
    package_dir={"": "src"},
    cmake_args=[
        f"-DCMAKE_TOOLCHAIN_FILE={hashgrid_cmake_toolchain_file}",
        f"-DDRJIT_CMAKE_DIR:STRING={hashgrid_drjit_cmake_dir}",
        f"-DPROJECT_VERSION_INFO={VERSION}",
    ],
    cmake_install_dir="src/hashgrid",
    include_package_data=True,
    python_requires=">=3.8",
)
