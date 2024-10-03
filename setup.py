# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import subprocess

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

cwd = Path(__file__).resolve().parent

package_name = "cra5"
version = "0.0.3.dev1"
git_hash = "unknown"


try:
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
    )
except (FileNotFoundError, subprocess.CalledProcessError):
    pass


def write_version_file():
    path = cwd / package_name / "version.py"
    with path.open("w") as f:
        f.write(f'__version__ = "{version}"\n')
        f.write(f'git_version = "{git_hash}"\n')


write_version_file()

def get_extensions():
    ext_dirs = cwd / package_name / "compressai/cpp_exts"
    print(ext_dirs)
    ext_modules = []

    # Add rANS module
    rans_lib_dir = cwd / "third_party/ryg_rans"
    rans_ext_dir = ext_dirs / "rans"

    extra_compile_args = ["-std=c++17"]
    if os.getenv("DEBUG_BUILD", None):
        extra_compile_args += ["-O0", "-g", "-UNDEBUG"]
    else:
        extra_compile_args += ["-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.ans",
            sources=[str(s) for s in rans_ext_dir.glob("*.cpp")],
            language="c++",
            include_dirs=[rans_lib_dir, rans_ext_dir],
            extra_compile_args=extra_compile_args,
        )
    )

    # Add ops
    ops_ext_dir = ext_dirs / "ops"
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}._CXX",
            sources=[str(s) for s in ops_ext_dir.glob("*.cpp")],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules

def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

TEST_REQUIRES = ["pytest", "pytest-cov", "plotly"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "flake8-bugbear",
    "flake8-comprehensions",
    "isort",
    "mypy",
]
POINTCLOUD_REQUIRES = [
    "pointops-yoda",
    "pyntcloud-yoda",  # Patched version of pyntcloud.
]

platforms = ['linux/Windows']
def get_extra_requirements():
    extras_require = {
        "test": TEST_REQUIRES,
        "dev": DEV_REQUIRES,
        "doc": ["sphinx", "sphinx-book-theme", "Jinja2<3.1"],
        "tutorials": ["jupyter", "ipywidgets"],
        "pointcloud": POINTCLOUD_REQUIRES,
    }
    extras_require["all"] = {req for reqs in extras_require.values() for req in reqs}
    return extras_require


setup(
    name=package_name,
    version=version,
    description="A large compression model for weather and climate data.",
    url="https://github.com/taohan10200/CRA5",
    py_modules=['cra5'],
    author="HAN Tao",
    author_email="hantao10200@gmail.com",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=parse_requirements('requirements.txt'),
    extras_require=get_extra_requirements(),
    license="BSD 3-Clause Clear License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=[], #get_extensions(),
    cmdclass={"build_ext": build_ext},
)