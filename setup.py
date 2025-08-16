import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List, Dict

# Define constants
PROJECT_NAME = "enhanced_cs.CL_2508.10860v1_From_Black_Box_to_Transparency_Enhancing_Automate"
VERSION = "1.0.0"
DESCRIPTION = "Enhanced AI project based on cs.CL_2508.10860v1_From-Black-Box-to-Transparency-Enhancing-Automate with content analysis"
AUTHOR = "Your Name"
EMAIL = "your@email.com"
URL = "https://github.com/your-username/your-repo-name"

# Define dependencies
DEPENDENCIES: List[str] = [
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "nltk",
    "spacy",
]

# Define development dependencies
DEV_DEPENDENCIES: List[str] = [
    "pytest",
    "flake8",
    "mypy",
    "black",
]

# Define test dependencies
TEST_DEPENDENCIES: List[str] = [
    "pytest",
    "pytest-cov",
]

# Define package data
PACKAGE_DATA: Dict[str, List[str]] = {
    "": ["*.txt", "*.md"],
}

# Define package directories
PACKAGE_DIRS: List[str] = [
    "src",
    "tests",
]

# Define entry points
ENTRY_POINTS: Dict[str, List[str]] = {
    "console_scripts": [
        "enhanced_cs=src.main:main",
    ],
}

# Define classifiers
CLASSIFIERS: List[str] = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

# Define long description
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Define setup function
def setup_package():
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        package_data=PACKAGE_DATA,
        install_requires=DEPENDENCIES,
        extras_require={
            "dev": DEV_DEPENDENCIES,
            "test": TEST_DEPENDENCIES,
        },
        entry_points=ENTRY_POINTS,
        classifiers=CLASSIFIERS,
        python_requires=">=3.8",
    )

# Define custom install command
class CustomInstallCommand(install):
    def run(self):
        install.run(self)

# Define custom develop command
class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)

# Define custom egg info command
class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)

# Define setup commands
setup_commands = {
    "install": CustomInstallCommand,
    "develop": CustomDevelopCommand,
    "egg_info": CustomEggInfoCommand,
}

# Run setup
if __name__ == "__main__":
    setup_package()