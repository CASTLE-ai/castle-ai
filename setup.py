from setuptools import setup, find_packages


def readme():
    try:
        with open("README.md", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Python 2.7 doesn't support encoding argument in builtin open
        import io

        with io.open("README.md", encoding="UTF-8") as readme_file:
            return readme_file.read()






def read_requirements():
    """Parse requirements.txt, ignoring comments and flags."""
    try:
        with open("requirements.txt", encoding="UTF-8") as req_file:
            requirements = []
            for line in req_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Skip pip flags like --extra-index-url
                if line.startswith("-"):
                    continue
                requirements.append(line)
            return requirements
    except IOError:
        return []


configuration = {
    "name": "castle-ai",
    "version": "0.0.18",
    "description": "Distinguish behavioral clusters Toolbox",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
    ],
    # "keywords": "dimension reduction t-sne manifold",
    "maintainer": "Raiso Liu, IsonaEi",
    "maintainer_email": "rainsoon717@gmail.com",
    "license": "AGPL-3.0 license",
    "packages": find_packages(),
    "install_requires": read_requirements(),

    "entry_points": {
        'console_scripts': [
            'castle=castle.cli.main:app',
        ],
    },
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)