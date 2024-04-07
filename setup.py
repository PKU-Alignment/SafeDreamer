import pathlib
import setuptools
from setuptools import find_namespace_packages


setuptools.setup(
    name='paper7020',
    version='0.0.0',
    description='Paper 7020',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(exclude=['example.py']),
    include_package_data=True,
    install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
)
