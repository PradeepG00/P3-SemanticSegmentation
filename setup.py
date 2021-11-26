# from setuptools import setup

# setup.py
# from setuptools import setup, find_packages
import setuptools


setuptools.setup(
    # defined as the cli package (Note the naming)
    name='mscgnet',
    version='0.1.0',
    author="",
    author_email="",
    url="",
    # functions as a search for the name of the package defined in the name variable
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points='''
    [console_scripts]
    mscgnet=cli.main:cli
    ''',
    python_requires=">=3.6"
)