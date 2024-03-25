from setuptools import setup, find_packages


setup(
    name="pycm",
    version="0.1",
    author="Georg Boecherer",
    author_email="mail@georg-boecherer.de",
    packages=find_packages(include="pycm"),
    url="https://github.com/gbsha/PAS",
    install_requires=['numpy <= 1.23'],
)
