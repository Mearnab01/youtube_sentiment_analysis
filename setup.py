from setuptools import find_packages, setup
setup(
    name = 'youtube_sentiment',
    version= '0.0.1',
    author= 'Arnab',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)