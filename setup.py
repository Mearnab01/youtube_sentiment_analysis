from setuptools import find_packages, setup
setup(
    name = 'youtube_sentiment',
    version= '0.0.1',
    author= 'Arnab',
    author_email= 'your_email@example.com',
    # This is the secret sauce for your 'src' folder:
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)