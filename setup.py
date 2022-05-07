from setuptools import setup, find_packages

setup(
    name='byol-a-2',
    version='1.0.0',
    url='https://github.com/zqevans/byol-a-2.git',
    author='Zach Evans',
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "torchvision",
        "pytorch_lightning",
        "librosa",
        "fire",
        "easydict",
        "pandas",
        "tqdm"
    ],
)
