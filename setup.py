from setuptools import setup, find_packages

setup(
    name='torchgym',
    version='0.1.0',
    description='A collection of Gym environments implemented with PyTorch.',
    author='Nathan Gavenski',
    author_email='nathangavenski@gmail.com',
    url='https://github.com/NathanGavenski/PyTorchGym',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'gymnasium>=0.29.1',
        'torchaudio>=0.9.0',
        'torchvision>=0.10.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9.15',
)
