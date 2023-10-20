from setuptools import find_packages, setup

setup(
    name='ssmnet',
    version='0.1.0',
    description='Music boundary estimation using SSM-Net',
    author='Geoffroy Peeters',
    url='https://github.com/geoffroypeeters/ssmnet_ISMIR2023',
    license='LGPL-3.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'ssmnet': ['weights_deploy/*'],  # To include the .pth
    },
    install_requires=[
        'librosa==0.10.1',
        'numpy==1.24.3',
        'pyyaml==6.0',
        'scikit-learn==1.3.1',
        'scipy==1.10.1',
        'torch==2.1.0'
    ],
    entry_points={
        'console_scripts': [
            'ssmnet=ssmnet.example:ssmnet_main',  # For the command line, executes function pesto() in pesto/main as 'pesto'
        ],
    }
)