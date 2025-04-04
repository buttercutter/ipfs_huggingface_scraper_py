from setuptools import setup, find_packages

setup(
    name='ipfs_huggingface_scraper_py',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ipfs_kit_py",
        "ipfs_datasets_py",
        "huggingface_hub>=0.16.0",
        "tqdm>=4.64.0",
        "pandas>=1.5.0",
        "pyarrow>=10.0.0",
        "tomli>=2.0.0",
        "tomli-w>=1.0.0",
        "urllib3",
        "requests",
        "boto3",
    ],
    entry_points={
        'console_scripts': [
            'hf-scraper=ipfs_huggingface_scraper_py.cli:main',
        ],
    },
    python_requires='>=3.7',
    description='A specialized module for scraping and processing model metadata from HuggingFace Hub',
    author='IPFS HuggingFace Scraper Team',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)