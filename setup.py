from setuptools import setup

setup(
	name='ipfs_huggingface_scraper_py',
	version='0.0.2',
	packages=[
		'ipfs_huggingface_scraper_py    ',
	],
	install_requires=[
        'ipfs_datasets_py',
        'ipfs_transfrormers_py',
        "ipfs_model_manager_py",
        "orbitdb_kit_py",
		"ipfs_kit_py",
		"faiss",
        "ipfs_faiss_py",
		'datasets',
		'urllib3',
		'requests',
		'boto3',
	]
)