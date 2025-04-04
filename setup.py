from setuptools import setup

setup(
	name='ipfs_huggingface_scraper_py',
	version='0.0.3',
	packages=[
		'ipfs_huggingface_scraper_py',
	],
	install_requires=[
        "orbitdb_kit_py",
		"ipfs_kit_py",
		'urllib3',
		'requests',
		'boto3',
	]
)