from setuptools import setup, find_packages

setup(
    name='ML-in-C',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',  # or 'tensorflow' if you prefer
        # Add other dependencies
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts if needed
        ],
    },
)
