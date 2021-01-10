from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as rm:
    long_desc = rm.read()

setup(
    name='LassyExtraction',
    version='0.3.dev1',
    author='Konstantinos Kogkalidis',
    author_email='k.kogkalidis@uu.nl',
    description='Tool suite for the representation of Modal Intuitionistic Linear Logic theorems and their extraction '
                'from Alpino-based dependency treebanks.',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/konstantinosKokos/Lassy-TLG-Extraction/tree/0.3.dev1',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8"
)