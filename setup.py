from setuptools import setup, find_packages

setup(
    name="twi_nlp",
    version="1.0.0",
    author="Teddy Boamah (Birdcore)",
    author_email="saintt442@gmail.com",
    description="A Twi NLP package for translation, POS tagging, tokenization, stemming, and lemmatization.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/birdcoreone/NLP-python",
    packages=find_packages(),
    include_package_data=True,  # Ensure data files are included
    package_data={
        "twi_nlp": ["data/*.csv"],  # Ensure `twi_words.csv` is included
    },
    install_requires=[
        "chardet",
        "nltk",
        "pandas",
        "regex",
        "urllib3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
