import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datapipeline",
    version="1.0",
    author="Aman",
    author_email="aman@instoried.com",
    description="Natural Language Toolkit for Indian Languages (iNLTK)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.google.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"
    ],
)
