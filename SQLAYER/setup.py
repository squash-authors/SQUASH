from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="sqlayer",
    version="0.0.10",
    description="Python Layer Code for SQUASH system",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XX/XX",
    author="XX XX",
    author_email="XX@XX.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy>=2.0.0", "boto3>=1.34.82", "bitarray==2.5.1"],
    extras_require={},
    python_requires=">=3.11",
)
