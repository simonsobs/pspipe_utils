from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="pspipe_utils",
    version="0.0.1",
    description="SO pipeline utilities",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "pixell>=0.16.0",
        "mflike @ git+https://github.com/simonsobs/LAT_MFLike@master#egg=mflike",
    ],
)
