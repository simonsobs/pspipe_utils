from setuptools import find_packages, setup

import versioneer

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="pspipe_utils",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="SO pipeline utilities",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "pspy>=1.5.3",
        "mflike @ git+https://github.com/simonsobs/LAT_MFLike@master#egg=mflike",
    ],
    package_data={"": ["data/**"]},
)
