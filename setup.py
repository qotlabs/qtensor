from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "scipy", "matplotlib", "torch", "tqdm"]

setup(
    name="qtensor",
    version="1.0",
    author="Sergei Kuzmin",
    author_email="ssk251198@yandex.ru",
    description="A software package designed to work with quantum states in the Tensor Trains format",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/SergeiKuzmin/qtensor.git",
    packages=find_packages(),
    install_requires=requirements,
)
