import setuptools
import ussegmentation

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ussegmentation",  # Replace with your own username
    version=ussegmentation.__VERSION__,
    author="Ivan Belyavtsev",
    author_email="djbelyak@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djbelyak/ussegmentation",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": ["ussegmentation = ussegmentation.__main__:main"]
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
