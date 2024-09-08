from setuptools import setup, find_packages

setup(
    name="conjecture",
    version="0.2",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "conjecture=conjecture.entry:main",
        ],
    },
    # Optional metadata
    author="James Stevenson",
    author_email="opensource@jamesstevenson.me",
    description="Conjecture on if 'content' was present in a models training data",
    url="https://github.com/user1342/conjecture",
)
