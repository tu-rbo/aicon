import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setuptools.setup(
    name="aicon",
    version="1.0.0",
    author="Vito Mengers",
    author_email="v.mengers@tu-berlin.de",
    description="Differentiable Online Multimodal Interactive Perception",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.tu-berlin.de/rbo/robotics/aicon",
    project_urls={
        "Bug Tracker": "https://git.tu-berlin.de/rbo/robotics/aicon/-/issues",
        "Documentation": "https://git.tu-berlin.de/rbo/robotics/aicon/-/wikis/home",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=1.0",
            "flake8>=6.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aicon=aicon.cli:main",
        ],
    },
)
