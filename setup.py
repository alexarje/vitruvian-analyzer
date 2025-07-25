"""
Setup script for Vitruvian Proportion Analyzer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vitruvian-proportion-analyzer",
    version="1.0.0",
    author="Vitruvian Proportion Analyzer Project",
    description="A Python tool that analyzes human body proportions against classical Vitruvian ideals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/vitruvian-proportion-analyzer",
    py_modules=["vitruvius_measurement", "batch_process"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vitruvian-analyzer=vitruvius_measurement:main",
            "vitruvian-batch=batch_process:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["pose.png", "README.md", "LICENSE"],
    },
    keywords="computer-vision pose-estimation body-proportions vitruvian mediapipe",
    project_urls={
        "Bug Reports": "https://github.com/your-username/vitruvian-proportion-analyzer/issues",
        "Source": "https://github.com/your-username/vitruvian-proportion-analyzer",
        "Documentation": "https://github.com/your-username/vitruvian-proportion-analyzer#readme",
    },
)
