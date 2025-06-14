from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="workforce_analysis_app",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered workforce analysis application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/workforce_analysis_app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Human Resources",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "workforce-analysis=app:main",
        ],
    },
) 