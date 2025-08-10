# setup.py
from setuptools import setup, find_packages

setup(
    name="attoscience_studio",
    version="1.0.0",
    description="Attoscience Studio: A handy simulation tool (mostly for raw data you get from Octopus.)",
    author="Erfan",
    license="GPLv3",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "PyQt5>=5.15.4",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "joblib>=1.0.0",
        "psutil>=5.8.0",
        "pyqtgraph>=0.12.0",
        "plotly>=5.0.0",
        "qtconsole>=5.0.0"
    ],
    entry_points={
        'gui_scripts': [
            'attoscience-studio=attoscience_studio.app:main'
        ]
    },
    python_requires='>=3.7',
)

