from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize


PACKAGE_NAME = "asset_asrl"
BASE_DIR = Path(__file__).parent


def find_pyx_modules(base_dir):
    extensions = []

    for file in base_dir.rglob("*.pyx"):

        rel_path = file.relative_to(BASE_DIR).with_suffix("")

        module = ".".join(rel_path.parts)

        extensions.append(
            Extension(
                module,
                [str(file)],
                language="c++"
            )
        )

    return extensions


ext_modules = cythonize(
    find_pyx_modules(BASE_DIR / PACKAGE_NAME),
    compiler_directives={
        "language_level": "3",
    }
)


setup(
    name="asset_asrl",
    version="0.0.1",
    packages=["asset_asrl"],

    ext_modules=ext_modules,
)