import os


def install_requirements():
    os.system("pip install -r requirements.txt")
    os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"
    os.system("pip install pykeen")
