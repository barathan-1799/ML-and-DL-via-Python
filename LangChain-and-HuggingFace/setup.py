## setup.py

# Importing dependencies
import subprocess
import sys

def install_packages(packages):
    """
    Install a list of packages via pip into the current Python environment.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

def main():
    # 1) Install the required libraries. Note: `mailbox` is part of the Python stdlib, no need to pip install it

    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "datasets",
        "evaluate",
        "langchain",
        "langchain-community",
        "chromadb",
        "sentence-transformers",
        "fastapi",
        "uvicorn",
        "spacy"
    ]
    print("Installing Python packages…")
    install_packages(packages)

    # 2) Download spaCy’s English model
    print("Downloading spaCy English model…")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    print("All done!")

if __name__ == "__main__":
    main()
