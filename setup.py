# setup.py
# AGPL v3 / VikaasLoop

import sys

from setuptools import find_packages, setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post installation script to check for CUDA availability."""

    def run(self):
        install.run(self)
        try:
            import torch

            if torch.cuda.is_available():
                print("\n" + "=" * 50)
                print(
                    "✅ CUDA is available. VikaasLoop Training Agent is fully supported!"
                )
                print("=" * 50 + "\n")
            else:
                self._print_cuda_warning()
        except ImportError:
            self._print_cuda_warning()

    def _print_cuda_warning(self):
        print("\n" + "!" * 60)
        print(
            "⚠️  WARNING: CUDA is not available or PyTorch is not compiled with CUDA support."
        )
        print("The VikaasLoop Training Agent highly recommends an NVIDIA GPU.")
        print("To train models locally, please ensure:")
        print("  1. You have an NVIDIA GPU.")
        print("  2. The correct CUDA toolkit is installed.")
        print(
            "  3. Install CUDA enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )
        print("!" * 60 + "\n")


setup(
    name="vikaasloop",
    version="1.0.0",
    description="Autonomous Self-Improving LLM Fine-Tuning Engine",
    author="Akshay Sharma",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "aiofiles>=23.2.1",
        "google-genai>=0.5.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "trl>=0.8.6",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.29.0",
        "datasets>=2.19.0",
        "torch>=2.2.0",
        "python-dotenv>=1.0.0",
        "pydantic-settings>=2.2.1",
        "sentence-transformers>=2.5.1",
        "numpy>=1.26.0",
        "tenacity>=8.2.0",
        "PyJWT>=2.8.0",
        "huggingface-hub>=0.22.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "black>=24.2.0", "isort>=5.13.2", "flake8>=7.0.0"],
    },
    cmdclass={
        "install": PostInstallCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
