# MNIST

## Description

This repository contains a project for Hack Club.
The program defines the structure of a Neural Network, in this case a CNN model, look [here](https://www.3blue1brown.com/lessons/neural-networks) for reference. The NN is built to recognize handwritten digits, it is trained using the MNIST dataset.

## Requirements

* Python >= 3.8

**Setting Up the Environment**

1. Activate your virtual environment (if you're using one).
2. Run the appropriate setup script for your operating system:

   * Windows: `./setup.bat`
   * Linux/macOS: `./setup.sh`

These scripts likely install required dependencies, and build a virtual environment for you if you don't have one.

## Running the Program

1. Navigate to the `bin` directory: `cd bin`

2. Choose your method:

   * **GUI:** `python gradio.py [--help]` (use `python3` on Linux/macOS)
   * **Model:** `python MNIST.py [--help]` (use `python3` on Linux/macOS)

   The `--help` flag displays available command-line arguments.

## Author

Neetre
