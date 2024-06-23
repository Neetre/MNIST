#!/bin/bash

source ./setup/setup_Linux.sh

echo "Want to run the GUI? (y/n):"
read UserInput

if [ $UserInput = "n" ]; then
    exit
fi

# Change the current directory to the one containing the script
cd /bin/

# Run the application
python3 mnist_gui.py
