# Simple App using UltraFace

This is a simple app using UltraFace. It is a face detector that can detect faces in 640x480 px images.

# Setup

## Prerequisites

This repo uses `.devcontainer` to setup the environment. You can use VSCode to open this repo and it will automatically setup the environment for you. Make sure you have `Docker` installed and the `devcontainer` extension installed.

## Reproduce on your machine

Click on "reopen in container" in the bottom right corner of VSCode. This will open a new VSCode window with the environment setup for you.

After the environment is setup, make sure to install all dependencies by running `pip install -r requirements.txt`.

## Run the app locally

Open the terminal in VSCode and run: 

'''bash
flask run
'''

This will start the app on `localhost:5000`.

To close the app, press `Ctrl+C` in the terminal.
