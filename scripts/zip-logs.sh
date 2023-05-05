#!/bin/bash

dir=$1

# Create a zip file with a timestamp in the filename
filename="logs.zip" # "folders_$(date +%Y-%m-%d_%H-%M-%S).zip"

find "$dir" -mindepth 1 -maxdepth 1 -type d -exec zip -r "$filename" {} \;

