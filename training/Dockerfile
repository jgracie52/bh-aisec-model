# Start from a truly empty base image.
# This image will contain only the files you explicitly add.
#FROM scratch

FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the entire application code into the container at /app
# Assuming your Flask app file is named app.py and is in the root of your project
COPY mnist_classifier.pth /app

# Copy the pre-trained model file into the root of the container.
# IMPORTANT: Ensure 'mnist_classifier.pth' exists in the same directory as this Dockerfile
# when you build the image.
#COPY mnist_classifier.pth /

# You cannot run commands like 'ls' or 'cat' in a scratch image
# unless you explicitly copy the binaries for those commands.
# This CMD simply indicates that the image is built and contains the file.
# For practical use, you would typically use this image in a multi-stage build
# or copy this model into another image that has a Python runtime for inference.
CMD ["/bin/true"]
