# Start from a truly empty base image.
# This image will contain only the files you explicitly add.
FROM scratch

# Copy the pre-trained model file into the root of the container.
# IMPORTANT: Ensure 'mnist_classifier.pth' exists in the same directory as this Dockerfile
# when you build the image.
COPY mnist_classifier.pth /

# You cannot run commands like 'ls' or 'cat' in a scratch image
# unless you explicitly copy the binaries for those commands.
# This CMD simply indicates that the image is built and contains the file.
# For practical use, you would typically use this image in a multi-stage build
# or copy this model into another image that has a Python runtime for inference.
CMD ["/bin/true"]
