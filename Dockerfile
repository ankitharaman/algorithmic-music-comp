# Use the official TensorFlow base image
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
# COPY . /app

RUN pip install scikit-learn music21

# (Optional) You can specify a command to run your app here, for example:
CMD ["bash"]
