# Base image with PyTorch and HuggingFace support
FROM huggingface/transformers-pytorch-gpu:latest

# Set the working directory inside the container
WORKDIR /mnt/code/multi_input_model

# Update the package list and install any needed packages
RUN apt-get update -y && apt-get install -y libgtk2.0-0

RUN pip install --upgrade pip

# Copy the local requirements.txt to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .