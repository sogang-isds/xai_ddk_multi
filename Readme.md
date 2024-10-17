# Project Setup and Inference Instructions

## Prerequisites

- Ensure `Docker`` is installed on your system.

## Steps to Run Inference

### 1. Build the Docker Image

Run the following command to build the Docker image:

```bash
./build_docker.sh
```

### 2. Run the Docker Container

Once the image is built, start the Docker container using:

```bash
./run_container.sh
```

Replace `<container_name>` with the name or ID of the running container. You can find the container name by running:

```bash
docker ps
```

### 3. Execute the Inference Script

Once inside the container, run the inference script with:

```bash
python3 inference.py
```

Ensure that the required input arguments are provided as per the script's requirements.

## Notes

- Make sure all dependencies are installed and included in the Dockerfile.
- If you encounter issues with missing libraries, ensure you run system updates and install packages like `libgtk2.0-0` where necessary.

