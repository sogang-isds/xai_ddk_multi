# Project Setup and Inference Instructions

## Prerequisites

- Ensure `Docker` is installed on your system.

## Steps to Run Inference

### 0. Download Checkpoint Files

Before proceeding, download the required checkpoint (`.ckpt`) files from the following Google Drive link:

**[Download Checkpoint Files](<Insert_Google_Drive_Link_Here>)**

Once downloaded, place the checkpoint files in the following directory within the project:

```

/path/to/your/project/checkpoints/

```

Ensure that the checkpoint files are correctly placed here, as the inference script will expect them to be available in this directory.

### 1. Build the Docker Image

Run the following command to build the Docker image:

```bash
./build_docker.sh
````

### 2\. Run the Docker Container

Once the image is built, start the Docker container using:

```bash
./run_container.sh
```

> [!NOTE] 
> Replace `<container_name>` with the name or ID of the running container. You can find the container name by running:
> 
> ```bash
> docker ps
> ```

### 3\. Execute the Inference Script

Once inside the container, run the inference script with:

```bash
python3 inference.py
```

Ensure that the required input arguments are provided as per the script's requirements.

> [!NOTE] 
> 
> *   Make sure all dependencies are installed and included in the Dockerfile.
> *   If you encounter issues with missing libraries, ensure you run system updates and install packages like `libgtk2.0-0` where necessary.
> 