# Specify the base image. Use a lightweight Python image compatible with linux/amd64.
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python scripts into the container
COPY process_pdfs.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Define the entrypoint command for Adobe Challenge 1A
CMD ["python", "process_pdfs.py"]