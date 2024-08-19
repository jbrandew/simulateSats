FROM python:3.10.12

# Set the working directory
WORKDIR /app

# Copy the local code to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir \
    pyyaml \
    numpy \
    matplotlib

# Command to run your application
CMD ["python", "your_script.py"]
