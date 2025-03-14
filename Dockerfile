# Use official Python 3.7.16 runtime as base image
FROM python:3.7.16-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies according to your setup script
RUN pip install --no-cache-dir basicsr
RUN pip install --no-cache-dir facexlib
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir realesrgan

# Copy application code
COPY . .

# Install the project in development mode
#RUN python setup.py develop

# Create directories for GFPGAN weights
RUN mkdir -p gfpgan/weights

# Expose port for Streamlit
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "gfpgan_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]