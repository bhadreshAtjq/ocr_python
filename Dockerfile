# Use a slim Python image for a smaller footprint
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /app

# Install system utilities needed for image processing and high-performance libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Set environment variables (Defaults - should be overridden in hosting settings)
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run the application using uvicorn
CMD ["uvicorn", "production_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
