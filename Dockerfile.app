# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Install system dependencies and Node.js (using NodeSource for Node 18)
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

# Copy files required for dependencies
COPY requirements.txt   requirements.txt
COPY frontend/ frontend/

# Install Python dependencies for the backend
RUN pip install --upgrade pip && \
pip install -r requirements.txt

# Install Node dependencies for the frontend
RUN cd frontend && npm install

# Copy the entire monorepo into the container
COPY . .

# Make the start script executable
RUN chmod +x start-prod.sh

# Expose the ports used by the frontend
EXPOSE 3000


# Start the processes using the start script
CMD ["bash", "start-dev.sh"]
