# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
# Use --default-timeout=100 to prevent timeouts on slow networks
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application code, FAISS index, and documents into the container at /app
# Ensure your .dockerignore file excludes unnecessary files (like .env, .git, __pycache__, venv)
COPY . .

# Set environment variable for HuggingFace tokenizers (avoids warning)
ENV TOKENIZERS_PARALLELISM=false

# Make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

# Add a health check to verify Streamlit is running
# Streamlit's default health check endpoint
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Define the command to run your app using streamlit
# Use --server.port to match the EXPOSE directive
# Use --server.address=0.0.0.0 to make it accessible from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]