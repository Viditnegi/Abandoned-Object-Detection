# Use the official Python image from the Docker Hub
FROM python:3.7

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the specified command within the container
CMD ["python", "src/main.py"]
