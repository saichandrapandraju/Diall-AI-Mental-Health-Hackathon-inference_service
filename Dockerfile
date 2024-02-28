FROM ubuntu:latest

RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get install -y python3.10 python3-pip

# # Use the official Python image
# FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install deps
RUN pip3 install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]