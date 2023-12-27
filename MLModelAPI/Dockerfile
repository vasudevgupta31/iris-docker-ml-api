# Start from the official Python base image or some other base image
# FROM python:3.9
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


# Set the current working directory to /code.
WORKDIR /code


# Copy the file with the requirements to the /code directory. Copy only the file with the requirements first, not the rest of the code.
# As this file doesn't change often, Docker will detect it and use the cache for this step, enabling the cache for the next step too.
COPY ./requirements.txt /code/requirements.txt


# Install the package dependencies in the requirements file.
RUN pip install --no-cache-dir -r /code/requirements.txt


# Copy the ./app directory inside the /code directory.
COPY ./app /code/app


# Set the command to run the uvicorn server.
# CMD takes a list of strings, each of these strings is what you would type in the command line separated by spaces.
# This command will be run from the current working directory, the same /code directory you set above with WORKDIR /code.
# Because the program will be started at /code and inside of it is the directory ./app with your code, Uvicorn will be able to see and import app from app.main.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]



# needs to be built with > docker build -t <project_name> .
# needs to be run with > docker run -p 80:80 <project_name>
