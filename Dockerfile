<<<<<<< HEAD
FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
=======
FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
>>>>>>> d7e2aaa135f49fc1c9c6994a84845e5182582fd0
RUN pip install -r requirements.txt