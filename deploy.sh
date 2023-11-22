#!/bin/sh

sudo docker build -t freebus-docker .
sudo docker run -it freebus-docker
