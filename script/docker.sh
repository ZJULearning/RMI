#!/bin/bash

# run the docker image
DIR_NOW=$(pwd)

cd ~
echo "current user : ${USER}"

echo ""
echo -n "input the docker image tag:"
read docker_image_tag

echo ""
echo -n "input the mapping port:"
read docker_image_port

docker_image="zhaosssss/torch_lab:"
docker_final_image="${docker_image}${docker_image_tag}"

echo "The docker image is ${docker_final_image}"
echo "run docker image..."


#/usr/bin/docker run --runtime=nvidia --rm -it  --memory-reservation 32G \
#					--shm-size 8G \
#					-v /home/${USER}:/home/${USER} -w ${DIR_NOW} \
#					-p $docker_image_port:$docker_image_port $docker_final_image bash

/usr/bin/docker run --runtime=nvidia --rm -it  --memory-reservation 32G \
						--shm-size 8G \
						-v /home/${USER}:/home/${USER} --user=${UID}:${GID} -w ${DIR_NOW} \
						-v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
						-v /mnt/disk2/wy:/mnt/disk2/wy \
						-v /mnt/disk1/wy:/mnt/disk1/wy \
						-p $docker_image_port:$docker_image_port $docker_final_image bash
