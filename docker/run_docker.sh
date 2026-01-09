#!/bin/bash

IMAGE_NAME=${1:-xbee_gnn_img}
CONTAINER_NAME=${2:-xbee_gnn_cont}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run \
  -it \
  --network host \
  --ipc host \
  --privileged \
  --volume /var/run/dbus:/var/run/dbus \
  --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
  --volume $SCRIPT_DIR/volumes/datasets:/root/resources/data \
  --volume $SCRIPT_DIR/volumes/models:/root/resources/models \
  --volume $SCRIPT_DIR/../xbee_dec_gnn:/root/other_ws/xbee_dec_gnn \
  --volume ~/.ssh/rpi_student_key:/root/.ssh/id_ed25519:ro \
  --volume /tmp/.x11-unix:/tmp/.x11-unix \
  --env DISPLAY=$DISPLAY \
  --env TERM=xterm-256color \
  --name $CONTAINER_NAME \
  $IMAGE_NAME