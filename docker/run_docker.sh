#!/bin/bash

IMAGE_NAME=${1:-xbee_gnn_img}
CONTAINER_NAME=${2:-xbee_gnn_cont}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run \
  -it \  # Interactive terminal
  --network host \  # Use host network (for LAN and Internet access)
  --ipc host \      # Share IPC namespace (to allow inter-process communication between nodes in different containers)
  --volume /var/run/dbus:/var/run/dbus \  # D-Bus socket for system services (needed by Avahi)
  --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \  # mDNS/Zeroconf (Avahi - to enable easy discovery on LAN)
  --volume $SCRIPT_DIR/volumes/datasets:/root/resources/data \        # Mount datasets
  --volume $SCRIPT_DIR/volumes/models:/root/resources/models \        # Mount models
  --volume $SCRIPT_DIR/../xbee_dec_gnn:/root/other_ws/xbee_dec_gnn \  # Mount source code
  --volume /tmp/.x11-unix:/tmp/.x11-unix \  # X11 display server (allows GUI apps)
  --env DISPLAY=$DISPLAY \                  # X11 display (allows GUI apps)
  --env TERM=xterm-256color \  # Enable terminal colors
  --name $CONTAINER_NAME \
  $IMAGE_NAME