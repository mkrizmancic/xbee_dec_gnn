#!/bin/bash

IMAGE_NAME=${1:-xbee_gnn_img}
CONTAINER_NAME=${2:-xbee_gnn_cont}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Auto-detect if running on Raspberry Pi
IS_RPI=false
if [ -f /proc/cpuinfo ] && grep -q "Raspberry Pi" /proc/cpuinfo; then
  IS_RPI=true
elif [ -f /sys/firmware/devicetree/base/model ] && grep -q "Raspberry Pi" /sys/firmware/devicetree/base/model; then
  IS_RPI=true
fi

if [ "$IS_RPI" = true ]; then
  echo "Running in RPI mode..."
  docker run \
    -it \
    --network host \
    --ipc host \
    --privileged \
    --volume /var/run/dbus:/var/run/dbus \
    --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    --volume $SCRIPT_DIR/volumes/datasets:/root/resources/data \
    --volume $SCRIPT_DIR/volumes/models:/root/resources/models \
    --env TERM=xterm-256color \
    --name $CONTAINER_NAME \
    $IMAGE_NAME
else
  echo "Running in default mode..."

  # Set up X11 forwarding (display)
  XAUTH=/tmp/.docker.xauth
  XSOCK=/tmp/.X11-unix
  xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
  if [ ! -f $XAUTH ]; then
      if [ ! -z "$xauth_list" ]; then
          echo $xauth_list | xauth -f $XAUTH nmerge -
      else
          touch $XAUTH
      fi
      chmod a+r $XAUTH
  fi

  # Set up SSH agent forwarding
  ln -sf $SSH_AUTH_SOCK ~/.ssh/ssh_auth_sock

  docker run \
    -it \
    --network host \
    --ipc host \
    --env TERM=xterm-256color \
    --volume ~/.ssh/ssh_auth_sock:/run/ssh-agent \
    --env SSH_AUTH_SOCK=/run/ssh-agent \
    --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    --volume=$XSOCK:$XSOCK:rw \
    --volume=$XAUTH:$XAUTH:rw \
    --env="XAUTHORITY=${XAUTH}" \
    --env DISPLAY=$DISPLAY \
    --device-cgroup-rule='c 188:* rmw' \
    --device-cgroup-rule='c 166:* rmw' \
    --volume /dev:/dev \
    --volume $SCRIPT_DIR/volumes/datasets:/root/resources/data \
    --volume $SCRIPT_DIR/volumes/models:/root/resources/models \
    --name $CONTAINER_NAME \
    $IMAGE_NAME
fi
