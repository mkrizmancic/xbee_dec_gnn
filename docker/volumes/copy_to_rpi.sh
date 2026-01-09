#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEST_DIR=${1:-/home/pi/xbee_dec_gnn/docker/volumes}

echo "Copying volumes folder to Raspberry Pis..."
echo "Destination directory: $DEST_DIR"

for i in {0..4}; do
  HOSTNAME="rpi$i.local"

  scp -q -r $SCRIPT_DIR/* pi@$HOSTNAME:$DEST_DIR/

  if [ $? -eq 0 ]; then
      echo "✓ Successfully copied to $HOSTNAME"
  else
      echo "✗ Failed to copy to $HOSTNAME"
  fi
done

echo ""
echo "Done!"
