#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Please specify the task to collect the checkpoints for!"
    echo "Example: bash get_checkpoints.sh POS"
    exit 1
fi

(
  if [[ $(basename "$(pwd)") == "arrays" ]]; then
      # Script should be called in the base directory
      cd ../..
  fi

  CHECKPOINTS=$(find . -type f -name "*.ckpt" | grep "$1" | sort -V)

  if [[ ${#CHECKPOINTS} -eq 0 ]]; then
      echo "No checkpoints were found for task $1!"
  else
      WRITE_FILE="lisa/arrays/${1,,}/all_checkpoints.txt"
      echo "Found $(echo "$CHECKPOINTS" | wc -l) checkpoints for task $1!"
      echo "Writing to $WRITE_FILE"
      echo "$CHECKPOINTS" > "$WRITE_FILE"
  fi
)
