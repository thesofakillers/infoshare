#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Please specify the task to collect the checkpoints for!"
    echo "Example: bash get_checkpoints.sh POS"
    exit 1
fi

if [ $# -lt 2 ]; then
  directory="." 
else
  directory=$2
fi

(
  CURRENT_DIRECTORY=$(basename "$(pwd)")
  # Script should be called in the base directory
  if [[ "$CURRENT_DIRECTORY" == "arrays" ]]; then
      cd ../..
  elif [[ "$CURRENT_DIRECTORY" == "lisa" ]]; then
      cd ..
  fi

  CHECKPOINTS=$(find $2 -type f -name "*.ckpt" | grep "$1" | sort -V)

  if [[ ${#CHECKPOINTS} -eq 0 ]]; then
      echo "No checkpoints were found for task $1!"
  else
      WRITE_FILE="lisa/arrays/${1,,}/all_checkpoints.txt"
      echo "Found $(echo "$CHECKPOINTS" | wc -l) checkpoints for task $1!"
      echo "Writing to $WRITE_FILE"
      echo "$CHECKPOINTS" > "$WRITE_FILE"
  fi
)
