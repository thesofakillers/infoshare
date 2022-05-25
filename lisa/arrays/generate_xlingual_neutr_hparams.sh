#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Please specify the task to generate the cross-lingual neutralizer hyper-parameters for!"
    echo "Example: bash generate_xlingual_neutr_hparams.sh POS"
    exit 1
fi

(
  CURRENT_DIRECTORY=$(basename "$(pwd)")
  # Script should be called in the 'arrays' directory
  if [[ "$CURRENT_DIRECTORY" == "arrays" ]]; then
      true # do nothing
  elif [[ "$CURRENT_DIRECTORY" == "lisa" ]]; then
      cd arrays
  else
      # assume we are in the base directory
      cd lisa/arrays
  fi

  WRITE_FILE="${1,,}/xlingual_neutr_hparams.txt"
  truncate -s 0 "$WRITE_FILE" # clear the file

  while read -r TARGET_CHECKPOINT; do
      while read -r NEUTR_CHECKPOINT; do
          if [[ "$TARGET_CHECKPOINT" != "$NEUTR_CHECKPOINT" ]]; then
              while read -r NEUTRALIZER; do
                  echo "--neutr_checkpoint ${NEUTR_CHECKPOINT} --target_checkpoint ${TARGET_CHECKPOINT} --neutralizer ${NEUTRALIZER}" >> "$WRITE_FILE"
              done <<< "$(cat "${1,,}/neutralizers.txt")"
          fi
      done <<< "$(cat "${1,,}/cherry_checkpoints.txt")"
  done <<< "$(cat "${1,,}/cherry_checkpoints.txt")"

  echo "Wrote $(wc -l < "$WRITE_FILE") lines to $WRITE_FILE"
)
