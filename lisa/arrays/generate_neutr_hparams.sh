if [ $# -eq 0 ]; then
    echo "Please specify the task to generate the neutralizer hyper-parameters for!"
    echo "Example: sh generate_neutr_hparams.sh POS"
    exit 1
fi

WRITE_FILE="${1,,}_neutr_hparams.txt"
truncate -s 0 "$WRITE_FILE" # clear the file

while read -r CHECKPOINT; do
    while read -r NEUTRALIZER; do
        echo "--checkpoint ${CHECKPOINT} --neutralizer ${NEUTRALIZER}" >> "$WRITE_FILE"
    done <<< "$(cat "${1,,}_neutralizers.txt")"
done <<< "$(cat "all_checkpoints_${1,,}.txt")"

echo "Wrote $(wc -l < "$WRITE_FILE") lines to $WRITE_FILE"
