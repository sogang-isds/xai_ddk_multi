SCRIPT_DIR=$(cd $(dirname "$0") && pwd)
PARENT_DIR=$(dirname "${SCRIPT_DIR}")

# check if config.py exists
if [ ! -f "$PARENT_DIR/config.py" ]; then
    echo "config.py not found!"
    exit 1
fi

# check if model exists
model_paths=(
  "$PARENT_DIR/models/multi_input_model.ckpt"
  "$PARENT_DIR/models/intelligibility_model.ckpt"
  "$PARENT_DIR/models/vad_model.ckpt"
)

for model_path in "${model_paths[@]}"; do
    if [ ! -f "$model_path" ]; then
        echo "Error: $model_path not found!"
        exit 1
    fi
done
