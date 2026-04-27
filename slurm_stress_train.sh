echo "--- Baseline: 224x224 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --num-classes 10 \
    --img-size 224 \
    "$DATA/val"

echo "--- Small: 112x112 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --num-classes 10 \
    --img-size 112 \
    "$DATA/val"

echo "--- Large: 336x336 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --num-classes 10 \
    --img-size 336 \
    "$DATA/val"

echo "--- Very large: 448x448 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --num-classes 10 \
    --img-size 448 \
    "$DATA/val"

echo "--- Tiny: 64x64 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --num-classes 10 \
    --img-size 64 \
    "$DATA/val"

echo "--- Huge: 512x512 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --num-classes 10 \
    --img-size 512 \
    "$DATA/val"