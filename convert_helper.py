from pathlib import Path
from qwen_3_asr_helper import convert_qwen3_asr_model

model_id = 'Qwen/Qwen3-ASR-0.6B'
model_name = model_id.split("/")[-1]
ov_model_dir = Path(f"{model_name}-OV")

# Convert model to OpenVINO format
# This will skip conversion if the model already exists
convert_qwen3_asr_model(
    model_id=model_id,
    output_dir=ov_model_dir,
    quantization_config=None,  # Set to {"mode": "INT8_SYM"} for INT8 quantization
)