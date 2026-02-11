from qwen_3_asr_helper import OVQwen3ASRModel

# 1. 初始化 OpenVINO 模型
device = "CPU" # 可改为 "GPU"
ov_model = OVQwen3ASRModel.from_pretrained(
    model_dir=str('Qwen3-ASR-0.6B-OV'),
    device=device,
    max_inference_batch_size=32,
    max_new_tokens=1024,
)

# 2. 准备音频推理
# 官方示例音频：https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav
audio_path = "asr_en.wav" 
# audio_path = "debug_wavs/capture_20260210_191452_0001.wav"

print("🎙️ 正在进行语音识别...")
results = ov_model.transcribe(
    audio=audio_path,
    language=None # 自动检测语种
)

# 3. 输出结果
print(f"【检测语种】: {results[0].language}")
print(f"【识别文本】: {results[0].text}")