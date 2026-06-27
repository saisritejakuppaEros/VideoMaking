import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------
# Load scene captions
# --------------------------------------------------------




json_path = "/mnt/data0/harsha/new_paper/VideoMaking/scene_captioning/outputs/scene_caption_op/3 surprising ways microplastics can enter your body/captions.json"

with open(json_path, "r") as f:
    scene_data = json.load(f)

scene_text = json.dumps(scene_data, indent=2)

# --------------------------------------------------------
# Prompt
# --------------------------------------------------------

prompt = f"""
You are an expert science writer.

Your task is NOT to summarize the captions.

Instead, reconstruct an entirely new narration from the scene descriptions.

The captions are noisy multimodal outputs containing:
- visual descriptions
- OCR text
- existing speech transcripts
- sounds

Treat the speech transcript only as weak evidence.
The visual description should be the primary source of information.

Your job is to infer what the video is trying to communicate and write a coherent educational narration.

Requirements:

1. Read ALL scenes first before writing.
2. Understand the complete story.
3. Produce one continuous narration.
4. The narration should sound like a researcher explaining the topic.
5. Never imitate TED-Ed or YouTube creators.
6. Never mention:
   - "the video shows"
   - "in this scene"
   - "as we can see"
   - camera movements
   - animations
7. Expand missing scientific context whenever appropriate.
8. If a scene is ambiguous, infer the most likely scientific explanation from neighboring scenes.
9. Preserve factual accuracy.
10. Keep transitions smooth.
11. Avoid repetition.
12. Build curiosity naturally without clickbait.
13. Explain cause-and-effect relationships.
14. Use precise scientific language while remaining easy to understand.
15. Output only the narration.

The narration should roughly follow the scene order but should read as one continuous article.

Scene Captions:

{scene_text}
"""

# --------------------------------------------------------
# Load model
# --------------------------------------------------------

model_name = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# --------------------------------------------------------
# Separate thinking and final answer
# --------------------------------------------------------

try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking = tokenizer.decode(
    output_ids[:index],
    skip_special_tokens=True
).strip()

script = tokenizer.decode(
    output_ids[index:],
    skip_special_tokens=True
).strip()

print("=" * 80)
print("THINKING")
print("=" * 80)
print(thinking)

print("\n" + "=" * 80)
print("SCRIPT")
print("=" * 80)
print(script)