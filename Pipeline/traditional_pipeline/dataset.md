# Preparing the dataset

**Input:** folder of creator videos (~5–10 min each)  
**Output:** precomputed clips for **LTX-2.3 audio-video LoRA** training  
**Scripts:** `VideoMaking/ltx_lora_training/`

## One-time setup

```bash
cd VideoMaking/ltx_lora_training
bash setup_trainer.sh      # LTX trainer env
bash download_models.sh    # ltx-2.3-22b-dev + Gemma
```

Put raw `.mp4` files in `debunk_exisiting_youtubers/outputs/vids/`.

## Pipeline (4 steps)

| Step | What | Output |
|------|------|--------|
| 1. Scene cut | PySceneDetect per video | `scene_captioning/outputs/scenes/<title>/` |
| 2. Caption | Qwen3-VL storyboard per clip | `scene_captioning/outputs/captions/<title>/captions.json` |
| 3. Build manifest | Merge clips + captions (≥5s) | `scene_captioning/outputs/scenes/dataset.json` |
| 4. Preprocess | LTX-2.3 VAE + audio + text embeds | `scene_captioning/outputs/scenes/.precomputed/` |

Run all four:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NUM_PROCESSES=7 \
  bash run_dataset_from_vids.sh
```

## Train LoRA

```bash
bash run_train.sh
```

Checkpoints → `ltx_lora_training/outputs/youtube_explainer_lora/`
