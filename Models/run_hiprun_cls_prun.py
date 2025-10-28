import time
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from PIL import Image
import numpy as np

def fastervlm_prune_by_cls_attention(v_feats, cls_attention_scores, reduction_ratio=0.9):
    B, N, D = v_feats.shape
    keep_k = max(1, int(N * (1.0 - reduction_ratio)))
    out = []
    for b in range(B):
        scores = cls_attention_scores[b]
        _, idx = torch.topk(scores, keep_k)
        out.append(v_feats[b, idx])
    out_tensor = torch.zeros(B, keep_k, D, device=v_feats.device)
    for b in range(B):
        out_tensor[b, :out[b].shape[0]] = out[b]
    return out_tensor

def hiprune_prune(v_feats, keep_ratio=0.5):
    B, N, D = v_feats.shape
    K = max(1, int(N * keep_ratio))
    energy = v_feats.pow(2).sum(-1)
    out = []
    for b in range(B):
        _, idx = torch.topk(energy[b], K)
        out.append(v_feats[b, idx])
    out_tensor = torch.zeros(B, K, D, device=v_feats.device)
    for b in range(B):
        out_tensor[b, :out[b].shape[0]] = out[b]
    return out_tensor

def pllava_baseline(v_feats):
    return v_feats

def load_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
    return processor, model

def extract_visual_tokens(model, image_tensor)
    vision_tower = model.vision_tower
    with torch.no_grad():
        out = vision_tower(image_tensor.unsqueeze(0))
    # Пусть out.tokens = B x N x D
    v_feats = out.last_hidden_state
    cls_scores = out.attentions_cls
    return v_feats, cls_scores

def evaluate_on_mvbench(model_id, method="baseline", reduction_ratio=0.9, max_samples=None):
    processor, model = load_model(model_id)
    ds = load_dataset("OpenGVLab/MVBench", split="validation")
    total = 0
    correct = 0
    times = []
    for idx, example in enumerate(ds):
        if max_samples and idx >= max_samples:
            break
        img = Image.open(example["image"]).convert("RGB")
        question = example["question"]
        choices = example["options"]
        gold = example["answer"]

        inputs = processor(images=img, text=question, return_tensors="pt").to("cuda", torch.float16)
        pixel_values = inputs["pixel_values"][0]

        start = time.perf_counter()
        v_feats, cls_scores = extract_visual_tokens(model, pixel_values)

        if method == "baseline":
            reduced_feats = pllava_baseline(v_feats)
        elif method == "hiprune":
            reduced_feats = hiprune_prune(v_feats, keep_ratio=1.0 - reduction_ratio)
        elif method == "fastervlm":
            reduced_feats = fastervlm_prune_by_cls_attention(v_feats, cls_scores, reduction_ratio=reduction_ratio)
        else:
            reduced_feats = v_feats

        visual_embeds = model.multi_modal_projector(reduced_feats)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                visual_embeds=visual_embeds,
                max_new_tokens=1
            )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        pred = processor.decode(outputs[0], skip_special_tokens=True).strip()
        total += 1
        if pred == gold.strip():
            correct += 1

        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx} samples...")

    avg_time = sum(times) / len(times)
    accuracy = correct / total
    print(f"\nModel: {model_id} | Method: {method} | Reduction_ratio: {reduction_ratio}")
    print(f"Samples: {total}, Accuracy: {accuracy:.4f}, Avg inference time: {avg_time:.4f} sec")
    return accuracy, avg_time

if __name__ == "__main__":
    model_ids = {
        "pllava_7b": "mm-eval/pllava-7b",
        "llava_hiprune_7b": "llava-hf/llava-1.5-7b-hf",
        "llava_fastervlm_7b": "llava-hf/llava-1.5-7b-hf"
    }
    for name, mid in model_ids.items():
        for method, rr in [("baseline", 0.0), ("hiprune", 0.5), ("fastervlm", 0.9)]:
            evaluate_on_mvbench(mid, method=method, reduction_ratio=rr, max_samples=200)
