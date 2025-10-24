import time
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from common_utils import load_vit, image_to_visual_tokens


def tokenpacker(feats, M=144):
    B, N, D = feats.shape
    centers = torch.randn(1, M, D, device=feats.device)
    q = centers @ feats.transpose(-1, -2)
    attn = torch.softmax(q, dim=-1)
    packed = attn @ feats
    return packed

def vistog(feats, M=144):
    B, N, D = feats.shape
    grouped = []
    for b in range(B):
        x = feats[b].cpu().numpy()
        kmeans = KMeans(n_clusters=M, n_init=5).fit(x)
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        grouped.append(centers)
    return torch.stack(grouped).to(feats.device)

def lftr(feats, keep_ratio=0.25):
    B, N, D = feats.shape
    K = max(1, int(N * keep_ratio))
    energy = feats.pow(2).sum(-1)
    out = []
    for b in range(B):
        _, idx = torch.topk(energy[b], K)
        out.append(feats[b, idx])
    # pad to K
    out_tensor = torch.zeros(B, K, D, device=feats.device)
    for b in range(B):
        out_tensor[b, :out[b].shape[0]] = out[b]
    return out_tensor

def star_prune(feats, attn, keep_ratio=0.5):
    B, H, Q, K = attn.shape
    score = attn.abs().sum(dim=(1,2)) 
    keep_k = max(1, int(K * keep_ratio))
    out = []
    for b in range(B):
        _, idx = torch.topk(score[b], keep_k)
        out.append(feats[b, idx])
    D = feats.shape[-1]
    out_tensor = torch.zeros(B, keep_k, D, device=feats.device)
    for b in range(B):
        out_tensor[b, :out[b].shape[0]] = out[b]
    return out_tensor


def load_llava(model_id="llava-hf/llava-0.5b-hf"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
    return processor, model


def reduce_visual_tokens(model, image, method="tokenpacker", M=144):
    vit = model.vision_tower
    with torch.no_grad():
        feats = image_to_visual_tokens(vit, image, device="cuda")
    if method == "tokenpacker":
        reduced = tokenpacker(feats, M=M)
    elif method == "vistog":
        reduced = vistog(feats, M=M)
    elif method == "lftr":
        reduced = lftr(feats, keep_ratio=M/feats.shape[1])
    else:
        reduced = feats

    proj = model.multi_modal_projector
    visual_embeds = proj(reduced)
    return visual_embeds


def evaluate_on_mvbench(model_id, method="tokenpacker", M=144, max_samples=None):
    processor, model = load_llava(model_id)
    ds = load_dataset("OpenGVLab/MVBench", split="validation")
    total = 0
    correct = 0
    times = []
    for idx, example in enumerate(ds):
        if max_samples and idx>=max_samples:
            break
        img = Image.open(example["image"]).convert("RGB")
        prompt = example["question"]
        choices = example["options"]
        gold = example["answer"]
        inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda", torch.float16)
        start = time.perf_counter()
        visual_embeds = reduce_visual_tokens(model, inputs["pixel_values"][0], method=method, M=M)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                visual_embeds=visual_embeds,
                max_new_tokens=1,
            )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        pred_token = outputs[0, -1].item()

        pred_answer = processor.decode(outputs[0], skip_special_tokens=True)
        total += 1
        if pred_answer.strip() == gold.strip():
            correct += 1
        if idx % 100 == 0 and idx>0:
            print(f"Processed {idx} samples...")
    avg_time = sum(times)/len(times)
    accuracy = correct/total
    print(f"\nResults for method={method}, M={M}")
    print(f"Samples: {total}, Accuracy: {accuracy:.4f}, Avg inference time: {avg_time:.4f} sec")
    return accuracy, avg_time

if __name__ == "__main__":
    for method in ["none", "tokenpacker", "vistog", "lftr"]:
        evaluate_on_mvbench("llava-hf/llava-0.5b-hf", method=method, M=128, max_samples=500)
