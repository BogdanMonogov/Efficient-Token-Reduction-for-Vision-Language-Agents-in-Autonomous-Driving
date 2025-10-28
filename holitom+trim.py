from llava.model.builder import load_pretrained_model #i added changes to installed llava
import os
from holitom.llava_arch import LlavaMetaForCausalLM_holitom #i added changes to installed holitom
from holitom.modeling_qwen2 import Qwen2Model_holitom
import time
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from llava.mm_utils import get_model_name_from_path 
import re

def load_holitom():

    os.environ["HOLITOM_k"] = '3'
    os.environ["HOLITOM_r"] = '0.5'
    def holitom(model):
        
        # print("################################")
        # print("############ HoliTom ###########")
        # print("################################")

        from llava.model.llava_arch import LlavaMetaForCausalLM
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_holitom.prepare_inputs_labels_for_multimodal
        LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_holitom.encode_images
        LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_holitom.encode_images_multi
        
        LlavaMetaForCausalLM.holitom = LlavaMetaForCausalLM_holitom.holitom
        LlavaMetaForCausalLM.cluster_dpc_knn = LlavaMetaForCausalLM_holitom.cluster_dpc_knn
        LlavaMetaForCausalLM.select_static_windows = LlavaMetaForCausalLM_holitom.select_static_windows
        LlavaMetaForCausalLM.get_static_dynamic_features = LlavaMetaForCausalLM_holitom.get_static_dynamic_features
        LlavaMetaForCausalLM.merge_tokens_by_attention_density = LlavaMetaForCausalLM_holitom.merge_tokens_by_attention_density
        LlavaMetaForCausalLM.merge_tokens_by_density = LlavaMetaForCausalLM_holitom.merge_tokens_by_density
        LlavaMetaForCausalLM.merge_tokens_by_clustering = LlavaMetaForCausalLM_holitom.merge_tokens_by_clustering
        LlavaMetaForCausalLM.add_newline_token = LlavaMetaForCausalLM_holitom.add_newline_token
        
        if os.getenv("HOLITOM_k") is not None and os.getenv("HOLITOM_r") is not None:
            print("HoliTom")
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
            Qwen2Model.forward = Qwen2Model_holitom.forward
        else:
            print("HoliTom (w/o M)")
        
        return model

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path = "lmms-lab/llava-onevision-qwen2-7b-ov",
        model_base = None,
        model_name = "llava_qwen",
        attn_implementation = "sdpa",
        multimodal = True, 
    )    


    model = holitom(model)
    ds = load_dataset("OpenGVLab/MVBench", split="validation")
    return ds, tokenizer, model, image_processor, max_length


def evaluate_on_mvbench_with_holitom(tokenizer, model, image_processor, ds, device = "cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device).eval()
    total, correct = 0, 0
    for ex in tqdm(ds):
        img = Image.open(ex["image"]).convert("RGB")
        question = ex["question"]
        gold = ex["answer"].strip()

        img_inputs = image_processor(images=img, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(
            device, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        text_inputs = tokenizer(question, return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1
            )


        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        total += 1
        correct += int(pred == gold)


    accuracy = correct / total if total else 0.0

    print(f"\nSamples: {total}, Accuracy: {accuracy:.4f}")
    return accuracy



# ds, tokenizer, model, image_processor, max_length = load_holitom()
# evaluate_on_mvbench(tokenizer, model, image_processor, ds)


def load_trim():
    model_path = "FreedomIntelligence/llava-v1.5-7b-TRIM"
    model_name = get_model_name_from_path(model_path)  

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=model_path,
        model_base=None,            
        model_name=model_name
    )
    ds = load_dataset("OpenGVLab/MVBench", split="validation")
    return ds, tokenizer, model, image_processor, max_length


        
def evaluate_on_mvbench_with_trim(tokenizer, model, image_processor, max_samples=None, device="cuda" if torch.cuda.is_available() else "cpu", clip_text_name="openai/clip-vit-large-patch14-336"):


    model.to(device).eval()
    clip_tok = CLIPTokenizer.from_pretrained(clip_text_name)
    clip_txt = CLIPTextModel.from_pretrained(clip_text_name).to(device).eval()

    vt = getattr(model, "get_vision_tower", lambda: getattr(model, "vision_tower", None))()
    vt_model = getattr(vt, "vision_tower", vt)

    def _trim_tokens(pixel_values, prompt_text):
            with torch.no_grad():
                vhid = vt_model(pixel_values=pixel_values).last_hidden_state.squeeze(0)   
                tok = clip_tok([prompt_text], return_tensors="pt", padding=True).to(device)
                eot = tok["attention_mask"].sum(-1) - 1
                up = clip_txt(**tok).last_hidden_state[0, eot.item()]                    
                sims = (F.normalize(vhid, dim=-1) @ F.normalize(up, dim=-1))              
                s = F.softmax(sims, dim=0)
                q1, q3 = torch.quantile(s, 0.25).item(), torch.quantile(s, 0.75).item()
                thr = q3 + 1.5 * (q3 - q1)
                sel = s >= thr
                if not torch.any(sel):
                    sel[torch.argmax(s)] = True
                keep = vhid[sel]
                rest = vhid[~sel].mean(0, keepdim=True) if (~sel).any() else keep.mean(0, keepdim=True)
                return torch.cat([keep, rest], 0)  

    ds = load_dataset("OpenGVLab/MVBench", split="validation")
    total = correct = 0
    times = []

    for i, ex in enumerate(ds):
        if max_samples is not None and i >= max_samples: break
        img = Image.open(ex["image"]).convert("RGB")
        q, gold = ex["question"], str(ex["answer"]).strip()

        pix = image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)
        trimmed = _trim_tokens(pix, q).unsqueeze(0)                                   
        image_embeds = model.mm_projector(trimmed)
        inp = tokenizer(q, return_tensors="pt").to(device)

        with torch.inference_mode():
            t0 = time.perf_counter()
            try:
                output_ids = model.generate(**inp, image_embeds=image_embeds, max_new_tokens=4, do_sample=False)
            except TypeError:
                output_ids = model.generate(**inp, pixel_values=pix, max_new_tokens=4, do_sample=False)
            times.append(time.perf_counter() - t0)

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        total += 1
        correct += int(pred == gold)

    acc = correct / total if total else 0.0
    print(f"\nSamples: {total}, Accuracy: {acc:.4f}")
    return acc

# ds, tokenizer, model, image_processor, max_length = load_trim()
# evaluate_on_mvbench_with_trim(tokenizer, model, image_processor, ds)

