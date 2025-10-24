# Efficient-Token-Reduction-for-Vision-Language-Agents-in-Autonomous-Driving
The following repository contains source code for evaluating different token-reduction techniques for VLMs 

All of the source code is stored in the Models folder, while the project presentation stored separately under the name "Presentation.pdf". 
Each filename in the folder reflects the name of the model used. All models are tested on the MVBench "Moving count" and "Action Sequence" datasets. Adopted metrics are acccuracy and computational time.

## Comparative Overview of Recent Token Optimization and Multimodal Adaptation Methods

### 1. TRIM — Token Reduction using CLIP Metric
**Reference:** [arXiv:2409.10994](https://arxiv.org/abs/2409.10994)

**Objective:**  
Efficiently reduce redundant visual tokens while retaining semantically relevant content in vision-language models (VLMs).

**Methodology:**  
1. Extract image features via CLIP’s visual encoder.  
2. Compute cosine similarity between each visual token and the pooled text embedding.  
3. Rank tokens by importance and identify outliers using the IQR (inter-quartile range) criterion.  
4. Retain the most relevant tokens; aggregate discarded ones into a single summary token to avoid context loss.  

---

### 2. HoliTom — Holistic Token Merging for Video LLMs
**Reference:** [arXiv:2505.21334](https://arxiv.org/abs/2505.21334)

**Objective:**  
Enable large video-language models (Video-LLMs) to process long sequences efficiently through hierarchical token merging.

**Methodology:**  
1. **Outer-LLM compression:** performs redundancy-aware temporal segmentation followed by spatio-temporal merging of redundant frames.  
2. **Inner-LLM compression:** merges semantically similar tokens inside the LLM itself using representation-level similarity.  
3. Both stages are *training-free* and complementary, ensuring consistency and efficiency.  

---

### 3. PLLaVA — Parameter-Free LLaVA Extension from Images to Videos
**Reference:** [arXiv:2404.16994](https://arxiv.org/abs/2404.16994)

**Objective:**  
Extend image-language alignment models to video understanding tasks without retraining.

**Methodology:**  
1. Reuse pre-trained image-language model weights.  
2. Apply temporal feature smoothing (pooling) to mitigate large-norm imbalance across frames.  
3. Integrate pooled features for dense video captioning and video-QA without fine-tuning.  

---

### 4. FasterVLM — [CLS] Attention is All You Need for Training-Free Visual Token Pruning
**Reference:** [arXiv:2412.01818](https://arxiv.org/abs/2412.01818)

**Objective:**  
Accelerate multimodal LLM inference by removing redundant visual tokens using attention-based saliency estimation.

**Methodology:**  
1. Analyze the attention distribution between the `[CLS]` token and visual tokens.  
2. Rank tokens according to their attention weight to `[CLS]`.  
3. Prune the least attended tokens before feeding them into the LLM.  

---

### 5. HiPrune — Hierarchical Attention-Based Visual Token Pruning
**Reference:** [arXiv:2508.00553](https://arxiv.org/abs/2508.00553)

**Objective:**  
Introduce a hierarchical, training-free framework for visual token pruning by exploiting layer-wise attention behavior.

**Methodology:**  
1. Examine hierarchical attention patterns in the vision encoder.  
2. Identify three token types:  
   - **Anchor tokens:** high mid-layer attention (object centers).  
   - **Buffer tokens:** spatial neighbors of anchors.  
   - **Register tokens:** global context tokens from deep layers.  
3. Preserve informative tokens, discard the rest dynamically.  

---

### 6. LLaVA-Scissor — Semantic Connected Component-Based Token Compression
**Reference:** [arXiv:2506.21862](https://arxiv.org/abs/2506.21862)

**Objective:**  
Perform semantic token compression for long video sequences in a *training-free* manner.

**Methodology:**  
1. Group visual tokens into **Semantic Connected Components (SCCs)** that preserve semantic completeness.  
2. Conduct a **two-stage compression**: spatial reduction across frames followed by temporal merging across segments.  
3. Reorder and feed compact tokens into the video LLM without additional training.  

---

### 7. LLaVA-OneVision — A Universal Multimodal Foundation Model
**Reference:** [arXiv:2408.03326](https://arxiv.org/abs/2408.03326)

**Objective:**  
Build a unified multimodal model capable of processing single images, multiple images, and videos under a shared architecture.

**Methodology:**  
1. Integrate datasets and representations from the LLaVA-NeXT family to create a general multimodal corpus.  
2. Design an architecture supporting flexible context windows across image and video modalities.  
3. Employ knowledge transfer and lightweight adaptation for diverse visual understanding tasks.
