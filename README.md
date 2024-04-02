# Me LLaMA: Foundation Large Language Models for Medical Applications
<div align="left">
    <a target='_blank'>Qianqian Xie<sup>1</sup></span>&emsp;
    <a target='_blank'>Qingyu Chen<sup>1</sup></span>&emsp;
    <a target='_blank'>Aokun Chen<sup>2</sup></span>&emsp;
    <a target='_blank'>Cheng Peng<sup>2</sup></a>&emsp;
    <a target='_blank'>Yan Hu<sup>3</sup></a>&emsp;
    <a target='_blank'>Fongci Lin<sup>1</sup></a>&emsp;
    <a target='_blank'>Xueqing Peng<sup>1</sup></a>&emsp;
    <a target='_blank'>Jimin Huang<sup>1</sup></a>&emsp;
    <a target='_blank'>Jeffrey Zhang<sup>1</sup></a>&emsp;
    <a target='_blank'>Vipina Keloth<sup>1</sup></a>&emsp;
    <a target='_blank'>Huan He<sup>1</sup></a>&emsp;
    <a target='_blank'>Lucila Ohno-Machido<sup>1</sup></a>&emsp;
    <a target='_blank'>Yonghui Wu<sup>2</sup></a>&emsp;
    <a target='_blank'>Hua Xu<sup>1</sup></a>&emsp;
    <a target='_blank'>Jiang Bian<sup>2</sup></a>&emsp;
</div>
<br />

<div align="left">
    <sup>1</sup>Section of Biomedical Informatics and Data Science, School of Medicine, Yale University, New
Haven, CT, USA&emsp;
    <sup>2</sup>Department of Health Outcomes and Biomedical Informatics, College of Medicine, University
of Florida, Gainesville, FL, USA&emsp;
    <sup>3</sup>School of Biomedical Informatics, University of Texas Health Science, Center at Houston,
Houston, TX, USA&emsp;
</div>

<br />


Me LLaMA introduces a groundbreaking suite of open-source medical Large Language Models (LLMs), including the foundation models Me LLaMA 13B/70B and their chat-enhanced versions, Me LLaMA 13B-chat/70B-chat. Developed through the innovative continual pre-training and instruction tuning of LLaMA2, these models leverage a vast medical corpus. This corpus encompasses selected PubMed papers and abstracts, a novel dataset of internationally-recognized medical guidelines, and a general domain corpus, positioning Me LLaMA at the forefront of medical AI research​​.

With its domain-specific advancements, Me LLaMA sets new benchmarks on a wide array of medical reasoning tasks. This makes Me LLaMA a significant asset for medical NLP applications and research​​​​.

## Legal Disclaimer
This software and model are provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

The Me LLaMA models are research tools intended for use in the field of computational linguistics and medicine. They are not intended to be used as diagnostic tools or for clinical decision-making without appropriate validation and regulatory approval. Users of the Me LLaMA models should be aware of their responsibilities to ensure the ethical and appropriate use of this technology, including adherence to any applicable legal and regulatory requirements.

The content and data provided with the models do not replace the expertise of healthcare professionals. Healthcare professionals should use their professional judgment in evaluating the outputs of the Me LLaMA models. Patients should not use the model outputs for self-diagnosis or treatment without consulting a qualified healthcare provider. The information is not intended for clinical decision making, is not intended to be used in the diagnosis or treatment of patients and may not be useful or appropriate for any clinical purpose.

Additionally, users are expressly prohibited from sharing or redistributing any outputs generated from the Me LLaMA models without explicit permission from the authors. This includes, but is not limited to, publishing, distributing, or making the generated outputs available to third parties in any form, whether for commercial purposes or not. This restriction is put in place to ensure responsible use of the technology and to respect the intellectual property rights associated with the models and their outputs. Violation of these terms may result in legal action and revocation of access to the models.

The code and models are available for non-commercial use.

## Model Details

- **Model License:** [LLAMA 2 COMMUNITY LICENSE AGREEMENT](https://ai.meta.com/llama/license/)
- **Code License:** [MIT LICENSE](https://opensource.org/licenses/MIT)
- **Continued-pretrained from model:** [Llama-2](https://huggingface.co/llama) models, extensively adapted for the medical domain through targeted pre-training and instruction tuning
- **Paper:** *[Me LLaMA: Foundation Large Language Models for Medical Applications](https://arxiv.org/abs/2402.12749)*

## Training Procedure

The development of Me LLaMA involved a meticulous process of continual pre-training and instruction tuning of the LLaMA2 models, incorporating an extensive 129B tokens and 214K instruction tuning samples from a diverse array of general, biomedical, and clinical domains. This comprehensive approach aimed to balance domain-specific knowledge with a broader understanding of general context, thereby effectively mitigating catastrophic forgetting issues.

### Continual Pre-training Data

The mixed continual pre-training dataset, comprising 129B tokens, includes a wide range of biomedical literature, clinical notes, and general domain data. This dataset is designed to ensure a deep focus on medical domain knowledge while incorporating a broad spectrum of general knowledge. The dataset's composition includes:

- **Biomedical Papers:** Integration of a vast collection from PubMed Central and PubMed Abstracts.
- **Clinical Notes:** Inclusion of de-identified free-text clinical notes from MIMIC-IV and MIMIC-CXR.
- **General Domain Data:** A subset from the RedPajama dataset, replicating LLaMA's pre-training data.

The pre-training utilized a ratio of 15:1:4 for biomedical, clinical to general domain data, aiming to maintain a strong medical focus while also broadening the model's understanding.

### Training Details

The Me LLaMA models, 13B and 70B, were developed through continuous pre-training and instruction tuning on the University of Florida's HiPerGator supercomputer, equipped with 160 A100 80GB GPUs. The process aimed to adapt the LLaMA2 models for enhanced comprehension and generation of medically relevant text. The training regimen involved:

- **Optimization:** Use of the AdamW optimizer with specific hyperparameters (β1=0.9, β2=0.95), a learning rate of 8e-6, and a weight decay of 0.00001.
- **Learning Rate Scheduler:** A cosine learning rate scheduler with a 0.05 warmup ratio for gradual adaptation.
- **Precision and Efficiency:** bf16 precision for computational efficiency and gradient accumulation over 16 steps, limited to one epoch.
- **Model Parallelism:** Utilization of DeepSpeed for effective model parallelism.

### Instruction Tuning

Following the pre-training phase, Me LLaMA models underwent instruction tuning using 8 H100 GPUs for 3 epochs, employing a learning rate of 1e-5. This phase focused on refining the models' ability to follow instructions and generalize across medical tasks, utilizing LoRA-based parameter-efficient fine-tuning for enhanced performance.

This detailed training procedure underscores the comprehensive approach taken in developing Me LLaMA models, leveraging advanced computational resources and methodologies to achieve state-of-the-art performance in the medical domain.

## How to use

Coming soon!

## Medical Benchmark Inference & Evaluation

### Evaluation

#### Preparation
```bash
git clone git@github.com:BIDS-Xu-Lab/Me-LLaMA.git --recursive
cd Me-LLaMA
pip install -r requirements.txt
cd Me-LLaMA/src/medical-evaluation
pip install -e .[multilingual]
```

#### Automated Task Assessment
Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

For automated evaluation, please follow these instructions:

1. Huggingface Transformer

   To evaluate a model hosted on the HuggingFace Hub (for instance, llama2-7b-hf), use this command:

```bash
python eval.py \
    --model "hf-causal-vllm" \
    --model_args "use_accelerate=True,pretrained=meta-llama/Llama-2-7b-chat-hf,use_fast=False" \
    --tasks "m2sum"
```

More details can be found in the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) documentation.

2. Commercial APIs

Please note, for tasks such as NER, the automated evaluation is based on a specific pattern. This might fail to extract relevant information in zero-shot settings, resulting in relatively lower performance compared to previous human-annotated results.

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-4 \
    --tasks m2sum
```

## Citation
<pre>
@misc{xie2024llama,
      title={Me LLaMA: Foundation Large Language Models for Medical Applications}, 
      author={Qianqian Xie and Qingyu Chen and Aokun Chen and Cheng Peng and Yan Hu and Fongci Lin and Xueqing Peng and Jimin Huang and Jeffrey Zhang and Vipina Keloth and Huan He and Lucila Ohno-Machido and Yonghui Wu and Hua Xu and Jiang Bian},
      year={2024},
      eprint={2402.12749},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>

