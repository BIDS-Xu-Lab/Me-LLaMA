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
    <sup>1</sup>Department of Biomedical Informatics and Data Science, School of Medicine, Yale University, New
Haven, CT, USA&emsp;
    <sup>2</sup>Department of Health Outcomes and Biomedical Informatics, College of Medicine, University
of Florida, Gainesville, FL, USA&emsp;
    <sup>3</sup>School of Biomedical Informatics, University of Texas Health Science, Center at Houston,
Houston, TX, USA&emsp;
</div>

<br />


[Me LLaMA](https://www.physionet.org/content/me-llama/1.0.0/) introduces a groundbreaking suite of open-source medical Large Language Models (LLMs), including the foundation models Me LLaMA 13B/70B and their chat-enhanced versions, Me LLaMA 13B-chat/70B-chat. Developed through the innovative continual pre-training and instruction tuning of LLaMA2, these models leverage a vast medical corpus. This corpus encompasses selected PubMed papers and abstracts, a novel dataset of internationally-recognized medical guidelines, and a general domain corpus, positioning Me LLaMA at the forefront of medical AI research​​.

With its domain-specific advancements, Me LLaMA sets new benchmarks on a wide array of medical reasoning tasks. This makes Me LLaMA a significant asset for medical NLP applications and research​​​​.

## Availability

The code, datasets, and models are available for non-commercial use.

- **Code**: See above.
- **Datasets**: Check our Hugging Face [collection](https://huggingface.co/collections/clinicalnlplab/ibe-65de0abfafad82f111fe5392).
- **Models**: Please visit our [PhysioNet repository](https://www.physionet.org/content/me-llama/1.0.0/). Note that a PhysioNet account, training, and data usage agreement are required.
- **New model based LLaMA3**: We have developed new version of Me-LLaMA model based on LLaMA3-8B model. The model can be accessed here: [YBXL/Med-LLaMA3-8B](https://huggingface.co/YBXL/Med-LLaMA3-8B).

## Legal Disclaimer
This software and model are provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

The Me LLaMA models are research tools intended for use in the field of computational linguistics and medicine. They are not intended to be used as diagnostic tools or for clinical decision-making without appropriate validation and regulatory approval. Users of the Me LLaMA models should be aware of their responsibilities to ensure the ethical and appropriate use of this technology, including adherence to any applicable legal and regulatory requirements.

The content and data provided with the models do not replace the expertise of healthcare professionals. Healthcare professionals should use their professional judgment in evaluating the outputs of the Me LLaMA models. Patients should not use the model outputs for self-diagnosis or treatment without consulting a qualified healthcare provider. The information is not intended for clinical decision making, is not intended to be used in the diagnosis or treatment of patients and may not be useful or appropriate for any clinical purpose.

Additionally, users are expressly prohibited from sharing or redistributing any outputs generated from the Me LLaMA models without explicit permission from the authors. This includes, but is not limited to, publishing, distributing, or making the generated outputs available to third parties in any form, whether for commercial purposes or not. This restriction is put in place to ensure responsible use of the technology and to respect the intellectual property rights associated with the models and their outputs. Violation of these terms may result in legal action and revocation of access to the models.


## Model Details

- **Model License:** [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/)
- **Code License:** [MIT LICENSE](https://opensource.org/licenses/MIT)
- **Continued-pretrained from model:** [Llama-2](https://huggingface.co/llama) models, extensively adapted for the medical domain through targeted pre-training and instruction tuning
- **Evaluation Datasets:** [Huggingface Evaluation Datasets Collection](https://huggingface.co/collections/clinicalnlplab/ibe-65de0abfafad82f111fe5392)
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

To utilize the Me LLaMA model locally, begin by acquiring the necessary model files from [our PhysioNet project](https://www.physionet.org/content/me-llama/1.0.0/).

First, ensure that both the `torch` and `transformers` libraries are installed in your Python environment. These libraries are required for working with the model.

For basic text generation, you'll employ a pipeline from the `transformers` library. This method simplifies the process of generating text. Here's how you can set it up:

```python
from transformers import pipeline

# Ensure you replace "FOLDER_PATH_TO_MODEL" with the actual path to your model files.
pipe = pipeline("text-generation", model="FOLDER_PATH_TO_MODEL")

# Example usage for generating text.
generated_text = pipe("The medical condition is characterized by", num_return_sequences=1)
print(generated_text)
```

This code snippet demonstrates how to generate text based on a prompt. The `num_return_sequences=1` argument specifies that you want to generate one sequence of text.

For tasks requiring more customization or fine-tuning capabilities, you might prefer directly loading the tokenizer and model. This approach gives you more control over the text generation process, allowing you to adjust parameters like the maximum length of the generated text. Here's a more detailed example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model from your local model directory.
# Don't forget to replace "FOLDER_PATH_TO_MODEL" with the actual path to your model files.
tokenizer = AutoTokenizer.from_pretrained("FOLDER_PATH_TO_MODEL")
model = AutoModelForCausalLM.from_pretrained("FOLDER_PATH_TO_MODEL")

# Tokenizing input text for the model.
input_ids = tokenizer("[INPUT SENTENCE]", return_tensors="pt").input_ids

# Generating output based on the input_ids.
# You can adjust the max_length parameter as necessary for your use case.
generated_tokens = model.generate(input_ids, max_length=50)

# Decoding the generated tokens to produce readable text.
generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(generated_text)
```

This setup allows for more nuanced interactions with the model, such as fine-tuning on specific datasets or modifying the generation parameters for different outputs. Remember to replace "[INPUT SENTENCE]" with the sentence or prompt you want the model to expand on or respond to.


## Medical Benchmark Inference & Evaluation

### Evaluation

#### Preparation
```bash
git clone git@github.com:BIDS-Xu-Lab/Me-LLaMA.git --recursive
cd Me-LLaMA
pip install poetry
poetry install
cd src/medical-evaluation
poetry run pip install -e .[multilingual]
poetry run python -m spacy download en_core_web_lg
```

#### Automated Task Assessment
Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

For automated evaluation, please follow these instructions:

1. Huggingface Transformer

   To evaluate a model hosted on the HuggingFace Hub (for instance, llama2-7b-hf), change this command in `scripts/run_evaluation.sh`:

```bash
poetry run python src/eval.py \
    --model "hf-causal-vllm" \
    --model_args "use_accelerate=True,pretrained=meta-llama/Llama-2-7b-chat-hf,use_fast=False" \
    --tasks "PUBMEDQA,MedQA,MedMCQA,EmrQA,i2b2,DDI2013,hoc,MTSample,PUBMEDSUM,MimicSum,BioNLI,MedNLI"
```
   
   Then run bash command:

```bash
bash scripts/run_evaluation.sh
```

More details can be found in the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) documentation.

2. Commercial APIs

Perform the same steps as the open-sourced models, first to change the bash file with:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
poetry run python src/eval.py \
    --model gpt-4 \
    --tasks "PUBMEDQA,MedQA,MedMCQA,EmrQA,i2b2,DDI2013,hoc,MTSample,PUBMEDSUM,MimicSum,BioNLI,MedNLI"
```

Please note, for tasks such as NER, the automated evaluation is based on a specific pattern. This might fail to extract relevant information in zero-shot settings, resulting in relatively lower performance compared to previous human-annotated results.

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

