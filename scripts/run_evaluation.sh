#! /bin/bash
eval_path='[Absolute path of mellama]'
export PYTHONPATH="$eval_path/src:$eval_path/src/medical-evaluation:$eval_path/src/metrics/BARTScore"
export CUDA_VISIBLE_DEVICES=
echo $PYTHONPATH

poetry run python src/eval.py \
    --model hf-causal-vllm \
    --tasks "PUBMEDQA,MedQA,MedMCQA,EmrQA,i2b2,DDI2013,hoc,MTSample,PUBMEDSUM,MimicSum,BioNLI,MedNLI" \
    --model_args "use_accelerate=True,pretrained=meta-llama/Llama-2-7b-chat-hf,use_fast=False" \
    --no_cache \
    --batch_size 50000 \
    --write_out \
    --output_path './results.json'
