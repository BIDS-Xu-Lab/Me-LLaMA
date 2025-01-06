import json
import ast
import random
import argparse
import pandas as pd
import re
from tqdm import tqdm
from datasets import Dataset, DatasetInfo, DatasetDict, Features, Value, load_dataset
from transformers import LlamaTokenizer

def read_jsonl_data(filename, subset, split):
    with open(filename) as f:
        data = f.readlines()
    data = [json.loads(val) for val in data]
    return data


def read_table_data(filename, subset, split):
    data = pd.read_table(filename).to_dict('records')
    return data


def read_json_data(filename, subset, split):
    data = json.load(open(filename))
    return data


def read_huggingface_repo(repo_name, subset, split):
    data = load_dataset(repo_name, subset, split=split)
    return data


READ_METHODS = {
    "json": read_json_data,
    "jsonl": read_jsonl_data,
    "tsv": read_table_data,
}


class InstructionDataset:
    BASE_REPO = "hippocrates"

    def build_finetuning_instruction(self, data, prompt, index):
        return {"id": f"{self.dataset}{index}", "conversations": [
            {"from": "human", "value": prompt},
            {"from": "agent", "value": data["answer"]},
        ], "text": prompt,}

    def build_classification_instruction(self, data, prompt, index):
        return {"id": f"{self.dataset}{index}", "query": prompt,
                    "answer": data["answer"], "choices": self.choices,
                    "gold": self.choices.index(data["answer"])}

    def build_absumm_instruction(self, data, prompt, index):
        return {"id": f"{self.dataset}{index}", "query": prompt,
                    "answer": data["answer"]}

    def build_multicalss_instruction(self, data, prompt, index):
        label_list = [0] * len(self.choices)
        input_labels = data["answer"].strip('\n').split(';')
        for each_label in input_labels:
            each_label = str(each_label)
            label_list[self.choices.index(each_label)] = 1
                
        return {"id": f"{self.dataset}{index}", "query": prompt,
                    "answer": data["answer"], "choices": self.choices,
                    "gold":label_list}


    def construct_instructions(self, data, eval_format=False, limit=None):
        instructions = []
        construct_dict = {
            "classification": self.build_classification_instruction,
            "abstractivesummarization": self.build_absumm_instruction,
            "multiclassification": self.build_multicalss_instruction}
        construct_method = construct_dict[self.task_type] if eval_format else self.build_finetuning_instruction

        for index, datum in enumerate(tqdm(data)):
            # if datum["source"] == "umls_relation":
            
            fetched_data = self.fetch_data(datum)
            filled_prompt = self.prompt.format(**fetched_data) 
            instruction = construct_method(fetched_data, filled_prompt, len(instructions))
            instructions.append(instruction)

        if not eval_format:
            random.shuffle(instructions)
            instructions = instructions if limit is None else instructions[:limit]

        with open("instructions.jsonl", "w") as f:
            for val in tqdm(instructions):
                f.write(json.dumps(val)+"\n")

        instructions = load_dataset("json", data_files="instructions.jsonl")['train']

        return instructions

    def build_and_push(self, train_filename=None, valid_filename=None, test_filename=None, for_eval=False, limit=None, validation=None, subset=None):
        dataset_dict = {}
        for filename, split in zip([train_filename, valid_filename, test_filename], ["train", "valid", "test"]):
            posix = train_filename.split(".")[-1]
            read_method = READ_METHODS.get(posix, read_huggingface_repo)
                # print(test_data[0]['labels'].strip('\n').split(';'))
            original_split = "validation" if validation and split == "valid" else split
            original_split = "validation" if validation and split == "test" else split
            dataset_dict[split] = self.construct_instructions(read_method(filename, subset, original_split), for_eval, limit)
            #break
        #train_data = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information", 'train')
        #dataset_dict["valid"] = self.construct_instructions(read_method(train_filename), for_eval, limit)
                #print(train_data['train'])
        #dataset_dict["test"] = self.construct_instructions(read_method(train_filename), for_eval, limit)
                #if filename:
        #dataset_dict["train"] = self.construct_instructions(load_dataset("medalpaca/medical_meadow_wikidoc", split="train"), for_eval, limit)
        #dataset_dict["valid"] = self.construct_instructions(load_dataset("allenai/mslr2022", "ms2", split="validation"), for_eval, limit)
        #dataset_dict["test"] = self.construct_instructions(load_dataset("allenai/mslr2022", "ms2", split="test"), for_eval, limit)
        #print(dataset_dict)
        dataset_dict = DatasetDict(dataset_dict)
        dataset_dict.push_to_hub(f"{self.BASE_REPO}/{self.dataset}_{'train' if not for_eval else 'test'}")


class MedNLI(InstructionDataset):
    dataset = "MedNLI" 
    task_type = "classification"
    choices = ["entailment", "contradiction", "neutral"]
    prompt = """
TASK: Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral. Return only the label.
###
INPUT: {text}
OUTPUT:
"""

    def fetch_data(self, datum):
        return {
            "text": "[PRE] "+datum["sentence1"]+" [HYP] "+datum["sentence2"],
            "answer": datum["gold_label"],
        }


class EmrQA(InstructionDataset):
    dataset = "EmrQA" 
    task_type = "abstractivesummarization"
    #choices = ["entailment", "contradiction", "neutral"]
    prompt = """
TASK: Given a medical context and an open-ended question related to it, extract the relevant text segment from the context as an answer. Expected output: Only extract and return the text segment from the provided context that directly answers the question. Do not add any new words.
###
Context: {text}
Question {question}
Answer:
"""

    def fetch_data(self, datum):
        match = re.search(r"Context:(.*?)\nQuestion:", datum["query"], re.DOTALL)
        match2 = re.search(r"Question:(.*?)\nAnswer:", datum["query"], re.DOTALL)
        #print(datum["query"])
        return {
            "text": match.group(1).strip(),
            "question": match2.group(1).strip(),
            "answer": datum["answer"],
        }

class MTSample(InstructionDataset):
    dataset = "MTSample" 
    task_type = "classification"
    choices = [ 'Surgery', 'Allergy / Immunology', 'Sleep Medicine', 'Pediatrics - Neonatal', 'SOAP / Chart / Progress Notes', 'Bariatrics', 'Pain Management', 'Lab Medicine - Pathology', 'Dermatology', 'Orthopedic', 'Dentistry', 'Psychiatry / Psychology', 'General Medicine', 'Office Notes', 'Letters', 'Neurosurgery', 'Radiology', 'Cosmetic / Plastic Surgery', 'Nephrology', 'Diets and Nutritions', 'Chiropractic', 'Gastroenterology', 'Cardiovascular / Pulmonary', 'Speech - Language', 'Hospice - Palliative Care', 'Autopsy', 'Endocrinology', 'Emergency Room Reports', 'Discharge Summary', 'ENT - Otolaryngology', 'Urology', 'Physical Medicine - Rehab', 'Neurology', 'Podiatry', 'Ophthalmology', 'Rheumatology', 'IME-QME-Work Comp etc.', 'Hematology - Oncology', 'Consult - History and Phy.', 'Obstetrics / Gynecology']
    prompt = """
TASK: The task is to determine the medical specialty or domain that a medical transcription belongs to. The input is a medical transcription. There are 40 medical specialties or domains and you need to decide which one is the transcription related to. The medical specialties or domains are: 'Surgery', 'Allergy / Immunology', 'Sleep Medicine', 'Pediatrics - Neonatal', 'SOAP / Chart / Progress Notes', 'Bariatrics', 'Pain Management', 'Lab Medicine - Pathology', 'Dermatology', 'Orthopedic', 'Dentistry', 'Psychiatry / Psychology', 'General Medicine', 'Office Notes', 'Letters', 'Neurosurgery', 'Radiology', 'Cosmetic / Plastic Surgery', 'Nephrology', 'Diets and Nutritions', 'Chiropractic', 'Gastroenterology', 'Cardiovascular / Pulmonary', 'Speech - Language', 'Hospice - Palliative Care', 'Autopsy', 'Endocrinology', 'Emergency Room Reports', 'Discharge Summary', 'ENT - Otolaryngology', 'Urology', 'Physical Medicine - Rehab', 'Neurology', 'Podiatry', 'Ophthalmology', 'Rheumatology', 'IME-QME-Work Comp etc.', 'Hematology - Oncology', 'Consult - History and Phy.', 'Obstetrics / Gynecology'. The output should be only one medical specialty or domain from the above 40 specialties or domains, that is most relevant to the medical transcription. Please note that each medical transcript can only be related to one medical specialty or domain. Output format: provide the name of the medical specialty or domain.
###
INPUT: {text}
OUTPUT:
"""

    def fetch_data(self, datum):
        #print(datum["query"])
        match = re.search(r"INPUT:(.*?)\nOUTPUT:", datum["query"], re.DOTALL)
        #extracted_text = match.group(1).strip()
        #print(extracted_text)
        return {
            "text": match.group(1).strip(),
            "answer": datum["answer"],
        }

class PubmedQA(InstructionDataset):
    dataset = "PubmedQA" 
    task_type = "classification"
    choices = ["yes", "no", "maybe"]
    prompt = """As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe.
{context}
{text}
The answer is:
"""

    def fetch_data(self, datum):
        return {
            "text": datum["QUESTION"], "context": '\n'.join(ast.literal_eval(datum["CONTEXTS"])),
            "answer": datum["final_decision"],
        }


class MedQA(InstructionDataset):
    dataset = "MedQA" 
    task_type = "classification"
    choices = ["A", "B", "C", "D", "E"]
    prompt = """You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.
Question: {text}

Options:
{options}

The answer is
"""

    def fetch_data(self, datum):
        question = datum["question"]
        if not question.endswith('?') and not question.endswith('.'):
            question += '?'
        return {
            "text": question, "options": '\n'.join([item["key"]+'. '+item["value"] for item in datum['options']]),
            "answer": datum["answer_idx"],
        }


class MedMCQA(InstructionDataset):
    dataset = "MedMCQA" 
    task_type = "classification"
    choices = ["A", "B", "C", "D"]
    prompt = """You are a medical doctor answering realworld medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple-choice question. Select one correct answer from A to D. Base your answer on the current and standard practices referenced in medical guidelines.
Question: {text}

Options:
{options}

The answer is
"""

    def fetch_data(self, datum):
        question = datum["question"]
        if not question.endswith('?') and not question.endswith('.'):
            question += '?'
        return {
            "text": question,
            "options": '\n'.join([f"A. {datum['opa']}", f"B. {datum['opb']}", f"C. {datum['opc']}", f"D. {datum['opd']}"]),
            "answer": self.choices[datum['cop']],
        }


class MMLU(InstructionDataset):
    dataset = "MMLU" 
    task_type = "classification"
    choices = ["0", "1", "2", "3"]
    prompt = """Your task is to analyze a multiple-choice question on a given topic. You will be presented with a specific question. Accompanying the question will be a list of four choices. Your goal is to read the question carefully and understand its context and requirements. Identify the one choice that best answers the question, based on your understanding and analysis.
Question: {text}
Choices: {options}.
Please respond with the index of the identified choice in the list: "0", "1", "2", or "3" only.
Answer:
"""

    def fetch_data(self, datum):
        return {
            "text": datum["question"], "options": datum["choices"],
            "answer": str(datum["answer"]),
            }

class DDI2013(InstructionDataset):
    dataset = "DDI2013" 
    task_type = "classification"
    choices = ["DDI-effect", "DDI-mechanism", "DDI-advise", "DDI-false", "DDI-int"]
    prompt = """TASK: The input is a sentence where the drug is labeled as @DRUG$. Extract the relationship between 2 @DRUG$ from the input sentence by selecting only one of the following options: 'DDI-effect', 'DDI-mechanism', 'DDI-advise', 'DDI-false', and 'DDI-int'.
DDI-effect, this type is used to annotate DDIs describing an effect or a pharmacodynamic mechanism
DDI-mechanism, this type is used to annotate DDIs that are described by their pharmacokinetic mechanism
DDI-advise, this type is used when a recommendation or advice regarding a drug interaction is given
DDI-false, this type is used when no DDI relation appears
DDI-int, this type is used when a DDI appears in the text without providing any additional information
###
INPUT: {text}
OUTPUT:
"""

    def fetch_data(self, datum):
        return {
            "text": datum["sentence"],
            "answer": datum["label"],
        }

class HoC(InstructionDataset):
    dataset = "HoC"
    task_type = "multiclassification"
    choices = ["sustaining proliferative signaling", "evading growth suppressors", "resisting cell death", "enabling replicative immortality", "inducing angiogenesis", "activating invasion and metastasis", "genomic instability and mutation", "tumor promoting inflammation", "cellular energetics", "avoiding immune destruction"]
    label_dict = {
        "sustaining proliferative signaling": 0,
        "evading growth suppressors": 0,
        "resisting cell death": 0,
        "enabling replicative immortality": 0,
        "inducing angiogenesis": 0,
        "activating invasion and metastasis": 0,
        "genomic instability and mutation": 0,
        "tumor promoting inflammation": 0,
        "cellular energetics": 0,
        "avoiding immune destruction": 0
    }
    prompt = """The task is to decide the Hallmarks of Cancer (HOC) taxonomy of the article based on its abstract. The input is an abstract text. There are 10 topics you will need to decide whether the article is related to. Topics: sustaining proliferative signaling, evading growth suppressors, resisting cell death, enabling replicative immortality, inducing angiogenesis, activating invasion and metastasis, genomic instability and mutation, tumor promoting inflammation, cellular energetics, avoiding immune destruction. The output should be topics from the above 10 topics, that are related to the input article. Please note one article can be related to multiple topics. Output format: provide a semicolon-separated list of relevant topics.
    ###
    INPUT: {text}
    OUTPUT:
"""
    def fetch_data(self, datum):
        label_dict = dict(self.label_dict)
        input_labels = datum["labels"].strip('\n').split(';')
        for each_label in input_labels:
            each_label =  str(each_label)
            if each_label in label_dict:
                label_dict[each_label] = 1
        return {
                 "text": datum["text"],
                 "answer": datum["labels"],
        }

class CochranePLS(InstructionDataset):
    dataset = "CochranePLS"
    task_type = "abstractivesummarization"
    prompt = """***TASK***
the task is to simplify the input abstract of a biomedical literature

***INPUT***
the input is the abstract of a biomedical literature

***OUTPUT***
the output is the simplified abstract for the input abstract of a biomedical literature

***DOCUMENTATION***

***EXAMPLES***

Input: {text}
Output:
"""
    def truncate_string(self, input_string, max_tokens=3500):
        # Load the Llama tokenizer
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        # Tokenize the input string
        tokens = tokenizer.encode(input_string, truncation=True, max_length=max_tokens)

        # Decode the tokens back to a string
        truncated_string = tokenizer.decode(tokens, skip_special_tokens=True)

        return truncated_string


    def fetch_data(self, datum):
        return {
            "text": self.truncate_string(datum["src"]) if len(datum["src"]) >= 3000 else datum["src"],
            "answer": datum["tgt"],
        }



class PLOS(InstructionDataset):
    dataset = "PLOS"
    task_type = "abstractivesummarization"
    prompt = """***TASK***
the task is to simplify the input abstract of a biomedical literature

***INPUT***
the input is the abstract of a biomedical literature

***OUTPUT***
the output is the simplified abstract for the input abstract of a biomedical literature

***DOCUMENTATION***

***EXAMPLES***

Input: {text}
Output:
"""
    def truncate_string(self, input_string, max_tokens=3500):
        # Load the Llama tokenizer
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        # Tokenize the input string
        tokens = tokenizer.encode(input_string, truncation=True, max_length=max_tokens)

        # Decode the tokens back to a string
        truncated_string = tokenizer.decode(tokens, skip_special_tokens=True)

        return truncated_string


    def fetch_data(self, datum):
        return {
            "text": self.truncate_string(datum["abstract"]) if len(datum["abstract"]) >= 3000 else datum["abstract"],
            "answer": datum["plain language summary"],
        }


class PubmedSumm(InstructionDataset):
    dataset = "PubmedSumm" 
    task_type = "abstractivesummarization"
    prompt = """***TASK***
the task is to summarize an input biomedical literature in six sentences

***INPUT***
the input is a biomedical literature

***OUTPUT***
the output is the summary of an input biomedical literature in six sentences

***DOCUMENTATION***

***EXAMPLES***

Input: {text}
Output:
"""

    def fetch_data(self, datum):
        return {
            "text": datum["src"],
            "answer": datum["tgt"],
        }


class Alpaca(InstructionDataset):
     dataset = "Alpaca"
     task_type = "abstractivesummarization"
     prompt = """Below is an input that describes a task, maybe paired with a context that provides further information. Write a response that appropriately completes the request.
INPUT: {text} 
CONTEXT: {context}
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["instruction"], "context": datum["input"],
                 "answer": datum["output"],
        }

class medical_meadow_advice(InstructionDataset):
     dataset = "medical_meadow_advice"
     task_type = "abstractivesummarization"
     prompt = """For the following statement, determine whether it offers: 1) Strong Advice: The statement gives a clear and assertive recommendation or guidance, or 2) Weak Advice: The statement provides a suggestion or mild recommendation but is not very assertive, or 3)  No Advice: The statement doesn't offer any recommendation or guidance.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }

class medical_meadow_mediqa(InstructionDataset):
     dataset = "medical_meadow_mediqa"
     task_type = "abstractivesummarization"
     prompt = """Answer the input medical question based on the given context.
INPUT: {text}
CONTEXT: {context}
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["instruction"],
                 "context": datum["input"],
                 "answer": datum["output"],
        }


class medical_meadow_advice(InstructionDataset):
     dataset = "medical_meadow_advice"
     task_type = "abstractivesummarization"
     prompt = """For the following statement, determine whether it offers: 1) Strong Advice: The statement gives a clear and assertive recommendation or guidance, or 2) Weak Advice: The statement provides a suggestion or mild recommendation but is not very assertive, or 3)  No Advice: The statement doesn't offer any recommendation or guidance.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }

class medical_meadow_mediqa(InstructionDataset):
     dataset = "medical_meadow_mediqa"
     task_type = "abstractivesummarization"
     prompt = """Answer the input medical question based on the given context.
INPUT: {text}
CONTEXT: {context}
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["instruction"],
                 "context": datum["input"],
                 "answer": datum["output"],
        }


class medicationqa(InstructionDataset):
     dataset = "medicationqa"
     task_type = "abstractivesummarization"
     prompt = """Answer this medical question truthfully.
INPUT: {text}
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["Question"],
                 "answer": datum["Answer"],
        }

class liveqa(InstructionDataset):
     dataset = "liveqa"
     task_type = "abstractivesummarization"
     prompt = """Given a medical query, provide a concise and clear answer based on the given details. 
INPUT: {text}
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["message"],
                 "answer": datum["answer"],
        }

class iCliniq(InstructionDataset):
     dataset = "iCliniq"
     task_type = "abstractivesummarization"
     prompt = """Given a medical query, provide a concise and clear answer based on the patient's description.
INPUT: {text}
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["answer_chatdoctor"],
        }


class HealthCareMagic(InstructionDataset):
     dataset = "HealthCareMagic"
     task_type = "abstractivesummarization"
     prompt = """Given a medical query, provide a concise and clear answer based on the patient's description.
INPUT: {text}
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }

class GuidelineQA(InstructionDataset):
     dataset = "GuidelineQA"
     task_type = "abstractivesummarization"
     prompt = """If you are a medical professional, answer this question truthfully.
INPUT: {text}
OUTPUT:
"""
     def fetch_data(self, datum):
         response = datum["response"]
         question_match = re.search(r"Question:\s*(.+?)\s*(?=\nAnswer:)", response, re.DOTALL)
         answer_match = re.search(r"Answer:\s*(.+)", response, re.DOTALL)
         question = question_match.group(1).strip() if question_match else None
         answer = answer_match.group(1).strip() if answer_match else None 
         if not answer: 
             print(response)
             return None
         else:
             return {
                 "text": question,
                 "answer": answer,
                 }


class medical_meadow(InstructionDataset):
     dataset = "medical_meadow"
     task_type = "abstractivesummarization"
     prompt = """If you are a medical professional, answer this question truthfully.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }

class medical_wikidoc(InstructionDataset):
     dataset = "medical_wikidoc"
     task_type = "abstractivesummarization"
     prompt = """If you are a medical professional, answer this question truthfully.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }

class PatientQA(InstructionDataset):
     dataset = "PatientQA"
     task_type = "abstractivesummarization"
     prompt = """Answer this question truthfully.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }
class medical_meadow_mmmlu(InstructionDataset):
     dataset = "medical_meadow_mmmlu"
     task_type = "abstractivesummarization"
     prompt = """Given a medical question and four options, select the correct answer from the four options.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }



class guideline_prediction(InstructionDataset):
     dataset = "guideline_prediction"
     task_type = "abstractivesummarization"
     prompt = """Write the next part for a clinical guideline. You're given a piece of the guideline, and your task is to continue it. The new part should match the style and detail of the original and be medically correct.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         sentences  =  datum['clean_text'].split(". ")
         split_index = len(sentences) // 2
         first_half = '. '.join(sentences[:split_index]) + '.'
         remaining_half = '. '.join(sentences[split_index:])
         return {
                 "text": first_half,
                 "answer": remaining_half,
        }

class UMLS(InstructionDataset):
     dataset = "UMLS"
     task_type = "abstractivesummarization"
     prompt = """Given a medical question, provide the answer to determine the relation between two medical terms in the question.
INPUT: {text} 
OUTPUT:
"""
     def fetch_data(self, datum):
         #if datum["source"] == "umls_relation": 
         return {
                 "text": datum["input"],
                 "answer": datum["output"],
        }


class Dolly(InstructionDataset):
    dataset = "Dolly"
    task_type = "abstractivesummarization"
    prompt = """Below is an input that describes a task, maybe paired with a context that provides further information. Write a response that appropriately completes the request.
    INPUT: {text} 
    CONTEXT: {context}
    OUTPUT:  
"""  
    def fetch_data(self, datum):
        return {
                "text": datum["instruction"], "context": datum["context"],
                "answer": datum["response"],
        }


class MedInstruct(InstructionDataset):
    dataset = "MedInstruct"
    task_type = "abstractivesummarization"
    prompt = """Below is an input that describes a medical task, maybe paired with a context that provides further input information. Write a response that appropriately completes the request.
    INPUT: {text} 
    CONTEXT: {context}
    OUTPUT:  
"""  
    def fetch_data(self, datum):
        print(datum)
        return {
                "text": datum["instruction"], "context": datum["input"],
                "answer": datum["output"],
        }

class MIMIC_SUM(InstructionDataset):
    dataset = "MIMIC_SUM"
    task_type = "abstractivesummarization"
    prompt = """Derive the impression from findings in the radiology report.
    INPUT: {text}
    OUTPUT:
    """
    def fetch_data(self, datum):
        return {
                "text": ' '.join(datum["findings"]),
                "answer": ' '.join(datum["impression"]),
        }

class PubMed_Summ(InstructionDataset):
    dataset = "PubMed_Summ"
    task_type = "abstractivesummarization"
    prompt = """Given an abstract of a biomedical paper, generate the title.
    INPUT: {text}
    OUTPUT:
    """
    def fetch_data(self, datum):
        return {
                "text": datum['paper']['abstract'],
                "answer": datum['paper']['title'],
        }

class mesh_term(InstructionDataset):
    dataset = "mesh_term"
    task_type = "abstractivesummarization"
    prompt = """Given an abstract of a biomedical paper, generate the MeSH terms associated with the paper.
    INPUT: {text}
    OUTPUT:
    """
    def fetch_data(self, datum):
        return {
                "text": datum['paper']['abstract'],
                "answer": datum['paper']['mesh_terms'],
        }

class MS2(InstructionDataset):
    dataset = "MS2" 
    task_type = "abstractivesummarization"
    prompt = """***TASK*** the task is to summarize an input biomedical literature in six sentences 
    ***INPUT*** the input is a biomedical literature 
    ***OUTPUT*** the output is the summary of an input biomedical literature in six sentences 
    ***DOCUMENTATION*** 
    ***EXAMPLES*** 
    Input: {text} 
    Output:
    """
    def truncate_string(self, input_string, max_tokens=3500):
        # Load the Llama tokenizer
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        # Tokenize the input string
        tokens = tokenizer.encode(input_string, truncation=True, max_length=max_tokens)

        # Decode the tokens back to a string
        truncated_string = tokenizer.decode(tokens, skip_special_tokens=True)

        return truncated_string


    def fetch_data(self, datum):
        abstract = " ".join(datum["abstract"])
        return {
                "text": self.truncate_string(abstract) if len(abstract) >= 4000 else abstract,
                "answer": datum["target"],
                }


DATASETS = {
    "MedNLI": MedNLI,
    "PubmedQA": PubmedQA,
    "MedQA": MedQA,
    "MedMCQA": MedMCQA,
    "DDI2013": DDI2013,
    "PubmedSumm": PubmedSumm,
    "Alpaca": Alpaca,
    "Dolly": Dolly,
    "MIMIC_SUM": MIMIC_SUM,
    "HoC": HoC,
    "medical_meadow": medical_meadow,
    "medical_meadow_advice": medical_meadow_advice,
    "medical_meadow_mediqa": medical_meadow_mediqa,
    "medical_meadow_mmmlu": medical_meadow_mmmlu,
    "iCliniq": iCliniq,
    "HealthCareMagic": HealthCareMagic,
    "medicationqa": medicationqa,
    "liveqa": liveqa,
    "UMLS": UMLS,
    "PubMed_Summ":PubMed_Summ,
    "guideline_prediction": guideline_prediction,
    "MedInstruct": MedInstruct,
    "PatientQA": PatientQA,
    "MTSample": MTSample,
    "EmrQA": EmrQA,
    "MS2": MS2,
    "PLOS": PLOS,
    "CochranePLS": CochranePLS,
    "medical_wikidoc": medical_wikidoc,
}


def main():
    parser = argparse.ArgumentParser(description='Process dataset for hippocrates training and evaluation.')

    parser.add_argument('-dataset', type=str, required=True, help='The dataset name')
    parser.add_argument('-train_filename', type=str, required=True, help='The training dataset filename')
    parser.add_argument('-valid_filename', type=str, required=True, help='The validation dataset filename')
    parser.add_argument('-test_filename', type=str, required=True, help='The test dataset filename')
    parser.add_argument('-for_eval', action='store_true', help='Set to true for evaluation, false otherwise')
    parser.add_argument('-limit', type=int, help='The generation number of the dataset')
    parser.add_argument('-subset', type=str, help='The subset of the dataset')

    args = parser.parse_args()

    dataset = args.dataset
    train_filename = args.train_filename
    valid_filename = args.valid_filename
    test_filename = args.test_filename
    for_eval = args.for_eval
    limit = args.limit
    subset = args.subset

    # Your processing code here
    data_class = DATASETS[dataset]()
    data_class.build_and_push(train_filename=train_filename, valid_filename=valid_filename, test_filename=test_filename, for_eval=for_eval, limit=limit, validation=True, subset=subset)


if __name__ == '__main__':
    main()
