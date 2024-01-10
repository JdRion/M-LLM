import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, load_peft_weights
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def peft_model_pipeline(MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
        ),
        device_map={"":0}
    )
    base_model.resize_token_embeddings(len(tokenizer))

    peft_model = PeftModel.from_pretrained(
        base_model,
        MODEL_ID,
        subfolder="loftq_init",
        is_trainable=True,
    )

    return peft_model, tokenizer
