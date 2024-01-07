import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel, load_peft_weights

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, lora=False):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.lora = lora
        if vector is not None:
            self.vector = vector
        else:
            if self.lora:
                assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
                with torch.no_grad():
                    if "LoftQ" in pretrained_checkpoint:
                        pretrained_state_dict = load_peft_weights(pretrained_checkpoint, subfolder="loftq_init")
                    else:
                        pretrained_state_dict = load_peft_weights(pretrained_checkpoint)
                    finetuned_state_dict = load_peft_weights(finetuned_checkpoint)
                    self.vector = {}
                    for key in pretrained_state_dict:
                        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                            continue
                        self.vector[key] = (finetuned_state_dict[key] - pretrained_state_dict[key]) * 0.5   # co: 0.5
            else:
                assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
                with torch.no_grad():
                    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                    self.vector = {}
                    for key in pretrained_state_dict:
                        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                            continue
                        self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector, lora=True) if self.lora else TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

    def apply_to_lora(self, pretrained_checkpoint):
        """Apply a task vector with LoRA to a pretrained model"""
        if not self.lora:
            assert ValueError("TaskVector type is not LoRA")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint, torch_dtype=torch.float16)
        pretrained_model = PeftModel.from_pretrained(
            model,
            pretrained_checkpoint,
            subfolder="loftq_init",
            is_trainable=False,
        )
        model_state_dict = pretrained_model.state_dict()  

        for name, value in self.vector.items():
            name = name.replace(".weight", ".default.weight")
            if name in model_state_dict:
                model_state_dict[name] = value

        pretrained_model.load_state_dict(model_state_dict)
        return pretrained_model, tokenizer