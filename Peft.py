#import libraries and create global variables
from model import LoadModel 
from Dataset import LoadDataset
from datasets import load_dataset
import torch
from transformers import DataCollatorForSeq2Seq, TrainingArguments,Trainer,EarlyStoppingCallback

from peft import LoraConfig,get_peft_model
import bitsandbytes as bnb 


class Peft:
    def __init__(self):
        self.model=None
        self.tokenizer=None
        self.device=torch.device("mps" if torch.mps.is_available() else "cpu")
    def loadModel_func(self,model_path):
        model_loader=LoadModel()
        self.model , self.tokenizer=model_loader.load_model_and_tokenizer(model_path)

    def loadDataset_func(self,index,dataset_path):
        self.dataset=load_dataset(dataset_path)
        self.dialogue=self.dataset["test"][index]["dialogue"]
        return self.dialogue

    def tokenizer_func(self,example):
        start_prompt='Summarize the following conversation.\n\n'
        end_prompt="\n\nSummary:"
        prompt=[start_prompt+dialogue+end_prompt for dialogue in example["dialogue"]]
        example["input_ids"]=self.tokenizer(prompt,padding="max_length",max_length=128,truncation=True,return_tensors="pt").input_ids.to(self.device)
        example["labels"]=self.tokenizer(example["summary"],padding="max_length",max_length=128,truncation=True,return_tensors="pt").input_ids.to(self.device)

        return example

    def tokenized_datasets_func(self):
        self.tokenized_datasets=self.dataset.map(self.tokenizer_func,batched=True)
        self.tokenized_datasets=self.tokenized_datasets.remove_columns(["id","topic","dialogue","summary"])
        return self.tokenized_datasets

    def Data_collator_func(self):
        self.data_collator=DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        return self.data_collator
    

    # if you want use QLora you can use this finction and assign this to LoraConfig but now we use Lora and we dont assign this.

    # def build_target_modules(self):
    #     cls=bnb.nn.modules.Linear4bit
    #     lora_module_names=set()
    #     for name , module in self.model.named_modules():
    #         if isinstance(module , cls):
    #             names=name.split(".")

    #             lora_module_names.add(names[0] if len(names)==1 else names[-1])
    #     if "lm_head" in lora_module_names:
    #         lora_module_names.remove("lm_head")
    #     return list(lora_module_names)
    

    

    #start PEFT 
    def build_Lora_config(self):
        self.lora_config=LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q","v"],
            task_type="SEQ_2_SEQ_LM",
        )
        return self.lora_config

    
    def adapt_lora(self):
        self.peft_model=get_peft_model(
            self.model,
            self.lora_config 

        )
        return self.peft_model
    #از این لحظه مدل تبدیل به یک مدل LoRA شده است.
    
    def print_trainable_param(self):
        self.peft_model.print_trainable_parameters()

    

    def training_args_func(self,output_dir_path:str,per_device_train_batch_size:int,per_device_eval_batch_size:int,epochs:int,gradient_accumulation_steps:int,learning_rate:int,logging_steps:int,weight_decay:float):
            self.training_args=TrainingArguments(
            output_dir=output_dir_path,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=logging_steps,
            weight_decay=weight_decay,
            load_best_model_at_end=True ,
        )
            return self.training_args

    def Trainer_func(self):
        trainer=Trainer(
           model= self.peft_model,
           args=self.training_args,
           data_collator=self.data_collator,
           train_dataset=self.tokenized_datasets["train"],
           eval_dataset=self.tokenized_datasets["test"],
           callbacks=[EarlyStoppingCallback(early_stopping_patience=3,)],
           processing_class=self.tokenizer,

        )
        trainer.train()




    def run(self):
        self.loadModel_func(model_path="./model/flan-t5-base")
        self.loadDataset_func(dataset_path="./dataset/dialog_dataset",index=200)
        self.tokenized_datasets_func()
        self.Data_collator_func()
        self.build_Lora_config()
        self.adapt_lora()
        self.print_trainable_param()
        self.training_args_func(output_dir_path="./logs",per_device_train_batch_size=2,per_device_eval_batch_size=2,epochs=3,gradient_accumulation_steps=2,learning_rate=0.0002,logging_steps=50,weight_decay=0.01)
        self.Trainer_func()

if __name__=="__main__":
    peft=Peft()
    peft.run()
        
