#import libraries and create global variables 

from model import LoadModel 
from Dataset import LoadDataset
from datasets import load_dataset
from typing import List,Tuple
import torch
from transformers import DataCollatorForSeq2Seq, TrainingArguments,Trainer

class SummerizationDemmo:
    def load_model(self,model_path):
        model_loader=LoadModel()
        self.model , self.tokenizer=model_loader.load_model_and_tokenizer(model_path)
        print("Loading Model...")
        self.device=torch.device("mps" if torch.mps.is_available() else "cpu")
    def load_dataset(self,model_path):
        self.data=LoadDataset()
        #load train_df
        data=LoadDataset()
        print("loading train_df....")
        self.data_train=data.load_train_df()
        #load test_df
        print("loading test_df....")
        self.data_test=data.load_test_df()
        #load test_df
        print("loading val_df....")
        self.data_val=data.load_val_df()
    #get refrence with model 
    def get_reference(self,exmaple_indices:List[int]):
        print("getting refrence from model...")
        dashline="-".join("" for _ in range(10))
        for i , index in enumerate(exmaple_indices):
            dialogue=self.data_test["dialogue"][index]
            summary=self.data_test["summary"][index]
            inputs=self.tokenizer(dialogue,return_tensors="pt")
            with torch.no_grad():
                output=self.tokenizer.decode(self.model.generate(**inputs,max_new_tokens=50)[0],skip_special_tokens=True)
                print(dashline)
                print("Exmaple",i+1)
                print(dashline)
                print(f"input Prompt:\n{dialogue}")
                print(dashline)
                print(f"Baseline human summary:\n{summary}")
                print(dashline)
                print(f"Model Generation - Without Prompt Enginnering:\n{output}")
                print()

if __name__=="__main__":
    sd=SummerizationDemmo()
    sd.load_model("./model/flan-t5-base")
    sd.load_dataset("./dataset/dialog_dataset")
    sd.get_reference([3,4])