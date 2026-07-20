#Reduction memmory for train model 
from peft import PeftModel,PeftConfig,LoraConfig , TaskType
# #To Evaluate model 
# import evaluate
# To choose device and etc...
import torch 
# To work with dataframes and bring data
import pandas as pd 
#To work with datas and normalize them
import numpy as np 
import os 

class LoadDataset:
    def __init__(self):
        self.dataset_path=".dataset/dialog_dataset/"
    def load_train_df(self):
        self.dataset_train=pd.read_csv(os.path.join(self.dataset_path,"train.csv"))
        return self.dataset_train
    def load_test_df(self):
        self.dataset_test=pd.read_csv(os.path.join(self.dataset_path,"test.csv"))
        return self.dataset_test
    def load_val_df(self):
        self.val_dataset=pd.read_csv(os.path.join(self.dataset_path,"test.csv"))
        return self.val_dataset


    


