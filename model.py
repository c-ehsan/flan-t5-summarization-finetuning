#import libraries or create variables
#To Download or  Load Tokenizer and Load model with configuration 
from transformers import AutoModelForSeq2SeqLM ,AutoConfig , GenerationConfig , AutoTokenizer 



class LoadModel:
    def __init__(self):
       self.model=None
       self.tokenizer=None
    def load_model_and_tokenizer(self,model_path):
        self.model=AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)
        return self.model , self.tokenizer
    def get_refrence_from_model(self,prompt):
        inputs=self.tokenizer(prompt,return_tensors="pt")
        outputs=self.tokenizer.decode(self.model.generate(**inputs,max_new_tokens=100)[0],skip_special_tokens=True)
                                      
        return outputs
    



        



