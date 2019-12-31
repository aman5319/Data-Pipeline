import torch
import torchtext
from pathlib import Path
from .tokenizer import BaseTokenizer,SentencepieceTokenizer
from .fields import  GetFields
import pandas as pd
import numpy as np

class ClassifierData:

    class __ClassifierData:
        def __init__(self,tokenizer_func,some_unique_name,pad_first=False,include_lengths=False):
            if not  isinstance(tokenizer_func , BaseTokenizer):
                raise Exception("The Tokenizer class should be inherited from BaseTokenizer")
            self.tokenizer_func = tokenizer_func
            self.use_vocab = False if isinstance(self.tokenizer_func ,SentencepieceTokenizer) else True
            self.text_field  = GetFields.getTextField(self.tokenizer_func,self.use_vocab,pad_first=pad_first,include_lengths=include_lengths,preprocessing=False)
            self.class_field = GetFields.getClassificationField()
            self.reg_field = GetFields.getRegressionField()
            self.fields=None

        def generate_fields(self,input_file):
            if self.fields is None:
                df = pd.read_csv(input_file)
                self.regression_col = df.select_dtypes(np.float).columns.tolist()
                self.text_col = [df.apply(lambda x:x.memory_usage(deep=True,index=False)).sort_index(ascending=False).index[0]]
                self.classification_col = set(df.columns)-set(self.regression_col)-set(self.text_col)
                c =[*[(i,self.text_field) for i in self.text_col ],*[(i ,self.class_field) for i in self.classification_col],*[(i ,self.reg_field) for i in self.regression_col]]
                del df
                self.fields = {i[0]:i for i in c}
            
        def set_data(self,input_file,data_type):
            self.generate_fields(input_file)
            setattr(self,data_type+"_dataset" , torchtext.data.TabularDataset(str(input_file) , format="csv",fields=self.fields))
            return self

        def split_dataset(self,input_file,fraction= [7,2,1]):
            assert  sum(fraction)==10 , "fraction sum must be Ten"
            if getattr(self,"train_dataset",None) is None:
                self.set_data(input_file,"train")
            temp = torch.utils.data.random_split(getattr(self,"train_dataset"),self._get_fraction(fraction,getattr(self,"train_dataset").__len__()))
            for i,j in zip(["train","valid","test"],temp):
                setattr(self,i+"_dataset",j.dataset)
                if i =="train":
                    self.class_field.build_vocab(getattr(self,"train_dataset"))
                    if self.use_vocab:
                        self.text_field.build_vocab(getattr(self,"train_dataset"))
                print(f"{i} dataset is built")
            return self

        def build_iterators(self,batch_size=64,device=None): 
            for data_type in ["train","valid","test"]:
                temp = getattr(self,data_type+"_dataset",None)
                if temp is not None:
                    setattr(self , data_type+"_iterator", torchtext.data.BucketIterator(temp,
                                                                                        batch_size=batch_size,
                                                                                        device=device,
                                                                                        train=True if data_type == "train" else False,
                                                                                        sort_key = lambda x : len(x.text),
                                                                                        sort_within_batch = True,
                                                                                       shuffle=True))
                    print(f"{data_type} iterator built.")
            return self

        def _get_fraction(self,fraction,length):
            l =[]
            for i in fraction:
                l.append(int(i*length/10))
        
            if sum(l)==length:
                return l
            else:
                l[-1] += length -sum(l)
                return l

        def get_fields(self,which_one="text"):
            if which_one=="text":
                return self.text_field
            elif which_one =="classification" and self.classification_col.__len__()>0:
                return self.class_field
            elif which_one=="regression" and self.regression_col.__len__()>0:
                return self.reg_field
            else:
                print("The Field doesn't exist")

        def get_vocab_size(self,which_one="text"):
            if isinstance(self.tokenizer_func , SentencepieceTokenizer) and which_one=="text":
                return self.tokenizer_func.sp.get_piece_size()
            elif which_one=="text" or which_one =="classification":
                return self.get_field(which_one).vocab.__len__()
            else:
                 print("The Field doesn't exist")
    d={}
    def __new__(cls,*args,**kwargs):
        if kwargs["some_unique_name"] not in ClassifierData.d:
            ClassifierData.d[kwargs["some_unique_name"]] =ClassifierData.__ClassifierData(*args,**kwargs)
        return ClassifierData.d[kwargs["some_unique_name"]]

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)

    @classmethod
    def get_instance(cls,some_unique_name):
        """
        Return the instance of the class
        """
        return ClassifierData.d[some_unique_name]
