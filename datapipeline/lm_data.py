import torch
import torchtext
from pathlib import Path
from .tokenizer import BaseTokenizer,SentencepieceTokenizer
from .fields import  GetFields
from .sentence_piece import  df_textcol_to_txt
import pandas as pd

__all__ = ["TrainLmData"]
class TrainLmData:
    """
        Implements Language Modelling Data pipeline in a singleton fashion. Give a unique name to instance of this class so that i can be refered any time back with the existing instance object.
        >>> config = sentence_piece.SentencePieceConfig(lang="en",input_path="hindi.txt",vocab_size=32000,model_type="bpe")
        >>> tokenizer_func = tokenizer.SentencepieceTokenizer(config)
        >>> dataset = lm_data.TrainLmData(tokenizer_func=tokenizer_func,some_unique_name="using_spm")
        >>> dataset.split_dataset("hindi.txt",(7,2,1))
        >>> dataset.build_iterators()
        >>> dataset.train_dataset ## will return the dataset.
    """
    class __TrainLmData:
        def __init__(self, tokenizer_func,some_unique_name,pad_first=False):
            if not  isinstance(tokenizer_func , BaseTokenizer):
                raise Exception("The Tokenizer class should be inherited from BaseTokenizer")
            self.tokenizer_func = tokenizer_func
            self.use_vocab = False if isinstance(self.tokenizer_func ,SentencepieceTokenizer) else True
            self.TEXT  = GetFields.getTextField(self.tokenizer_func,self.use_vocab,pad_first=pad_first)

        def set_data(self,input_file,data_type):
            """
                Method to be used for setting train, valid and test data.
                :args
                        :input_file: The input file should be in text format.
                        :data_type: Should be one among ["train","valid","test"]
            """
            assert data_type in ["train","valid","test"] , "data_type should be either train, valid or test"
            setattr(self,data_type+"_dataset" , torchtext.datasets.LanguageModelingDataset(str(input_file) ,self.TEXT, newline_eos=False))
            if self.use_vocab and data_type=="train":
                self.TEXT.build_vocab(getattr(self,"train_dataset",None))
            print(f"{data_type} dataset is built.")
            return self

        def split_dataset(self,input_file,fraction = (7,2,1)):
            """
            Splits a single data blob into train ,valid,test, according to the fraction.
            :args 
                    :input_file: Input data source in text format or a Dataframe.
                    :fraction: It should be a tuple and sum should be 10. The fraction order should be train, valid,test.
                               fraction = (8,2) signifies 80% train and 20% valid.
                               fraction = (7,2,1) signifies 70% train , 20% valid and 10% test.
            """
            assert  sum(fraction)==10 , "fraction sum must be Ten"
            if isinstance(input_file,pd.DataFrame):
                input_file = df_textcol_to_txt(input_file)
            with open(input_file) as f:
                temp_data = f.readlines()
                for i,data_type in zip(fraction,["train","valid","test"]):
                    tmp_path = Path(data_type+"_data_temp.txt")
                    count_to=0
                    count_from = int((i/10)*len(temp_data))
                    with open(tmp_path,"w") as temp_file:
                        temp_file.writelines(temp_data[count_to:count_from])
                    count_to = count_from
                    self.set_data(tmp_path ,data_type)
                    tmp_path.unlink()
            return self

        def get_field(self,):
            """
            Get the Text field.
            """
            return self.TEXT
        
        def get_vocab_size(self,):
            """
            Gets the vocab size.
            """
            if isinstance(self.tokenizer_func , SentencepieceTokenizer):
                return self.tokenizer_func.sp.get_piece_size()
            else:
                return self.get_field().vocab.__len__()

        def detokenize(self, tokens):
            """
            Detokenize function to convert int indices to string. tokens should have dimension either 2D or 1D.
            """
            assert not tokens.dim()>2 , "Dimension should be one or two"
            tokens = tokens.cpu().clone().detach().squeeze(0).tolist()
            if self.use_vocab: tokens = [self.TEXT.vocab.itos[i] for i in tokens]
            return self.tokenizer_func.detokenize(tokens)
        
        def build_iterators(self,batch_size=64 , bptt_len=70,device=None):
            """
            Build itertator according to available train, valid and test dataset.
            """
            for data_type in ["train","valid","test"]:
                temp = getattr(self,data_type+"_dataset",None)
                if temp is not None:
                    setattr(self , data_type+"_iterator", torchtext.data.BPTTIterator(temp,
                                                                                 batch_size=batch_size,
                                                                                 bptt_len=bptt_len,
                                                                                device=device,
                                                                                 train=True if data_type == "train" else False))
                    print(f"{data_type} iterator built.")
            return self
    d={}
    def __new__(cls,*args,**kwargs):
        if kwargs["some_unique_name"] not in TrainLmData.d:
            TrainLmData.d[kwargs["some_unique_name"]] =TrainLmData.__TrainLmData(*args,**kwargs)
        return TrainLmData.d[kwargs["some_unique_name"]]

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)

    @classmethod
    def get_instance(cls,some_unique_name):
        """
        Return the instance of the class
        """
        return TrainLmData.d[some_unique_name]
