from .tokenizer import SentencepieceTokenizer ,BaseTokenizer
import torchtext
import  torch

class GetFields:

    @staticmethod
    def getTextField(tokenizer_func ,use_vocab, pad_first =False , include_lengths=False,preprocessing=True):
        """
        Represents the TEXT field
        :args
                :tokenizer_func: the Tokenizer function to be used.
                :use_vocab:  True or False
                :pad_first: True or False
                :include_lengths: True or False
                :preprocessing: True or False
        """
        tokenizer_params = dict(tokenize = tokenizer_func.tokenize,
                                init_token = tokenizer_func.start_token,
                                eos_token = tokenizer_func.end_token,
                                pad_token = tokenizer_func.pad_token,
                                unk_token = tokenizer_func.unk_token,
                                batch_first=True,
                                pad_first=pad_first,
                                include_lengths=include_lengths,
                                use_vocab =  use_vocab,
                                preprocessing=( lambda s: [tokenizer_func.start_token,*s,tokenizer_func.end_token]) if preprocessing else None )
        return torchtext.data.Field(**tokenizer_params)
    
    @staticmethod
    def getClassificationField():
        """ The classification Label Field"""
        return torchtext.data.LabelField()

    @staticmethod
    def getRegressionField():
        """ The Regression Label field with dtype as float"""
        return torchtext.data.LabelField(use_vocab=False,dtype=torch.float64)


