import sentencepiece as spm
import spacy
from collections import defaultdict
from .utils.dir_utils import create_dir
from pathlib import  Path

__all__ = ["BaseTokenizer" , "SpaceSplitTokenizer" , "SpacyTokenizer" , "SentencepieceTokenizer"]


class BaseTokenizer:
    """The Base class for creating a tokenizer """
    def __init__(self,lang):
        self.lang = lang
    def tokenize(self, text):
        raise NotImplementedError("Implement the tokenizer")
    def detokenize(self,tokens):
        raise NotImplementedError("Implement the detokenizer")
    @property
    def start_token(self,):
        return "<s>"
    @property
    def end_token(self,):
        return "</s>"
    @property
    def unk_token(self,):
        return "<unk>"
    @property
    def pad_token(self,):
        return "<pad>"

class SpaceSplitTokenizer(BaseTokenizer):
    """The Space split tokenizer"""
    def __init__(self,lang):
        super().__init__(lang)
    def tokenize(self,text):
        return text.split(" ")
    def detokenize(self,tokens):
        return " ".join(tokens)

class SpacyTokenizer(BaseTokenizer):
    """The Spacy Tokenizer"""
    def __init__(self,lang):
        super().__init__(lang)
        self.nlp = spacy.load(self.lang)
    def tokenize(self,text):
        return [token.text for token in self.nlp.tokenizer(text)]
    def detokenize(self,tokens):
        return " ".join(tokens)

class SentencepieceTokenizer(BaseTokenizer):
    def __init__(self,config ):
        super().__init__(config.lang)
        self.sp  =  spm.SentencePieceProcessor()
        self.sp.load(str(config.output_path)+".model")
    def tokenize(self,text):
        return  self.sp.encode_as_ids(text)
    def detokenize(self,tokens):
        return self.sp.decode_ids(tokens)
    @property
    def pad_token(self,):
        return self.sp.get_piece_size()
    @property
    def start_token(self,):
        return self.sp.piece_to_id(super().start_token)
    @property
    def end_token(self,):
        return self.sp.piece_to_id(super().end_token)
    @property
    def unk_token(self,):
        return self.sp.piece_to_id(super().unk_token)
