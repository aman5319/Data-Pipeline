from pathlib import Path
from ..utils.dir_utils import create_dir

__all__ = ["SentencePieceConfig"]
class SentencePieceConfig:
    def __init__(self,lang , train_string=None , **kwargs):
        self.lang = lang
        self.input_path = kwargs.pop("input_path" ,create_dir(Path("sentencepiece"))/"input.txt" )
        self.output_path = kwargs.pop("output_path" ,create_dir(Path("sentencepiece"))/"output" )
        self.vocab_size = kwargs.pop("vocab_size",8000)
        self.model_type = kwargs.pop("model_type","unigram")
        self.train_string = train_string
    def __str__(self,):
        if self.train_string is None:
            return  f"--input={str(self.input_path)} --model_prefix={self.output_path} --vocab_size={self.vocab_size} --model_type={self.model_type}"
        else:
            return self.train_string
