import sentencepiece as spm
from .config_sentencepiece import SentencePieceConfig
import pandas as pd
from ..utils.dir_utils import create_dir
from  pathlib import Path

__all__ = ["df_textcol_to_txt" , "train_spm_model"]

def df_textcol_to_txt(df:pd.DataFrame,column ="text"):
    """Converts dataframe to text column"""
    path = create_dir(Path("sentencepiece"))/"input.txt"
    with open(path , "w") as f: 
        f.writelines([ i +" \n " for i in df[column]])
    print(f"The {column} of the dataframe has been written to sentencepiece/input.txt")
    return path

def train_spm_model(config:SentencePieceConfig): 
    """
        Training sentence piece model with a sentencepiececonfig.
        >>> config = sentence_piece.SentencePieceConfig(lang="en",input_path="hindi.txt",vocab_size=32000,model_type="bpe")
        >>> sentence_piece.train_spm_model(config)

    """
    spm.SentencePieceTrainer.train(str(config))
    return config


