import sentencepiece as spm
from .config_sentencepiece import SentencePieceConfig
import pandas as pd
from ..utils.dir_utils import create_dir
from  pathlib import Path

__all__ = ["df_textcol_to_txt" , "train_spm_model"]
def df_textcol_to_txt(df:pd.DataFrame,column ="text"):
    with open( create_dir(Path("sentencepiece"))/"input.txt", "w") as f: 
        f.writelines([ i +" \n " for i in df[column]])
    print(f"The {column} of the dataframe has been written to sentencepiece/input.txt")

def train_spm_model(config:SentencePieceConfig): 
    spm.SentencePieceTrainer.train(str(config))
    return config


