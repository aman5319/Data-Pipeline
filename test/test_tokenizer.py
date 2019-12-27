import pytest
import sys
print(sys.path)
from datapipeline import tokenizer
sampletext = "My name is Aman"

def test_space_split():

    s = tokenizer.tokenizer_dict("space")("en")
    assert s.tokenize(sampletext).__len__()==4 ,  "No length match"

