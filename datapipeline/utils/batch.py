import torch

class Batches:
    def __init__(self,iterator,text_col,classification_col,regression_col):
        setattr(self,"text",getattr(iterator,text_col[0]))
        setattr(self,"classify" , torch.cat([getattr(iterator,i).unsqueeze(1) for i in classification_col],dim=1))
        if regression_col.__len__()>0:
            setattr(self,"regres", torch.cat([getattr(iterator,i).unsqueeze(1) for i in regression_col],dim=1))
    def __repr__(self,):return str(self)
    
    def __str__(self,):
        if isinstance(self.text,tuple):
            s = "("+" ".join([f"'[{i.dtype} {i.shape}]'" for i in self.text]) +")"
        else:
            s = f"[{self.text.dtype} {self.text.shape}]"
        e= f" The Batch size is {self.classify.size(-1)}\
        \n\t[.text]:{s}\
        \n\t[.classify][{self.classify.dtype} {self.classify.shape}]"
        e1 = f"\n\t[.regres][{self.regres.dtype} {self.regres.shape}]"  if getattr(self,"regres" ,None) is not None else ""
        return e+e1


        
class BatchWrapper:
    def __init__(self,iterator,text_col,classification_col,regression_col):
        self.iterator=iterator
        self.text_col = text_col
        self.classification_col = classification_col
        self.regression_col = regression_col
    def __iter__(self):
        for i in self.iterator:
            yield Batches(i,self.text_col,self.classification_col,self.regression_col)
            
