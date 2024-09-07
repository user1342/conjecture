from abc import ABC, abstractmethod

class ExtractorBaseClass(ABC):
    
    def __init__(self, model, tokenizer) -> None:
        
        self._model = model
        self._tokenizer = tokenizer

        super().__init__()