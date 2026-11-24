"""Final PI-STHNN softmax fusion; retain the trained common initialization."""
from base_model import PISTHNN as Base

class StudyModel(Base):
    def __init__(self,edge_index,edge_weight,method='paper_window',window=30):
        if method!='paper_window':raise ValueError('Only the final paper_window configuration is released')
        # Same Kaiming query initialization as the adopted paired experiment.
        super().__init__(edge_index,edge_weight,window=window,fusion='sigmoid_he')
        self.fusion='softmax_original'
        self.method=method
