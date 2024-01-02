import os

import mne
from .sedf_physionet import Sedf_PhysioNet

class SEDF_SC(Sedf_PhysioNet):
    def dataset_name(self):
        return "sedf_sc"
    
    def read_psg(self, record):
        psg_path, hyp_path = record    
        
        x = dict()
        y = [] 
        
        # region x
        data = mne.io.read_raw_edf(psg_path, verbose=False)
        sample_rate = data.info["sfreq"]
            
        for channel in self.channel_mapping().keys():
            channel_data = data.get_data(channel)[0]
            chnl_len = len(channel_data)
            
            x[channel] = (channel_data, sample_rate)
        # endregion
        
        # region y
        hyp = mne.read_annotations(hyp_path)
        
        labels = list(hyp.description)
        labels.pop() # Removing last element as it contains unknown sleep stage due to missing signal.
        durations = list(hyp.duration)
        durations.pop()
        
        assert len(labels) == len(durations)
        
        for label, duration in zip(labels, durations):
            assert label != None
            assert duration != None
            
            dur_in_epochs = int(duration/30)
                    
            for e in range(dur_in_epochs):
                y.append(label)
      
        if len(y)*self.sample_rate()*30 != chnl_len: # TODO: Figure out why lengths don't match
            return None
        # endregion

        assert len(y)*self.sample_rate()*30 == chnl_len, "Length of signal does not match the number of labels."
        
        return x, y