import os

from .sedf_physionet import Sedf_PhysioNet
import mne

class SEDF_ST(Sedf_PhysioNet):
    def dataset_name(self):
        return "sedf_st"
    
    def read_psg(self, record):
        psg_path, hyp_path = record    
        
        x = dict()
        y = [] 
        
        # region x
        data = mne.io.read_raw_edf(psg_path, verbose=False)
        sample_rate = data.info["sfreq"]
        hyp = mne.read_annotations(hyp_path)

        onset = list(hyp.onset)
        durations = list(hyp.duration)
        
        start_time = onset[0] - data.first_time
        end_time = onset[-1] + durations[-1] - data.first_time
        
        # Code from MNE to avoid near-zero errors
        #https://github.com/mne-tools/mne-python/blob/maint/1.3/mne/io/base.py#L1311-L1340
        if -sample_rate / 2 < start_time < 0:
            start_time = 0

        try:
            data.crop(start_time, end_time, True)
        except ValueError:
            self.log_error("Could not crop data", subject=None, record=(psg_path, hyp_path))
            return None

        labels = list(hyp.description)
        
        y = []
        
        for label, duration in zip(labels, durations):
            assert label != None
            assert duration != None
            
            dur_in_epochs = int(duration/30)
                    
            for e in range(dur_in_epochs):
                y.append(label)

        label_len = int(len(y)*sample_rate*30)

        x = dict()
        
        for channel in self.channel_mapping().keys():
            channel_data = data.get_data(channel)[0]
            chnl_len = len(channel_data)
            
            if abs(chnl_len-label_len) > 2:
                self.log_info(f"Diff was {abs(chnl_len-label_len)}")
                return None
           
            x[channel] = (channel_data[0:label_len], sample_rate)
 
        return x,y
