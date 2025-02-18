
import librosa
import os
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

SAMPLE_RATE = 44100
DATA_PATH_Gita = '/home/jdariasl/Documents/Experiments/Experiments_EmiroIbarra/Parkinson_datasets/Gita/Pataka/'
DATA_PATH_NeuroV = '/home/jdariasl/Documents/Experiments/Experiments_EmiroIbarra/Parkinson_datasets/Neurovoz/PorMaterial_limpios1_2_downsized/PATAKA/'

class Pataka_Dataset(Dataset):
    """
    Code for reading the PD datasets
    """

    def __init__(self, 
                 DBs = ['Gita'], #Options are ['Gita','Neurovoz']
                 train_size=0.8,
                 mode = 'train', #Options are ['train','test']
                 seed=1234
                 ):
        self.seed = seed
        self.train_size = train_size
        self.DBs = DBs
        self.mode = mode
        self.paths, self.labels, self.speaker_ids = self.read_data()
        self.signals, self.y_label, self.subject_group = self.process_select_signals(SAMPLE_RATE)
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, index):
        spec = compute_norm_spect(self.signals[index], SAMPLE_RATE)
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(self.y_label[index], dtype=torch.float32), torch.tensor(self.subject_group[index], dtype=torch.float32)


    def read_data(self):
        paths = []
        labels = []
        speaker_ids = []
        for DB in self.DBs:
            if DB == 'Gita':
                paths_Gita, labels_Gita, speaker_ids_Gita = Read_Gita_DB(DATA_PATH_Gita)
                paths.extend(paths_Gita)
                labels.extend(labels_Gita)
                speaker_ids.extend(speaker_ids_Gita)
            elif DB == 'Neurovoz':
                Nspeakers = len(speaker_ids)
                paths_NeuroV, labels_NeuroV, speaker_ids_NeuroV = Read_NeuroVoz_DB(DATA_PATH_NeuroV,Nspeakers)
                paths.extend(paths_NeuroV)
                labels.extend(labels_NeuroV)
                speaker_ids.extend(speaker_ids_NeuroV)
        return paths, labels, speaker_ids

        #selecting 400ms overlap in 50ms of audio signal example
    def process_select_signals(self, SAMPLE_RATE):
        time_leng=0.4
        sample_leng=int(time_leng*SAMPLE_RATE)
        overloap=2
        signals, y_label, subject_group, tono =[],[],[],[]


        #Processs data to train
        for data_ind, file_path in enumerate(self.paths):
            audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
            audio_len=len(audio)
            #Normalize audio
            audio=audio/np.max(abs(audio))
            indx=[i for i,x in enumerate(np.sqrt(abs(audio))) if x>.30]
            segments=0
            if (indx[0]+sample_leng)<audio_len:
                for i in range(int((-indx[0]+indx[len(indx)-1])/(sample_leng/overloap))):
                    ind_start = i * int(sample_leng/overloap)+indx[0]
                    ind_end = ind_start + sample_leng
                    if ind_end <= indx[len(indx)-1]:
                        signal=np.zeros(sample_leng)
                        signal = audio[ind_start:int(ind_end)]
                        signals.append(signal)
                
                        y_label.append(self.labels[data_ind])
                        subject_group.append(self.speaker_ids[data_ind])
                        segments=segments+1
                print(" Processed {}/{} files".format(self.speaker_ids[data_ind],len(self.paths)-1),end='')
                print(" Time audio: {} Segments {} ".format((audio_len-1)/sample_rate,segments))
            else:
                print(" Processed {}/{} files".format(self.speaker_ids[data_ind],len(self.paths)-1),end='')
                print(" Time audio: {} Segments {} ".format((audio_len-1)/sample_rate,0))
        signals = np.stack(signals,axis=0)
        y_label = np.stack(y_label,axis=0)
        subject_group = np.stack(subject_group,axis=0)

        #Shuffle data
        np.random.seed(self.seed)
        if self.mode == 'train':
            indx = np.random.permutation(len(self.speaker_ids))[:int(self.train_size*len(self.speaker_ids))]
        else:
            indx = np.random.permutation(len(self.speaker_ids))[int(self.train_size*len(self.speaker_ids)):]
        
        subj_ind = np.isin(subject_group, indx).astype(int)
        signals = signals[subj_ind==1]
        y_label = y_label[subj_ind==1]
        subject_group = subject_group[subj_ind==1]
        return signals, y_label, subject_group
     
def Read_Gita_DB(DATA_PATH_Gita):
    Speaker_PD=0
    Speaker_HC=50
    Speaker_IDs = []
    Labels = []
    Paths = []
    for dirname, _, filenames in os.walk(DATA_PATH_Gita):
    
        for filename in filenames:
            file_path = os.path.join(dirname, filename)

            if dirname.find('PD')!=-1:
                Speaker_PD+=1
                Speaker_ID=Speaker_PD
                Label=1
            else: 
                Speaker_HC+=1
                Speaker_ID=Speaker_HC
                Label=0

            Speaker_IDs.append(Speaker_ID)
            Labels.append(Label)
            Paths.append(file_path)
    print("The number of files in Gita DB is {}".format(len(Paths)))
    return Paths, Labels, Speaker_IDs
    
def Read_NeuroVoz_DB(DATA_PATH_NeuroVoz,offset=0):
    Speaker_IDs = []
    Labels = []
    Paths = []
    for dirname, _, filenames in os.walk(DATA_PATH_NeuroV):
        for filename in filenames:
                file_path = os.path.join(dirname, filename)
                if filename.find('wav')!=-1:
                    identifiers=filename.split('.')[0].split('_')
                    Speaker_ID=int(identifiers[2]) + offset
                    if identifiers[0]=='PD':
                        Label=1
                    else:
                        Label=0
                    Speaker_IDs.append(Speaker_ID)
                    Labels.append(Label)
                    Paths.append(file_path)
    print("The number of files in Neurovoz DB is {}".format(len(Paths)))
    return Paths, Labels, Speaker_IDs

def compute_norm_spect(signals, sample_rate):
    n_fft = 2048
    win_length = int(0.015*sample_rate) 
    hop_length = int(0.010*sample_rate)
    n_mels = 65 

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )


    mel_spectrograms = []
    scaler = StandardScaler()
    print("Calculating mel spectrograms")
    for i in range(signals.shape[0]):
        mel_spect = librosa.power_to_db(mel_spectrogram(torch.from_numpy(signals[i,:])))
        mel_spect_norm=scaler.fit_transform(mel_spect)
        mel_spectrograms.append(mel_spect_norm)
        print("\r Processed {}/{} files".format(i,signals.shape[0]),end='')
    mel_spectrograms = np.stack(mel_spectrograms,axis=0)
    return mel_spectrograms

