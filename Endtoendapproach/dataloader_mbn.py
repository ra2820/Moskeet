from io import BytesIO
import numpy as np
from numpy.core.arrayprint import DatetimeFormat 
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import librosa
import datetime
import os
from scipy.signal import find_peaks
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder





IMEI = 0
dataPath =''
ProductID = 0
GPS_Latitude = 0
GPS_Longitude = 0
TimeStamp = 0
SamplingFrequency = 0
BitsPerSample = 0
NumOfSamples = 0
NumOfChannels = 0
ChData = []
Ch0Max = 0
Ch0Min = 4095
Ch1Max = 0
Ch1Min = 4095
MbnFilesPath = []
counter = 0	

def writeFile(filename, data):
	#make file
	file = open(filename, "wb")
	#write to file
	file.write(bytes(data))




def fillHeader():
	global SamplingFrequency
	global BitsPerSample
	global NumOfSamples
	#global NumOfChannels
	NumOfChannels = 1
	fileLength = int((BitsPerSample * NumOfChannels * NumOfSamples) / 8)
	
	wavHeader = bytearray()
	#RIFF 
	#offset 0, Len - 4
	wavHeader.append(ord('R'))
	wavHeader.append(ord('I'))
	wavHeader.append(ord('F'))
	wavHeader.append(ord('F'))
	
	#Filesize - 8 bytes 
	#offset 4, Len - 4
	var = (44 + fileLength - 8)
	wavHeader += (var.to_bytes(4, 'little'))
	
	#"WAVE"
	#offset 8, Len - 4
	wavHeader.append(ord('W'))
	wavHeader.append(ord('A'))
	wavHeader.append(ord('V'))
	wavHeader.append(ord('E'))
	
	#"fmt "
	#offset 12, Len - 4
	wavHeader.append(ord('f'))
	wavHeader.append(ord('m'))
	wavHeader.append(ord('t'))
	wavHeader.append(ord(' '))
	
	#Length of format data - 16
	#offset 16, Len - 4
	var = 16
	wavHeader += (var.to_bytes(4, 'little'))
	
	#Type of format (1 is for PCM)
	#offset 20, Len - 2
	var = 1
	wavHeader += (var.to_bytes(2, 'little'))
	
	#Number of Channels 
	#offset 22, Len - 2
	wavHeader += NumOfChannels.to_bytes(2, 'little')
	
	#Sample Rate
	#offset 24, Len - 4
	wavHeader += SamplingFrequency.to_bytes(4, 'little')
	
	#Byte Rate
	#offset 28, Len - 4
	var = ((SamplingFrequency * BitsPerSample * NumOfChannels) / 8)
	wavHeader += ((int(var)).to_bytes(4, 'little'))

	#Number of bytes in one Sample(including all channels)
	#offset 32, Len - 2
	var = ((BitsPerSample * NumOfChannels) / 8)
	wavHeader += ((int(var)).to_bytes(2, 'little'))
	
	#Bits per Sample
	#offset 34, Len - 2
	var = BitsPerSample
	wavHeader += (var.to_bytes(2, 'little'))
	
	#"data" chunk header
	#offset 36, Len - 4
	wavHeader.append(ord('d'))
	wavHeader.append(ord('a'))
	wavHeader.append(ord('t'))
	wavHeader.append(ord('a'))
	
	#Size of the data section
	#offset 40, Len - 4
	var = fileLength
	wavHeader += (var.to_bytes(4, 'little'))

	return wavHeader





class MBN(Dataset):
    """MBN dataset."""

    def __init__(self, csv_file_path):
        """
        Args:
            csv_file (string): Path to the csv file with train/val/test splits.
            
        """
        self.mbn_files_data = pd.read_csv(csv_file_path)
        species = self.mbn_files_data['Species']
        values = np.array(species)
        #print(values)
        # integer encode
        label_encoder = LabelEncoder()
        self.integer_encoded = label_encoder.fit_transform(values)
        #print(integer_encoded)
        
        

    def __len__(self):
        return len(self.mbn_files_data)
    
    

    def MbnFile(filename):
        waveFiles = []
        data = open(str(filename), "rb").read()

        global IMEI
        global ProductID
        global GPS_Latitude
        global GPS_Longitude
        global TimeStamp
        global SamplingFrequency
        global BitsPerSample
        global NumOfSamples
        global NumOfChannels
        global ChData
        global counter
        global dataPath

        IMEI = data[4:19]
        ProductID = data[19:83]
        GPS_Latitude = data[83:96]
        GPS_Longitude = data[97:112]
        #print('gps:',GPS_Latitude,GPS_Longitude)
        

        #TimeStamp = datetime.datetime((((data[116] - 48) * 1000) + ((data[117] - 48) * 100) + ((data[118] - 48) * 10) + ((data[119] - 48))), (((data[112] - 48) * 10) + ((data[113] - 48))), ((data[114] - 48) * 10) + ((data[115] - 48)), ((data[120] - 48) * 10) + ((data[121] - 48)), ((data[122] - 48) * 10) + ((data[123] - 48)), 0)
        SamplingFrequency = (data[125] << 8) | (data[124])
        BitsPerSample = data[126]
        NumOfSamples = (data[128] << 8) | (data[127])
        #print('samples',NumOfSamples)
        NumOfChannels = data[129]
        #print('channels',NumOfChannels)
        ChData = []
        Ch0Max = 0
        Ch1Max = 0
        Ch0Min = 4095
        Ch1Min = 4095
        surgeCount0 = 0
        surgeCount1 = 0
        if BitsPerSample == 0: 
            print(filename)



        for ch in range(0, NumOfChannels):
            ChData.append([])
            waveFiles.append(bytearray())
            waveFiles[ch] += fillHeader()
        for offset in range(512, ((NumOfChannels * NumOfSamples * int(BitsPerSample / 8)) + 512), (int(BitsPerSample / 8)) * (NumOfChannels)):
            for ch in range(0, NumOfChannels):
                val = ((ch * int(BitsPerSample / 8)) + offset)
            
                waveFiles[ch].append(data[val])
                waveFiles[ch].append(data[val + 1])
                ChData[ch].append((data[val + 1] << 8) | (data[val]))
                if(ch == 0):
                    if(Ch0Max < (data[val + 1] << 8) | (data[val])):
                        Ch0Max = (data[val + 1] << 8) | (data[val])
                    if(Ch0Min > (data[val + 1] << 8) | (data[val])):
                        Ch0Min = (data[val + 1] << 8) | (data[val])
                    if(((data[val + 1] << 8) | (data[val])) > 1567): #1577
                
                        surgeCount0 = surgeCount0 + 1
                elif(ch == 1):
                    if(Ch1Max < (data[val + 1] << 8) | (data[val])):
                        Ch1Max = (data[val + 1] << 8) | (data[val])
                    if(Ch1Min > (data[val + 1] << 8) | (data[val])):
                        Ch1Min = (data[val + 1] << 8) | (data[val])
                    if(((data[val + 1] << 8) | (data[val])) > 1567): #1553
            
                        surgeCount1 = surgeCount1 + 1
                

        #print("Filename: {} \t Number of Channels: {}".format(filename, NumOfChannels))

        wavFileName = bytearray(filename, 'utf-8')

        wavFileName[len(wavFileName) - 4] = ord('_')
        wavFileName[len(wavFileName) - 3] = ord('C')
        wavFileName[len(wavFileName) - 2] = ord('h')
        wavFileName[len(wavFileName) - 1] = ord('0')
        wavFileName.append(ord('.'))
        wavFileName.append(ord('w'))
        wavFileName.append(ord('a'))
        wavFileName.append(ord('v'))


        from scipy.fftpack import fft
        fftData = []
        Filter_LowerCutoff_Hz = 300
        Filter_UpperCutoff_Hz = 1000
        goingUp = 0
        downCnt = 0
        InflectionPoints = []
        MaxFrequency = []
        MaxValue = []
        FirstHarmonicMag = []
        FirstHarmonicFreq = []
        bestChOrder = []
        currentCh = 0
        currentInflectionPnt = 0
        currentMagnitude = 0
        bestIpnt = 7
        bestIch = 0
        bestM = 0
        bestMch = 0
        bestMagnitude = 0

        F = []
        bestChannel = -1
        MaxChVal = 0.0
        for idx in range(0, int((NumOfSamples / 2))):
            F.append(SamplingFrequency * idx / NumOfSamples)
        for ch in range(0, NumOfChannels):
            fftData.append([])
            fftData[ch] = fft(ChData[ch])
            InflectionPoints.append(0)
            MaxFrequency.append(0)
            MaxValue.append(0)
            FirstHarmonicMag.append(0)
            FirstHarmonicFreq.append(0)

        for ch in range(0, NumOfChannels):
            for idx in range(0, int(len(fftData[ch])/2)):
                if((F[idx] >= 100) and (F[idx] < Filter_LowerCutoff_Hz)):
                    if(abs(fftData[ch][idx]) > FirstHarmonicMag[ch]):
                        FirstHarmonicMag[ch] = abs(fftData[ch][idx])
                        FirstHarmonicFreq[ch] = F[idx]
            
                if ((F[idx] >= Filter_LowerCutoff_Hz) and
                        (F[idx] <= Filter_UpperCutoff_Hz)):
                    if(abs(fftData[ch][idx]) > MaxValue[ch]):
                        MaxValue[ch] = abs(fftData[ch][idx])
                        MaxFrequency[ch] = F[idx]

        for ch in range(0, NumOfChannels):
            if(((MaxFrequency[ch] / 2) >= (((FirstHarmonicFreq[ch] + 0) * 1) - 15)) and
                ((MaxFrequency[ch] / 2) <= (((FirstHarmonicFreq[ch] + 0) * 1) + 15))):
                for chnl in range(0, NumOfChannels):
                    InflectionPoints[chnl] = 8
                #print("**-- Noise -- Ch{}".format(ch))
            for idx in range(1, int(len(fftData[ch])/2)):

                if((F[idx] >= 100) and (F[idx] < Filter_LowerCutoff_Hz)):
                    if(abs(fftData[ch][idx]) > 10000):
                        #print("** LowFreqHighMag break -- Ch{} Freq:{} Mag:{}".format(ch,F[idx], abs(fftData[ch][idx])))
                        for chnl in range(0, NumOfChannels):
                            InflectionPoints[chnl] = 8

                if ((F[idx] >= (MaxFrequency[ch] - 250.0)) and
                        (F[idx] <= (MaxFrequency[ch] + 250.0))):
                    if (abs(fftData[ch][idx]) > (MaxValue[ch] * 0.38)):
                        if (abs(fftData[ch][idx]) > (abs(fftData[ch][idx - 1]))):
                            goingUp = 1
                            downCnt = 0
                        if(goingUp == 1):
                            if (abs(fftData[ch][idx]) < (abs(fftData[ch][idx - 1]))):
                                goingUp = 0
                                downCnt = 1
                                InflectionPoints[ch] = int(InflectionPoints[ch]) + 1



        for ch in range(0, NumOfChannels):

            bestChannel = ch
            log = open(dataPath + "log.csv","a")
            log.write("{},{},{},{}\r".format(bestChannel, MaxFrequency[bestChannel], MaxValue[bestChannel], filename))
            log.close()
            #print("Best Channel: {}, Mag: {}, Fundamental Frequency: {}".format(bestChannel, MaxValue[bestChannel], MaxFrequency[bestChannel]))
            var = 48 + bestChannel
            wavFileName[len(wavFileName) - 5] = ord(int(var).to_bytes(1,'little'))
            writeFile((wavFileName.decode()),waveFiles[bestChannel])
            #print("Wave file: {} saved.!!".format(wavFileName.decode()))
            counter = counter + 1
            #print(counter)




    def __getitem__(self, idx):

        file_path = self.mbn_files_data['NewPath_vol2'][idx]

        """
       
        #MBN.MbnFile(file_path)
        wav1_file_name = os.path.basename(file_path)[:8]+'_Ch0'+'.wav'
        wav2_file_name = os.path.basename(file_path)[:8]+'_Ch1'+'.wav'
        wav1_path = os.path.join(os.path.split(file_path)[0],wav1_file_name)
        wav2_path = os.path.join(os.path.split(file_path)[0],wav2_file_name)
        y1,sr1 = librosa.load(wav1_path)
        y2,sr2 = librosa.load(wav2_path)
        y1_trim = y1[50:-50]
        y2_trim = y2[50:-50]

        trimmed_y = []

        for y_trim in (y1_trim,y2_trim): 
            peaks,properties = find_peaks(-y_trim,distance=500)
            y_values=[]

            for item in peaks: 
                y_value = -y_trim[item]
                y_values.append(y_value)

            x_y_list = list(zip(peaks,y_values))
            sorted_list = sorted(x_y_list, key=lambda x: x[1])
    
            highest_peak_x = sorted_list[-1][0]
            highest_peak_y = sorted_list[-1][1]
            

            if highest_peak_x - 1000 <0: 
                left_index = 0
                right_index = 2000
            elif highest_peak_x + 1000 >len(y_trim): 
                right_index= len(y_trim)
                left_index = len(y_trim)-2000
            else: 
                left_index = int(highest_peak_x-1000)
                right_index = int(highest_peak_x+1000)

            #print(left_index,right_index)
            y_trim = y_trim[left_index:right_index]

            trimmed_y.append(y_trim)
        
        """
        name_1 = os.path.basename(file_path)[:8] + '_trim1.wav'
        name_2 = os.path.basename(file_path)[:8] + '_trim2.wav'
        path_1 = os.path.join(os.path.split(file_path)[0],name_1)
        path_2 = os.path.join(os.path.split(file_path)[0],name_2)
        
        
        
        
        wav_trim_1,sr1 = librosa.load(path_1)
        wav_trim_2,sr2 = librosa.load(path_2)
        
        species = self.mbn_files_data['Species'][idx]
        #print(species)
        species_encoded = self.integer_encoded[idx]
        #print(species_encoded)
        set = self.mbn_files_data['Set'][idx]
        #print(trimmed_y)
        #sample = {'wav_tensor': [wav1_tensor,wav2_tensor], 'species': species, 'set':set}
        sample = {'file_path' : file_path,'channel_arrays': np.stack((wav_trim_1,wav_trim_2)), 'species': torch.tensor(species_encoded), 'set':set}

        #sample = {'mbn_tensor': mbn_tensor, 'species': species, 'set':set}

        return sample


if __name__ == '__main__':
    mbn_dataset = MBN(csv_file_path='/vol/bitbucket/ra2820/BITBUCKET/combined_split_vol.csv')


    df = pd.read_csv('/vol/bitbucket/ra2820/BITBUCKET/combined_split_vol.csv')
    index = df.index
    condition_train = df["Set"] == "train"
    condition_test = df["Set"] == "test"
    condition_val = df["Set"] == "val"
    train_indices = index[condition_train]
    test_indices = index[condition_test]
    val_indices = index[condition_val]


    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(val_indices)


    dataloader_train = DataLoader(mbn_dataset, batch_size=100,sampler = train_sampler,num_workers=6)
    dataloader_test = DataLoader(mbn_dataset, batch_size=100,sampler = test_sampler, num_workers=6)
    dataloader_val = DataLoader(mbn_dataset, batch_size=100,sampler = valid_sampler, num_workers=6)




