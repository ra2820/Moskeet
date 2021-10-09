# Base line Results
Run baseline_results.py. This will load the entire data (All_features.csv) and perform classification

# Utils 
Run PCA.py. This will run PCA and TSNE on 7_AMCA_Cleaned.csv. 

# End to End model 
In Moskeet/Endtoendapproach run main.py.
This will import the chosen model from either neural_net.py or resnet.py. 
Chosen hyper parameters will be imported from args.txt.
The model will be trained and evaluated accordingly. 

Hyper parameter sweeps are contained in the .yaml files and the wav files are trimmed according to wav_trim.py.

