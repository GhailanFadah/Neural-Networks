import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa 
import tensorflow as tf
import librosa.display
from sklearn.utils import shuffle

def create_labels():
    '''
    creates np array of labels from metadata file and then map into int coding instead of strings
    '''
    raw_labels = pd.read_csv("data/bird_songs_metadata.csv", usecols=['species'])
    labels = pd.DataFrame(raw_labels.values)
    
    mapping = {'bewickii':0, 'polyglottos':1, 'migratorius':2, 'melodia':3, 'cardinalis':4}
    int_labels = labels.replace(mapping)
    
    labels_np = np.array(np.squeeze(int_labels))
    #print(len(labels_np))
    return labels_np
    
def create_files():
    #read metadata into array of filenames
    names = pd.read_csv("data/bird_songs_metadata.csv", usecols=['filename'])
    names = np.squeeze(names.values)
    #print(len(names))
    
    birdfiles = np.array([])
    for file in names:
        birdfiles = np.append(birdfiles, 'data/wavfiles/'+file)
        
    #print(birdfiles)
    #print(len(birdfiles))
    
    return birdfiles

def create_spec(file): 
    #create mel spectrogram with filepath as input
    audio_data, sample_rate = librosa.load(file, duration=10)
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate) 
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

def create_spec_audio(data, sr):
    #create mel spectrogram using the waveform as input
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr) 
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec


def create_dataset(train_split =5000, val_split=122, test_split=300):
    '''
    creates dataset for our birdsong classifier split along train_split, val_split, test_split. 
     We need sum of the lengths to be less than the number of smaples in the dataset
    '''
    # shuffle files
    files = create_files()
    print(len(files))
    labels = create_labels()
    files, labels = shuffle(files, labels)
    
    #files = pd.DataFrame(files, dtype=pd.StringDtype())
    #print(files.head())
    #spec = files.apply(lambda x: create_spec(x))
    spec_list = []
    for file in files:
        spec = create_spec(file)
        spec_list.append(spec)
        
    spec_array = np.array(spec_list)
    print(len(spec_array))
    x_train = spec_array[:train_split]
    y_train = labels[:train_split]
    x_test = spec_array[train_split:train_split+test_split]
    y_test = labels[train_split:train_split+test_split]
    x_val = spec_array[train_split+test_split:train_split+test_split+val_split]
    y_val = labels[train_split+test_split:train_split+test_split+val_split]
    
    return x_train, y_train, x_test, y_test, x_val, y_val


def split_data_specie():
    """
    returns data as arrays of filepaths corresponding to each label.
    """
    df = pd.read_csv("data/bird_songs_metadata.csv", usecols=['filename', 'species'])
    mapping = {'bewickii':0, 'polyglottos':1, 'migratorius':2, 'melodia':3, 'cardinalis':4}
    df = df.replace(mapping)
    df_path= pd.DataFrame({"species": df["species"], "path": "data/wavfiles/" + df["filename"]})
   
    df_path["mel_spec"] = df_path["path"].apply(lambda x: create_spec(x))
    df_path = df_path[['species', 'mel_spec']]
    df_path0= df_path[df_path['species'] ==0]
    df_path1= df_path[df_path['species'] ==1]
    df_path2= df_path[df_path['species'] ==2]
    df_path3= df_path[df_path['species'] ==3]
    df_path4= df_path[df_path['species'] ==4]
   
    return np.array(df_path0), np.array(df_path1), np.array(df_path2), np.array(df_path3), np.array(df_path4)

def split_data_specie_wave():
    '''
    returns that data as waveforms instead of as paths to be loaded later (split_data_specie)
    '''
    
    df = pd.read_csv("data/bird_songs_metadata.csv", usecols=['filename', 'species'])
    mapping = {'bewickii':0, 'polyglottos':1, 'migratorius':2, 'melodia':3, 'cardinalis':4}
    df = df.replace(mapping)
    df_path= pd.DataFrame({"species": df["species"], "path": "data/wavfiles/" + df["filename"]})
   
    # Removed the line that generates spectrograms
    df_path = df_path[['species', 'path']]  # Changed 'mel_spec' to 'path'
    df_path0= df_path[df_path['species'] ==0]
    df_path1= df_path[df_path['species'] ==1]
    df_path2= df_path[df_path['species'] ==2]
    df_path3= df_path[df_path['species'] ==3]
    df_path4= df_path[df_path['species'] ==4]


    # Load the audio data from the files
    data0 = [librosa.load(p, sr=None)[0] for p in df_path0['path']]
    data1 = [librosa.load(p, sr=None)[0] for p in df_path1['path']]
    data2 = [librosa.load(p, sr=None)[0] for p in df_path2['path']]
    data3 = [librosa.load(p, sr=None)[0] for p in df_path3['path']]
    data4 = [librosa.load(p, sr=None)[0] for p in df_path4['path']]

    return np.array(data0), np.array(data1), np.array(data2), np.array(data3), np.array(data4)

def create_superposition():
    
    a0,a1,a2,a3,a4 = split_data_specie()
    
    # make linear combinations with second specie being reduced by 0.7 
    a0_1 = a0[:,1] + (a1[0:893,1] *0.5)
    a2_3 = a2[:,1] + (a3[0:1017,1] *0.5)
    a3_4 = a3[0:1074,1] + (a4[:,1] *0.5)

    # make class labels for dominate specie
    zeros = np.zeros((893,1), int)
    twos = np.ones((1017, 1), int)*2
    threes = np.ones((1074, 1), int) *3
    
    # add labels to superposition data to create final dataset
    a0_1 = np.hstack((zeros,np.reshape(a0_1, (893,1))))
    a2_3 = np.hstack((twos, np.reshape(a2_3, (1017,1))))
    a3_4 = np.hstack((threes, np.reshape(a3_4, (1074 ,1))))
    
    #returns superposition data where the label for each sample is the dominate specie
    return a0_1, a2_3, a3_4
    
    
    
def alternate_create_superposition():
    '''
    creates a superposition of all combinations of dominant/nondominant species from our subset of xeno-canto
    
    Returns: 
    -------------------------
    data: the spectrograms of the superpositons of all combinations
    data_label: the spectrogram of the original/dominant samples with indices corresponding to data
    labels: int coded dominant classes of the generated data
    '''
    
    a0,a1,a2,a3,a4 = split_data_specie_wave()

    print(a0.shape)

    arrays = [a0, a1, a2, a3, a4]
    data = []
    data_label =[]

    for i in range(len(arrays)):
        for j in range(len(arrays)):
            if i != j:
                superposition = arrays[i][:893] + (arrays[j][:893] *0.4)
                data.append(superposition)
                data_label.append(arrays[i][:893])

    data = [create_spec_audio(d, 22050) for d in data]
    data_label = [create_spec_audio(d, 22050) for d in data_label]

    #labels
    labels_a0 =np.zeros((893*4,)) 
    labels_a1 =np.ones((893*4,)) 
    labels_a2 =np.ones((893*4,))*2
    labels_a3 =np.ones((893*4,))*3

    # Concatenate labels
    labels = np.concatenate((labels_a0, labels_a1, labels_a2, labels_a3))

    return data, data_label, labels
    
    
    
   
  
    