import argparse
import sys
import os
import json

import dataloader_mbn
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt 

from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from torch.optim.optimizer import Optimizer
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import wandb
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Mosquito Classifier')
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    parser.add_argument('--batch_size', type=int, help='number of instances to pass through network', default=100)
    parser.add_argument('--epochs', type=int, help='number of epochs to train', default=1000)
    parser.add_argument('--model', type=str, help='which model to load', default='small')
    parser.add_argument('--weighted_loss', type=str2bool, help='Do you want to apply a weighted loss?', default=False)
    parser.add_argument('--batch_norm', type=str2bool, help='Do you want to apply batchnorm?', default=False)
    if len(sys.argv) == 1:
        print('using txt')
        with open(os.getcwd()+'/Endtoendapproach/args.txt', 'r') as f:
            args = argparse.Namespace(**json.loads(f.read()))
    else:
        print('not using txt')
        args = parser.parse_args()
    
    return args
                        
def plot_confusion(ys, y_hts):
    confusion_array = confusion_matrix(ys, y_hts)
    df_cm = pd.DataFrame(confusion_array, index = x_axis_labels,
                columns = y_axis_labels)
    fig, ax = plt.subplots(figsize = (10,10))
    ax = sns.heatmap(df_cm, annot=True)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Actual')
    return fig

def check_accuracy(loader, model):
    # function for test accuracy on validation and test set

    correct = 0
    total = 0
    model.eval()  # set model to evaluation mode 
    labels_list=[]
    predicted_list=[]
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['channel_arrays'],batch['species']

            if inputs.shape == (100,2,2000): 
                inputs = inputs.reshape(100,1,2,2000)
            
            else: 
                inputs = inputs.reshape(82,1,2,2000)

            #print(inputs)

            labels = torch.flatten(labels)
            labels = labels.type(torch.LongTensor)
            outputs_val = net(inputs.to(device))
            #print(outputs_val)

            _, predicted = torch.max(outputs_val.data, 1)
            #print(predicted)
            #print(labels)
            correct += (predicted.cpu() == labels).sum().item()
            total += labels.size(0)

            for item in predicted.squeeze().tolist(): 
                predicted_list.append(item)
            for item in labels.squeeze().tolist(): 
                labels_list.append(item)
    val_bal_accuracy = balanced_accuracy_score(labels_list, predicted_list)        
    acc = float(correct) / total
    val_accuracies.append(acc)
    #bal_acc = balanced_accuracy_score(y_test,y_pred)
    cm = confusion_matrix(np.array(labels_list), np.array(predicted_list))
    print(cm)
    fig = plot_confusion(np.array(labels_list), np.array(predicted_list))
    fig.savefig('experiment.png')
    wandb.log({"img": [wandb.Image(fig, caption="Confusion matrix")]})
    print('Got %d / %d correct (%.2f)' % (correct, total, 100 * acc))
    wandb.log({'val_acc': val_bal_accuracy})


def get_model(args):

    '''
    Loads the correct model
    '''
    if args.model == 'small':
        from neural_net import SmallNet as Net
    elif args.model == 'medium':
        from neural_net import MediumNet as Net
    elif args.model == 'large':
        from neural_net import LargeNet as Net
    elif args.model == 'huge':
        from neural_net import HugeNet as Net
    else:
        raise ValueError('This model is not implemented')
    net = Net()

    return net




if __name__ == '__main__':

    import wandb

    args = parse_args()
    print(args)
    wandb.init(entity='mosquito', project='Preliminary Analysis', config=args)
    net = get_model(args)
    wandb.watch(net)
    print(net)
   # Create dataloader and calculate weights
    torch.manual_seed(0)
    np.random.seed(0)

    mbn_dataset = dataloader_mbn.MBN(csv_file_path='/vol/bitbucket/ra2820/BITBUCKET/combined_split_vol.csv')


    df = pd.read_csv('/vol/bitbucket/ra2820/BITBUCKET/combined_split_vol.csv')


    y = df['Species']
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(integer_encoded),y=integer_encoded)
    class_weights=torch.tensor(class_weights,dtype=torch.float).cuda()
    print(f'integer_encoded {integer_encoded}')
    print(f'class_weights {class_weights}')
    print(f'unique counts {np.unique(integer_encoded)}')
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


    dataloader_train = DataLoader(mbn_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    dataloader_test = DataLoader(mbn_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4)
    dataloader_val = DataLoader(mbn_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=4)

        
    x_axis_labels = list(label_encoder.classes_) # labels for x-axis
    y_axis_labels = list(label_encoder.classes_)        



    params = list(net.parameters())


    criterion = nn.CrossEntropyLoss(weight=class_weights if args.weighted_loss else None,
                                    reduction='mean')
    optimizer = optim.Adam(net.parameters(),lr=args.lr)






    USE_GPU = True
    dtype = torch.float32 

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('Using device:', device)

    net.to(device)


   
    val_accuracies = []

    ##### TRAINING ####
    epochs_data_final= {'epoch':[], 'epoch_i_batch':[], 'epoch_loss':[], 'epoch_accuracy':[]}


    for epoch in range(args.epochs):  # loop over the dataset multiple times

        epochs_data={'i_batch':[],'loss':[],'accuracy':[]}

        running_loss = 0.0
        step_1 = time.perf_counter()
        for i_batch, sample_batch in enumerate(dataloader_train,0):
            
            if i_batch == 0:
                step_2 = time.perf_counter()
            inputs, labels = sample_batch['channel_arrays'],sample_batch['species']

        
            optimizer.zero_grad()

            if inputs.shape == (100,2,2000): 
                inputs = inputs.reshape(100,1,2,2000)
            
            else: 
                inputs = inputs.reshape(56,1,2,2000)

            outputs = net(inputs.to(device))

            if i_batch == 0:
                step_3 = time.perf_counter()

            
            labels = torch.flatten(labels)
            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels.to(device))
            loss.backward()

            optimizer.step()

            if i_batch == 0:
                step_4 = time.perf_counter()

                print(f'Time to load batch {step_2 - step_1:0.4f}')
                print(f'Time to pass batch through network {step_3 - step_2:0.4f}')
                print(f'Time to backprop + update {step_4 - step_3:0.4f}')
            running_loss += loss.item()
    
            if i_batch % 10 == 9:    
            
                _, predicted = torch.max(outputs.data, 1)   
                accuracy = balanced_accuracy_score(labels.detach().cpu().numpy(),predicted.detach().cpu().numpy())
                

                print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                    (epoch + 1, i_batch + 1, running_loss / 10, accuracy))
                
                epochs_data['i_batch'].append(i_batch+1)
                epochs_data['loss'].append(running_loss/10)
                epochs_data['accuracy'].append(accuracy)
                wandb.log({'train_acc': accuracy})
                wandb.log({'loss': loss.item()})
                running_loss = 0.0
        
        epochs_data_final['epoch'].append(epoch)
        epochs_data_final['epoch_i_batch'].append(epochs_data['i_batch'])
        epochs_data_final['epoch_loss'].append(epochs_data['loss'])
        epochs_data_final['epoch_accuracy'].append(epochs_data['accuracy'])

        if epoch % 5 ==0: 
            check_accuracy(dataloader_val,net)
    


        
    print('Finished Training')
    print(epochs_data_final)
