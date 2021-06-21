
import dataloader_mbn


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt 

from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from torch.optim.optimizer import Optimizer
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import time

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 2,2 kernel
        self.conv1 = nn.Conv2d(1, 6, (2,3))
        self.conv2 = nn.Conv2d(6, 10, (1,3))
        self.conv3 = nn.Conv2d(10, 16, (1,3))
        self.fc1 = nn.Linear(16 * 73, 84) 
        self.fc2 = nn.Linear(84, 28)

    def forward(self, x):
        # Max pooling over a (1, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (1, 3))
    
        x = F.max_pool2d(F.relu(self.conv2(x)), (1,3))
        x = F.max_pool2d(F.relu(self.conv3(x)), (1,3))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
      
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x= F.softmax(x)
        return x

if __name__ == '__main__':

    import wandb
    wandb.init(entity='mosquito', project='Preliminary Analysis')
    net = Net()
    wandb.watch(net)
    print(net)
        

    # hyperparameters 
    batch_size = 100
    lr = 0.0002

    params = list(net.parameters())


    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #change to Adam
    optimizer = optim.Adam(net.parameters(),lr=lr)






    USE_GPU = True
    dtype = torch.float32 

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('Using device:', device)

    net.to(device)


    val_accuracies = []

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
        print('Got %d / %d correct (%.2f)' % (correct, total, 100 * acc))
        wandb.log({'val_acc': val_bal_accuracy})







    ##### TRAINING ####
    epochs_data_final= {'epoch':[], 'epoch_i_batch':[], 'epoch_loss':[], 'epoch_accuracy':[]}



    for epoch in range(100):  # loop over the dataset multiple times

        epochs_data={'i_batch':[],'loss':[],'accuracy':[]}

        running_loss = 0.0
        step_1 = time.perf_counter()
        for i_batch, sample_batch in enumerate(dataloader_mbn.dataloader_train,0):
            
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

        if epoch % 1 ==0: 
            check_accuracy(dataloader_mbn.dataloader_val,net)
    


        
    print('Finished Training')
    print(epochs_data_final)














"""
epochs_data_final= {'epoch':[], 'epoch_i_batch':[], 'epoch_loss':[], 'epoch_accuracy':[]}



for epoch in range(10):  # loop over the dataset multiple times

    epochs_data={'i_batch':[],'loss':[],'accuracy':[]}

    running_loss = 0.0

    for i_batch, sample_batch in enumerate(dataloader_mbn.dataloader_train,0):
 
        inputs, labels = sample_batch['channel_arrays'],sample_batch['species']

     
        optimizer.zero_grad()

        if inputs.shape == (100,2,2000): 
            inputs = inputs.reshape(100,1,2,2000)
        
        else: 
            inputs = inputs.reshape(56,1,2,2000)

        outputs = net(inputs)
    
        labels = torch.flatten(labels)
        labels = labels.type(torch.LongTensor)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
 
        if i_batch % 10 == 9:    
           
            _, predicted = torch.max(outputs.data, 1)   
         
            accuracy = balanced_accuracy_score(labels.detach().numpy(),predicted.detach().numpy())
            

            print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / 10, accuracy))
            
            epochs_data['i_batch'].append(i_batch+1)
            epochs_data['loss'].append(running_loss/10)
            epochs_data['accuracy'].append(accuracy)
             
            running_loss = 0.0
    
    epochs_data_final['epoch'].append(epoch)
    epochs_data_final['epoch_i_batch'].append(epochs_data['i_batch'])
    epochs_data_final['epoch_loss'].append(epochs_data['loss'])
    epochs_data_final['epoch_accuracy'].append(epochs_data['accuracy'])

    if epoch % 5 ==4: 
        check_accuracy(dataloader_mbn.dataloader_val,net)
   






       
print('Finished Training')
print(epochs_data_final)

plt.plot(epochs_data['i_batch'],epochs_data['loss'])
plt.title('1 Epoch - Loss vs. batches')
plt.xlabel('Batch number')
plt.ylabel('Loss')
plt.show()

plt.plot(epochs_data['i_batch'],epochs_data['accuracy'])
plt.title('1 Epoch - accuracy vs. batches')
plt.xlabel('Batch number')
plt.ylabel('accuracy')
plt.show()

"""