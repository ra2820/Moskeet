
import torch.optim as optim
import matplotlib.pyplot as plt 

from sklearn.metrics import balanced_accuracy_score

import dataloader_mbn


import torch
import torch.nn as nn
import torch.nn.functional as F


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


net = Net()
print(net)
    

# hyperparameters 
batch_size = 100
lr = 0.001

params = list(net.parameters())


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)







USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    






def check_accuracy(loader, model):
    # function for test accuracy on validation and test set

    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['channel_arrays'],batch['species']
            inputs = inputs.reshape(5284,1,2,2000)
            print(inputs)
            labels = torch.flatten(labels)
            labels = labels.type(torch.LongTensor)
            outputs_val = model(inputs)
            print(outputs_val)
            _, predicted = torch.max(outputs_val.data, 1)
            #print(predicted)
            #print(labels)
            num_correct += (predicted == labels).sum().item()
            num_samples += predicted.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))










for epoch in range(1):  # loop over the dataset multiple times

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
       
print('Finished Training')


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
