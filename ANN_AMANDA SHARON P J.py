import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F          
from torch.utils.data import DataLoader  #load data
from torchvision import datasets, transforms
import streamlit as st
#%matplotlib inline

transform = transforms.ToTensor()
dataset = st.sidebar.text_input("Dataset:", value="MNIST")
model_name = st.sidebar.text_input("Model Name:", value="model-mnist")

train_data = datasets.MNIST(root='/CNN/Data', train=True, download=True, transform=transform)

test_data = datasets.MNIST(root='/CNN/Data', train=False, download=True, transform=transform)

#@st.cache_data
def main():
    st.title("Final Project : Multimodal Biomedical Image Analysis")
    st.text("Amanda Sharon Purwanti Junior // 5023201044")
    
    st.header( "Artificial Neural Network")
    st.text("Artificial Neural Network (ANN) adalah jenis model jaringan saraf tiruan \nyang terinspirasi dari struktur dan fungsi jaringan saraf biologis.ANN\ndigunakan dalam bidang kecerdasan buatan (AI) dan pembelajaran mesin \n(machine learning) untuk memodelkan kemampuan manusia dalam belajar \ndan memecahkan masalah.")
    traindata= int(st.sidebar.number_input("Training Data Number:", min_value=1, max_value = 600))
    image, label = train_data[traindata]
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,4))
    ax.imshow(train_data[traindata][0].reshape((28,28)),cmap="gray")
    ax.set_title('Train Data')
    st.pyplot(fig)

    st.write(f'Shape : {image.shape}')
    st.write(f'Label : {label}')
    #plt.imshow(train_data[0][0].reshape((28,28)), cmap="gray");


    torch.manual_seed(101)  # agar mendapatkan hasil yang konsisten
    train_batch_size = int(st.sidebar.number_input("Train Batch Size:", min_value=1, max_value = 600))
    test_batch_size = int(st.sidebar.number_input("Test Batch Size:", min_value=1, max_value = 600))
    epochs = st.sidebar.number_input('Enter Epoch Value', 0, 20, 1)


    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    #image,label = train_data[0]
    st.sidebar.text_input('Shape:', image.shape)


    from torchvision.utils import make_grid
    np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}')) # format to widen the printed array

    # Grab the first batch of images
    for images,labels in train_loader: 
        break

# Print the first 12 labels from 100 labels (remember we set 1 batch = 100 images)
    labell = st.sidebar.slider("Banyak Batch Data:", 1,test_batch_size)

# Print the first 12 images
    im = make_grid(images[:labell], nrow=14)  
    fig = plt.figure(figsize=(10,4))
    ax.set_title('Batch Data')
    
# We need to transpose the images from CWH (Color, Width, Height) to WHC (Width, Height, Color)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
    st.pyplot(fig)

    st.text_area('labels: ', labels[:labell].numpy(),)


    class MultilayerPerceptron(nn.Module):
        def __init__(self, in_sz=784, out_sz=10, layers=[120,84]):
            super().__init__()
            self.fc1 = nn.Linear(in_sz,layers[0]) #input layer, layers[0]==> taken grom layer=120, look the function header
            self.fc2 = nn.Linear(layers[0],layers[1]) #hidden layer 1
            self.fc3 = nn.Linear(layers[1],out_sz) #hidden layer 2 to output
    
        def forward(self,X):
            X = F.relu(self.fc1(X)) #forward input layer to hidden layer 1
            X = F.relu(self.fc2(X)) #result from previous layer, pass to the hidden layer 1 - 2
            X = self.fc3(X) #forward process from hidden layer 2 to output
            return F.log_softmax(X, dim=1) #dim=dimension

    torch.manual_seed(101)
    model = MultilayerPerceptron()
    model
#bias = True meaning that we add bias in each neuron

    criterion = nn.CrossEntropyLoss()
    learnrate = st.slider("learn rate(/10):", 0.000, 1.000)
    optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)

    # Load the first batch, print its shape
    for images, labels in train_loader:
        st.write('Batch shape:', {images.size()})
        break
    

    images.view(100,-1).size()

    #-1 , meaning grab all after array 100 until sequence before -1, and combine them into a single dimension

    #TRAIN DATA
    import time
    start_time = time.time()

    
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0 #train correct currently
        tst_corr = 0 #test correst
    
    # Run the training batches
    # with enumerate, we're actually going to keep track of what batch number we're on with B.
        for b, (X_train, y_train) in enumerate(train_loader): #y_train=output = label, b = batches, train_loader = return back the image and its label
            b+=1
        
            # Apply the model
            y_pred = model(X_train.view(train_batch_size, -1))  # Here we flatten X_train
            loss = criterion(y_pred, y_train) #calculating error difference
 
        # calculate the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1] #check print(y_pred.data) to know data of one epoch, 1 = actual predicted value
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
        
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Print results per epoch:
            if b%200 == 0:
                st.write(f'epoch: {i:2}  batch: {b:4} [{100*b:6}/60000]  loss: {loss.item():10.8f}  \ accuracy: {trn_corr.item()*100/(100*b):7.3f}%')
    
    # Update train loss & accuracy for the epoch
        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())
        
        # Run the testing batches
        with torch.no_grad(): #don't update weight and bias in test data
            for b, (X_test, y_test) in enumerate(test_loader):

                # Apply the model
                y_val = model(X_test.view(test_batch_size, -1))  # Here we flatten X_test, 500 because batch size for test data in cell above = 500

                # Calculating the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()
    
    # Update test loss & accuracy for the epoch
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
        
    st.write(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed            

# Evaluation
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='training loss')
    ax.plot(test_losses, label='validation loss')
    ax.set_title('Loss at the end of each epoch')
    ax.legend();
    st.pyplot(fig)

    fig, ax = plt.subplots()
    train_size = 60000/train_batch_size
    test_size= 10000/train_batch_size
    ax.plot([t/train_size for t in train_correct],label='training accuracy')
    ax.plot([t/test_size for t in train_correct],label='validation accuracy')
    ax.set_title('accuracy at the end of each epoch')
    ax.legend()
    st.pyplot(fig)

#Evaluate test data
# Extract the data all at once, not in batches
    test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

    with torch.no_grad():
        correct = 0
        for X_test, y_test in test_load_all:
            y_val = model(X_test.view(len(X_test), -1))  # pass in a flattened view of X_test
            predicted = torch.max(y_val,1)[1]
            correct += (predicted == y_test).sum()
    st.write(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')


#Confusion Matrix
# print a row of values for reference
    np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}')) #x:4 => giving space each data
    #print(np.arange(10).reshape(1,10))
    #print()

# print the confusion matrix
    st.write(confusion_matrix(predicted.view(-1), y_test.view(-1))) #The view(-1) operation flattens the tensor,
if __name__ == '__main__':
    main()

