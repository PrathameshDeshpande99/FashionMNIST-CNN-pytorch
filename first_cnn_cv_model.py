import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import requests
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm
import pandas as pd
import random
import torchmetrics
import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

#Getting a dataset (FashionMNIST dataset)
train_data = datasets.FashionMNIST(
    root= "data_1",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root= "data_1",
    train=False,    #if train is false then test is true
    download=True,
    transform=ToTensor(),
    target_transform=None
)
class_names = train_data.classes
class_to_idx = train_data.class_to_idx

#Preapare data loader
'''we would break the 60k images into batches(mini-batches) as its more computationally efficient,
it gives more chances to update its gradient per epoch'''

BATCH_SIZE = 32
#turn data to batches
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

#A non-linear model
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 outut_shape: int,
                 hidden_units: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=outut_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)
    
#create instance of non-linear model
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784,
                              hidden_units=10,
                              outut_shape=len(class_names)).to(device)

#loss functoin, optimizer and accuracy function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)
def accuracy_fn(y_true, y_pred):
    #Calculates accuracy between truth labels and predictions.
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

#Defining training and testing step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs a training with model trying to learn on data_loader."""    
    train_loss, train_acc = 0,0
    model.train()

    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)

        #accumulate loss and accuracy for each batch
        loss = loss_fn(y_pred, y)

        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))   #for accuracy we need prediction labels (logits -> pred_prob -> pred_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """Performs testing with model trying to learn on data_loader"""
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)
            
            #accumulate loss and accuracy for each batch
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1)) #for accuracy we need prediction labels (logits -> pred_prob -> pred_labels)
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

#Checking how much time it takes to run
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}")
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_on_gpu = timer()

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    #prints difference between start and end time
    total_time = end - start
    print(f"Train time on {device} : {total_time:.3f} seconds")
    return total_time

total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)

'''NOTE: sometimes depending on your data/hardware your model may train faster on cpu than gpu
it could be that the overhead for copying data/model to and from gpu outweights the compute benefits offered by gpu.
'''

#Creating a function to evaluate our model
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = None):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:

            X,y = X.to(device), y.to(device)

            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
            
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__, #only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)
print(model_1_results)

print("\n\nFrom here on REAL CNN shit begins\n\n")
#Modle 2: CNNS (This NN replicates TinyVGG architecture on CNN Explainer website)
class FashionMNISTModelV2(nn.Module):
    """Model architecture replicating TinyVGG model from CNN explainer website"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, #there is a trick to calculating this...
                                                    #what ever is the output of conv_block_2 is flatten out and that is passed through linear
                                                    #layer so the in_fesatures must match accordingly
                                                    #there is a formula for this(or we can just calculate it) but as its simple and small data,
                                                    #so we can just pass a random tensor of same size and see what size output we get
                      out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
    
torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,    #its the number of colour_channels
                              hidden_units=10,  #just like normal hidden units
                              output_shape=len(class_names)).to(device) #no. of classes we want to classify to




#passing dummy data to our model to check if there is any shape mismatches error 
rand_image_tesnsor = torch.randn(size=(1,28,28)).to(device)
model_2(rand_image_tesnsor.unsqueeze(0)).to(device)

#Setup loss function/eval metric/optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)
def accuracy_fn(y_true, y_pred):
    #Calculates accuracy between truth labels and predictions.
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 3
train_time_start_model_2 = timer()
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end = train_time_end_model_2,
                                            device= device)

#get model 2 results
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)
print(model_2_results)

#Compare Data results and training time 
compare_results = pd.DataFrame([model_1_results,
                                model_2_results])
#compare training time too
compare_results["training_time"] = [total_train_time_model_1,
                                    total_train_time_model_2]
print(compare_results)

#Visualize our model results
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("accuracy(%)")
plt.ylabel("model")
plt.show() 

#Make and evaluate random predictions with cnn model
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs=[]
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #prepare the sample (add a batch dimensioin and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logits = model(sample)

            #Prediction probabilities 
            pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)

            #Get pred_prob off the GPU for further calc
            pred_probs.append(pred_prob.cpu())

    #stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

test_samples = []
test_labels = []
#get some random samples from the test dataset
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

#View the first sample shape
print(test_samples[0].shape)

plt.imshow(test_samples[0].squeeze(), cmap="gray")
plt.title(class_names[test_labels[0]])
plt.show()

#make predictions
pred_probs = make_predictions(model=model_2,
                              data=test_samples)

#View first two prediction probabilities 
print(pred_probs[:2])

#convert prediction probabilites to prediction labels 
pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)


#Plot prediction
plt.figure(figsize=(9,9))
nrows = 3
nclos = 3
for i, sample in enumerate(test_samples):

    plt.subplot(nrows, nclos, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")

    #find prediction label
    pred_label = class_names[pred_classes[i]]

    #get truth label
    truth_label = class_names[test_labels[i]]

    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    #Check for equality between pred and truth and change colour of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize = 10, color="green")
    else:
        plt.title(title_text, fontsize=10, color="red")
plt.show()


#Make a Confusion matrix (a way to evaluate CNN models visually)
#make predictions with trained model
y_preds=[]
model_2.to(device)
model_2.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader, desc="Make predictions..."):
        #send the data and labels to target device
        X,y = X.to(device), y.to(device)

        y_logits = model_2(X)

        y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)

        y_preds.append(y_pred.cpu())
    
    #concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)

print(y_pred_tensor)

#setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)   #confusion matrix tensor, targets = labels

#plot confusion matrix
fig, axis = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(6,6)
)
plt.show()

#Save and load best performing model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)
MODEL_NAME = "CNN_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#saving model state dict
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)

#loading and creating new instance from saved one
torch.manual_seed(42)
loaded_model_2 = FashionMNISTModelV2(input_shape=1, #should have same parameters as our saved model
                                     hidden_units=10,
                                     output_shape=len(class_names))
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_2.to(device)

#Evaluate loaded model
torch.manual_seed(42)
loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)
print(loaded_model_2_results)

#check if model results are close
print(torch.isclose(torch.tensor(model_2_results["model_loss"]),
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol=1e-04))   #atol is used for absoute tolerance (check if results are same upto 4 decimal pts in this case)