#Import PyTorch and Torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import copy


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train(dataloader, model, loss_fn, optimizer):
    t_start = time.time()
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #print(batch)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if batch % 100 == 0:
        #    loss, current = loss.item(), batch * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    print(f"Time taken:{time.time()-t_start:>3f}")

def SingleTrain (epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Single Train Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)


def NetworkIteration(num_devices,dataloader, global_model,global_optimizer, loss_fn,local_model,local_optimizer):

    for lm in local_model:
        lm.load_state_dict(global_model.state_dict())
        
    t_start = time.time()
    size = len(dataloader.dataset)
    print(f"Len of DataLoader={size}")
    for batch, (X, y) in enumerate(dataloader):
        model_idx = batch % num_devices
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = local_model[model_idx] (X)
        loss = loss_fn(pred, y)
        local_optimizer[model_idx].zero_grad()
        loss.backward()
        local_optimizer[model_idx].step()

    print(f"Train Time taken:{time.time()-t_start:>3f}")
    MergeResultHighK(global_model,local_model)
    #MergeResultAllReduce(global_model,local_model)


def MergeResultAllReduce(global_model,local_model): # No compression

    key_list  = []
    key_to_idx = {}
    for name in global_model.state_dict():
        key_to_idx[name] =len(key_list)
        key_list.append(name)

    #print(key_list)
    #print(key_to_idx)
    global_dict = global_model.state_dict()
    diff_lst = []
    for lm in local_model:
        local_diff = {}
        local_dict = lm.state_dict()
        for key in key_list:
            local_diff[key] = local_dict[key] - global_dict[key]
            
        diff_lst.append(local_diff)
    
    for key in key_list:
        reduce_sum = diff_lst[0][key]
        for idx in range(1, len(diff_lst)):
            reduce_sum = reduce_sum + diff_lst[idx][key]
        reduce_sum = reduce_sum/len(diff_lst) 
        global_dict[key] = global_dict[key]+reduce_sum
        
    global_model.load_state_dict(global_dict)

def MergeHighK(keys,global_dict,HighK_lst):
    print("MergeHighK")
    delta_lst = []
    delta_count = []
    for k in keys:
        #print('one ',k)
        z = torch.zeros_like(global_dict[k] , device = "cpu")
        delta_lst.append(torch.reshape(z, (-1,)))
        cnt = torch.zeros_like(global_dict[k] , device = "cpu")
        delta_count.append(torch.reshape(cnt, (-1,)))
    
    for high_k in HighK_lst:
        cnt = 0
        prog = int(len(high_k) / 4)
        for item in high_k:
            if cnt % prog == 0 and cnt !=0:
                print(cnt/prog*25,"%")
            cnt += 1
            value = item[1]
            kid = item[2]
            loc = item[3]
            delta_lst[kid][loc] += value
            delta_count[kid][loc] += 1


    for idx in range(len(keys)):
        k = keys[idx]
        #print('two ',k)
        delta = delta_lst[idx]
        cnt = delta_count[idx]
        for i, x in enumerate(cnt.numpy()):
            if x > 0:
                delta[i] /= x 

        delta = torch.reshape(delta, global_dict[k].size()).to(device)
        global_dict[k] = global_dict[k]+delta


def MergeResultHighK(global_model,local_model): # High K

    K = 0
    count_total = 0
    count_trainable = 0

    key_trainable = []
    key_trainable_idx = {}
    for name, param in global_model.named_parameters():
        if param.requires_grad:
            ele = 1
            sz = global_model.state_dict()[name].size()
            for s in sz:
                ele = ele * s
            count_trainable += ele
            key_trainable_idx[name] = len(key_trainable)
            key_trainable.append(name)

    key_total = []
    key_other = []
    for name in global_model.state_dict():
        
        key_total.append(name)
        ele = 1
        sz = global_model.state_dict()[name].size()
        for s in sz:
                ele = ele * s
        if len(sz) == 0:
            ele = 0
        count_total += ele
        if name not in key_trainable_idx:
            key_other.append(name)

    print(f"Total {len(key_total)} {count_total}")
    print(f"Trainable {len(key_trainable)} {count_trainable}")
    print(f"Others {len(key_other)} {count_total - count_trainable}")

    #print(count_total,count_trainable,count_total - count_trainable,(count_total - count_trainable)/count_total  )

    K = int(count_trainable * 0.003)
    print("K=",K)

    # Compute Gradient
    global_dict = global_model.state_dict()
    diff_lst = []
    for lm in local_model:
        local_diff = {}
        local_dict = lm.state_dict()
        for key in key_total:
            local_diff[key] = local_dict[key] - global_dict[key]
        diff_lst.append(local_diff)

     # Merge Other
    for key in key_other:
        reduce_sum = diff_lst[0][key]
        for idx in range(1, len(diff_lst)):
            reduce_sum = reduce_sum + diff_lst[idx][key]
        reduce_sum = reduce_sum/len(diff_lst) 
        global_dict[key] = global_dict[key]+reduce_sum
    
    # Merge High-K
    HighK_lst = []
    for idx in range(len(diff_lst)):
        print('indxing ',idx)
        sort_lst = []
        for key in key_trainable:
            kid = key_trainable_idx[key]
            X = torch.reshape(diff_lst[idx][key], (-1,))
            for i, x in enumerate(X.cpu().numpy()):
                sort_lst.append( (abs(x),x,kid,i) )
        sorted_lst = sorted(sort_lst, key=lambda ls: ls[0],reverse=True) # sort by the abs
        HighK_lst.append(sorted_lst[:K])

    MergeHighK(key_trainable,global_dict,HighK_lst)

    global_model.load_state_dict(global_dict)




def NetworkTrain (num_devices,epochs ):
    print(f"Netwrok Train num_devices={num_devices}, epochs={epochs}")

    model.train()
    local_model = []
    local_optimizer = []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for r in range(num_devices):     
        model_copy = copy.deepcopy(model)
        model_copy.train()
        local_model.append(model_copy)
    for r in range(num_devices):   
        local_optimizer.append(torch.optim.SGD(local_model[r].parameters(), lr=1e-3))

    print(f"Single Train Epoch {0}\n-------------------------------")
    #train(train_dataloader, model, loss_fn, optimizer)
    #test(test_dataloader, model, loss_fn)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        NetworkIteration(num_devices,train_dataloader, model,optimizer, loss_fn,local_model,local_optimizer)
        test(test_dataloader, model, loss_fn)

    # Get cpu or gpu device for training.
device = "mps"

print(f"Using {device} device")

#from torchvision.models import resnet50, ResNet50_Weights
#Resmodel = resnet50(weights = ResNet50_Weights.DEFAULT)

#from torchvision.models import resnet34, ResNet34_Weights
#Resmodel = resnet34(weights = ResNet34_Weights.DEFAULT)

from torchvision.models import resnet18, ResNet18_Weights
Resmodel = resnet18(weights = ResNet18_Weights.DEFAULT)


num_classes = 10          # set output class number 
num_ftrs = Resmodel.fc.in_features
Resmodel.fc = nn.Linear(num_ftrs, num_classes)


model = Resmodel.to(device)   # set current net

loss_fn = nn.CrossEntropyLoss()

training_dataSVHN = datasets.SVHN(
    root="data",
    split ='train',
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_dataSVHN = datasets.SVHN(
    root="data",
    split ='test',
    download=True,
    transform=ToTensor(),
)

batch_size = 384

# Create data loaders.
train_dataloader = DataLoader(training_dataSVHN, batch_size=batch_size,shuffle=False)
test_dataloader = DataLoader(test_dataSVHN, batch_size=batch_size,shuffle=False)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


epochs = 20
#SingleTrain(epochs)
NetworkTrain(4,epochs)
print("Done!")