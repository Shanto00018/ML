import h5py
import numpy as nm

#function that read data from .h5 file
def dataRead(file_name):
    train_file=h5py.File(file_name,"r")

    #print(train_file.keys())
    keys_list=list(train_file.keys())
    x_key=keys_list[1]
    y_key=keys_list[2]

    train_data=train_file[x_key]
    train_label=train_file[y_key]

    train_label=nm.reshape(train_label,(-1,1))

    return (train_data,train_label)





#function that preprocess data for calculation like reshaping array
def dataPreProcess(train_data):
    dataset_size=train_data.shape[0]

    image_height=train_data.shape[1]
    image_width=train_data.shape[2]

    data=nm.zeros((image_height*image_width*3,dataset_size))

    for i in range (dataset_size):
        current_image=train_data[i]

        r=current_image[:,:,0]
        g=current_image[:,:,1]
        b=current_image[:,:,2]

        r=r.reshape(-1,1)
        g=g.reshape(-1,1)
        b=b.reshape(-1,1)

        image_array=nm.vstack((r,g,b))

        data[:,i]=image_array[:,0]

    return data


#forward propagation calculation
def forward_propagation(weight_array,bias,train_data):
    A=nm.dot(weight_array.T,train_data)+bias
    A=nm.clip(A,-700,700)
    A=1/(1+nm.exp(-A))
    return A.T


#backward propagation calculation
def backward_propagation(A,train_label,train_data,dataset_size):
    dz=A-train_label
    bias=nm.sum(dz)
    db=bias/dataset_size

    dw=nm.dot(train_data,dz)
    dw=dw/dataset_size

    return(dw,db)

#test analysis
def model_analysis(test_result,test_label):
    test_data_size=test_result.shape[0]
    match=0
    for i in range(test_data_size):
        if(test_result[i][0]>=.5):
            test_result[i][0]=1
        else:
            test_result[i][0]=0
        if(test_result[i][0]==test_label[i][0]):
            match+=1
    accuracy=(match/test_data_size)*100
    print("Total data ",test_data_size)
    print("Correct prediction number ",match)
    print("The accuracy of model is ",accuracy)
        

#main
train_data_file="train_catvsnoncat.h5"
test_data_file="test_catvsnoncat.h5"


training_set=dataRead(train_data_file)
train_data=training_set[0]
train_label=training_set[1]


dataset_size=train_data.shape[0]

data=dataPreProcess(train_data)

weight_array=nm.zeros((train_data.shape[1]*train_data.shape[2]*3,1))
bias=0
alpha=0.05
steps=2000


# print(weight_array)
# print(weight_array.shape)

for i in range(steps):
    A=forward_propagation(weight_array,bias,data)
    backwardpropagation_data=backward_propagation(A,train_label,data,dataset_size)
    dw=backwardpropagation_data[0]
    db=backwardpropagation_data[1]
    weight_array=weight_array-(alpha*dw)
    bias=bias-(alpha*db)

print("model trained")

test_set=dataRead(test_data_file)
test_data=test_set[0]
test_label=test_set[1]

test_array=dataPreProcess(test_data)
test_result=forward_propagation(weight_array,bias,test_array)

model_analysis(test_result,test_label)














