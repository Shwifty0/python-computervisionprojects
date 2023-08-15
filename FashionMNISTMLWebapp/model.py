import torch
from torch import nn
from torchvision.datasets import  FashionMNIST
from torchvision.transforms import ToTensor


# Instantiate the model 
class FashionMNISTModel(nn.Module):
    """
    Model capable of predicting on MNIST Dataset.
    """
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.conv_block_1  = nn.Sequential( 
            nn.Conv2d(in_channels=input_shape,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1, 
                     padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                     out_channels=hidden_units, 
                     kernel_size=3, 
                     stride=1,
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv_block_2 = nn.Sequential( 
            nn.Conv2d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1, 
                     padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                     out_channels=hidden_units, 
                     kernel_size=3, 
                     stride=1,
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= 490,
                     out_features=output_shape))
        
    def forward(self, x):
        x = self.conv_block_1(x)
        #print(f"output shape of conv block 1: {x.shape}")
        x = self.conv_block_2(x)
        #print(f"output shape of conv block 2: {x.shape}")
        x = self.classifier(x)
        #print(f"output shape of classifier: {x.shape}")
        return x

model = FashionMNISTModel(input_shape=1, output_shape=10, hidden_units=10)
model.load_state_dict(torch.load("FashionMNISTmodel.pth"))
model

""" from PIL import Image
from torchvision.transforms import transforms

image = Image.open("./uploaded_images/img1.png")
print(image.format)
print(image.size)


# Using torchvision.transforms to manipulate the shape of input shape
transform1 = transforms.Grayscale(1)
transform2 = transforms.PILToTensor()

# Applying transformation
img_tensor = transform1(image)
tensor_image = transform2(img_tensor)
tensor_image = tensor_image.type(torch.float)
print(f"tensor_image shape:{tensor_image.shape}, tensor_image dtype: {tensor_image.dtype}")


# Making prediciton
model.eval()
with torch.inference_mode():
    prediction = torch.argmax(torch.softmax(model(tensor_image.unsqueeze(dim = 0)), dim = 1), dim = 1)
print(f"The predicted label of the class is : {prediction}, LETSFUCKING GO!")
 """




test_data = FashionMNIST(root='data', train=False, transform=ToTensor(), download= True)
#print(f"Test_data classes: {test_data.classes}")
#print(test_data.classes[0])