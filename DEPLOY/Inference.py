import numpy as np
import torchvision.transforms as transforms
from trism import TritonModel
from torchvision.datasets import MNIST

model = TritonModel(
    model="mnist_cnn",  
    version=1,         
    url="localhost:8001", 
    grpc=True  
)

def prepare_input():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    image, label = testset[0] 
    
    input_data = image.numpy().astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)  

    return input_data, label


input_data, true_label = prepare_input()

outputs = model.run(data=np.array([input_data]))

print("Outputs:", outputs)

output_key = list(outputs.keys())[0]  
output_data = outputs[output_key]
predicted_class = np.argmax(output_data)

print(f"True Label: {true_label}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence Scores: {output_data}")

del model
