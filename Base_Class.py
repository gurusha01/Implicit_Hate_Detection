import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model_class import YourModel


CLASSES = [['Pop culture reference', 'Pop-Culture references',  'pop-vulture references', 'Pop-culture references',   'pop-culture reference',  'pop-culture references',  'Pop-culture reference'],
['detailed', 'Detailed', 'Deatiled'],   
# ['Non hate', 'non-hate', 'Non Hate', 'non- hate', 'non hate'  'non  hate'],
['Stereotype', 'Stereptype',  'stereotype', 'stereeotype', 'sterotype', 'Sterotype', 'Stereotypes', 'Steretype'],
['Euphemiasm', 'Euphenism', 'euphemism',  'Euphemism',  'Euphimism'], 
['Sarcasm',  'sarcasm',  'saracsm'],
['Exaggeration', 'exaggeration', 'Exaggeration '], 
# ['Explicit hate', 'Explicit', 'explicit hate',    'Explicit Hate'],
['normative',  'Normative'], 
['Analogy', 'analogy', 'Anaology'],
['Circumlocution', 'circumlocution',  'circumlocation'],
['pun',  'Pun ', 'Pun'],
['humor', 'Humor',]]

class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder, text_file, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.text_file = text_file
        self.transform = transform
        self.image_names = [name for name in os.listdir(self.image_folder) if name.endswith('.jpg')]
        self.textdata = pd.read_csv(self.text_file)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        # breakpoint()

        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
        image_number = image_name[5:-4]
        csv_file = os.path.join(self.label_folder, os.path.splitext(image_name)[0] + '.csv')
        labels = self.read_labels_from_csv(csv_file, image, image_number)

        row = self.textdata.loc[self.textdata['image'] == image_name[:-4]]
        text = row["text"].values[0]
    

        if self.transform:
            image = self.transform(image)
        # print(labels)
        return image, text, torch.tensor(labels)

    def read_labels_from_csv(self, csv_file, image, image_number):
        # You can customize this function to read and process labels from your CSV file
        # For example, if the CSV has multiple labels per row, you can return a list of labels
        # This example assumes a single label per row
        # breakpoint()
        with open(csv_file, 'r') as file:
            label = file.readlines()  # Read the first line of the CSV file
        
        image_width, image_height = image.size
        labels = []
        keywords = []
        coordinates = []
        file_path = "labels/test/test" + image_number + ".txt"
        with open(file_path, "w") as file:
            # header = "class_id center_x center_y bbox_width bbox_height"
            # file.write(header+"\n")
            for row in label:
                if(len(row) > 6):
                    # print("ROW:",row)
                    contents = row.split(",")
                    labels.append(contents[0])
                    keywords.append(contents[1])
                    coordinates.append(contents[-4:])
                    try:
                        x1, y1, x2, y2 = [float(i) for i in coordinates[-1]]
                        center_x = ((x1+x2)/2) / image_width
                        center_y = ((y1+y2)/2)  / image_height
                        width = abs(x1-x2)  / image_width
                        height = abs(y1-y2) / image_height
                        class_label = 0
                        for i in range(len(CLASSES)):
                            if(contents[0] in CLASSES[i]):
                                class_label = i
                        data_list = [class_label, center_x, center_y, width, height]
                        data_str = " ".join(str(item) for item in data_list)
                        file.write(data_str+"\n")
                        image.save('images/test/test'+image_number+'.jpg')
                    except:
                        pass
                            

        one_hot_labels = [0]*len(CLASSES)
        for j in range(len(labels)):
            for i in range(len(CLASSES)):
                if(labels[j] in CLASSES[i]):
                    one_hot_labels[i] = 1
                    
        # breakpoint()
        return one_hot_labels

image_folder_path = 'test/test'
label_folder_path = 'all_annotation/all_annotation'
image_text_path = 'test_prop.csv'

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to (256, 256)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

dataset = CustomDataset(image_folder=image_folder_path, label_folder=label_folder_path, text_file=image_text_path, transform=transform)

batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = YourModel()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        # breakpoint()
        images, texts, batch_labels = data
        # optimizer.zero_grad()
        # outputs = model(images, texts)
        # loss = criterion(outputs, batch_labels)
        # loss.backward()
        # optimizer.step()
        # running_loss += loss.item()
        # if i % 10 == 9:  # Print every 10 mini-batches
        #     print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 10:.3f}")
        #     running_loss = 0.0

print("Training finished!")



