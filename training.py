import os
import time
import pandas as pd
from sklearn.model_selection import RepeatedKFold as K_Fold

import torch
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from EarlyStopping import EarlyStopping

#### Modifiable variables ####

class Train:
    """_summary_
    """    
    def __init__(self, path_train, name, name_instrument, net="ResNet18"):
        """_summary_

        Args:
            path_train (_type_): _description_
            name (_type_): _description_
            name_instrument (_type_): _description_
        """        
        self.path_train = path_train
        self.name = name
        self.name_instrument = name_instrument
        self.num_classes = self.instrument(name_instrument)
        self.model = None
        self.net = net

        # Training parameters
        self.batch_size = 64  # Batch size for DataLoader
        self.num_epochs = 100  # Maximum number of epochs to train
        self.learning_rate = 0.01  # Learning rate for optimizer
        self.num_of_splits = 5  # Number of splits in K-fold cross-validation
        self.num_of_repeats = 1  # Number of times K-fold cross-validation is repeated
        self.remake_model = False  # Set to True if a new model is to be trained every time
        self.patience = 30  # Early stopping patience (number of epochs without improvement)
        self.min_delta = 0.0  # Minimum delta for improvement to reset early stopping counter

            
    def instrument(self, name_instrument):
        """_summary_

        Args:
            name_instrument (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # Label mapping (class labels)
        if name_instrument == "planktivore":
            labels_map = {
                0: "Aggregate",
                1: "Bad_Mask",
                2: "Blurry",
                3: "Camera_Ring",
                4: "Ciliate",
                5: "Copepod",
                6: "Diatom_Long_Chain",
                7: "Diatom_Long_Single",
                8: "Diatom_Spike_Chain",
                9: "Diatom_Sprial_Chain",
                10: "Diatom_Square_Single",
                11: "Dinoflagellate_Circles",
                12: "Dinoflagellate_Horns",
                13: "Phaeocystis",
                14: "Radiolaria"
            }
        if name_instrument == "issis":
                labels_map = {
                0: "aggregate",
                1: "artifact",
                2: "centric_diatom",
                3: "copepod",
                4: "diatom_chain",
                5: "fecal_pellet",
                6: "football",
                7: "gelatinous",
                8: "larvacean",
                9: "rhizaria",
                10: "phaeocystis",
                11: "long_particle_blur",
                12: "particle_blur"
            }
        elif name_instrument == "microscopy":
                labels_map = {
                0: "aggregate",
                1: "fiber_blur",
                2: "long_pellet",
                3: "noise",
                4: "phyto_dino",
                5: "phyto_round",
                6: "salp_pellet",
                7: "swimmer",
                8: "bubble",
                9: "fiber_sharp",
                10: "mini_pellet",
                11: "none",
                12: "phyto_long",
                13: "rhizaria",
                14: "short_pellet"
            } 
                
        # Counter for labeled data
        label_count = {label: 0 for label in labels_map.values()}

        # Number of classes in the dataset
        num_classes = len(labels_map)

        return num_classes



    def prepare_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        #Load the dataset
        # Define the transformations to apply to the images
        transform = transforms.Compose([ 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #This changes the pixel to have mean of 0 and a std of 1. We will probably need to change this for our dataset. subtract the mean and divide by the std.
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)
        ])

        dataset = ImageFolder(root = self.path_train, transform=transform)
        return dataset

    def load_model(self):
        """_summary_
        """        
        #Load the resnet18 model on first run unless a pre-run model is found
        if (not os.path.exists('{}.pth'.format(self.name)) or self.remake_model):
            print("Previous model does not exist, loading resnet18")
            self.model = models.resnet18(weights='DEFAULT')#, weights='HM_weights')
            #Modify the last layer of the model
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
        else:
            print("Previous model found, loading {}".format(self.name))
            self.model = models.resnet18(weights='DEFAULT')
            #Modify the last layer of the model
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
            self.model = torch.load('{}.pth'.format(self.name))
            print("Loading the model with the weights of: ", self.name)

        # Set the device
        print("Cuda Found: " + str(torch.cuda.is_available()))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def save_model(self):
        """
        Save the trained model as a .pth file. The model is saved in the 'History/Models' directory, organized by the 
        current date. The filename is based on the ISO date format.
        
        Args:
            model (torch.nn.Module): The trained PyTorch model to be saved.
        """
        torch.save(self.model, '{}.pth'.format(self.name))
        #filename = time.strftime("%Y-%m-%dT%H:%M_Model.pth", time.gmtime())
        filename = '{}.pth'.format(self.name)

        try:
            path = os.path.join("Models", self.name)
            os.makedirs(path, exist_ok=True)
            torch.save(self.model, os.path.join(path, self.net))
        except OSError as error:
            print(error)



    def training(self, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer):
        """_summary_

        Args:
            train_loader (_type_): _description_
            val_loader (_type_): _description_
            train_dataset (_type_): _description_
            val_dataset (_type_): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Create early stopping object to monitor validation loss
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=self.min_delta)

        # Training loop over epochs
        for epoch in range(self.num_epochs):
            # Set the model to train mode
            self.model.train()

            running_loss = 0.0  # Tracks loss across batches
            running_corrects = 0  # Tracks correct predictions across batches

            # Iterate through the training data
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available

                optimizer.zero_grad()  # Reset gradients from the previous step

                outputs = self.model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get the predictions
                loss = criterion(outputs, labels)  # Compute the loss

                loss.backward()  # Backward pass (compute gradients)
                optimizer.step()  # Update model parameters

                running_loss += loss.item() * inputs.size(0)  # Accumulate loss
                running_corrects += torch.sum(preds == labels.data)  # Accumulate correct predictions

            # Compute average training loss and accuracy for the epoch
            train_loss = running_loss / len(train_dataset)
            train_acc = running_corrects.double() / len(train_dataset)

            # Validation phase: set the model to evaluation mode
            self.model.eval()
            running_val_loss = 0.0
            running_val_corrects = 0

            # Disable gradient calculation for validation (saves memory and computation)
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.model(inputs)  # Forward pass
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)  # Compute validation loss

                    running_val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss
                    running_val_corrects += torch.sum(preds == labels.data)  # Accumulate validation accuracy

            # Compute average validation loss and accuracy
            val_loss = running_val_loss / len(val_dataset)
            val_acc = running_val_corrects.double() / len(val_dataset)

            # Print epoch statistics
            print(f'Epoch [{epoch+1}/{self.num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, '
                f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

            # Check if early stopping is triggered based on validation loss
            if early_stopping(val_acc, self.model):
                print("\nEarly stopping triggered, halting training.\n")
                break

        # Load the best model after early stopping
        early_stopping.load_checkpoint(self.model).to(device)

    
    def create_image_csv(self):
        """
        Create a CSV file that records image names and their corresponding labels from the args.pathtrain directory.
        
        The CSV file is saved in the 'History/CSVs' directory, organized by the current date. The filename is based on the
        ISO date format.
        """
        print("Saving CSV")

        folder_path = self.path_train
        data = {
            "image": [],
            "label": []
        }

        # Iterate through all subdirectories in args.path, adding image filenames and labels to data
        for dirs in os.listdir(folder_path):
            working_dir = os.path.join(folder_path, dirs)
            for files in os.listdir(working_dir):
                data["image"].append(files)
                data["label"].append(dirs)

        df = pd.DataFrame(data)

        try:
            path = os.path.join("Models", self.name, "training")
            os.makedirs(path, exist_ok=True)
            df.to_csv(os.path.join(path, "training_data"), header=False, index=False)
                    # Write the metrics to the text file (saving in the same directory as the CSV)
            path = os.path.join("Models", self.name, self.net)
            metrics_file_path = os.path.join(path, 'metrics_report.txt')
            with open(metrics_file_path, 'w') as f:
                f.write(f'Model: {self.name}\n')
                f.write(f'batch_size: {self.batch_size}\n')
                f.write(f'num_epochs: {self.num_epochs}\n')
                f.write(f'learning_rate: {self.learning_rate}\n')
                f.write(f'num_of_splits: {self.num_of_splits}\n')
                f.write(f'batch_size: {self.batch_size}\n')
                f.write(f'num_of_repeats: {self.num_of_repeats}\n')
                f.write(f'remake_model: {self.remake_model}\n')
                f.write(f'patience: {self.patience}\n')
                f.write(f'min_delta: {self.min_delta}\n')
                f.write(f'intrument: {self.name_instrument}\n')
        except OSError as error:
            print(error)
    
    def run(self):
        """_summary_
        """    

        self.load_model()

        #set up cross-validation
        kf = K_Fold(n_splits = self.num_of_splits, n_repeats = self.num_of_repeats) 

        #Freeze all the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        #Modify the last layer of the model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)



        # Define the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.fc.parameters(), lr=self.learning_rate, momentum=0.9)


        dataset = self.prepare_data()

        #Cross_Validation
        for fold, (train_index,val_index) in enumerate(kf.split(dataset)):
            print(f'Fold {fold + 1}')

            #Create the train and the evaluate subsets from the cross-validation split
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            self.training(train_loader, val_loader, train_subset, val_subset, criterion, optimizer, num_epochs=self.num_epochs)

        # Unfreeze all the layers and fine-tune the entire network for a few more epochs
        for param in self.model.parameters():
            param.requires_grad = True

        # Fine-tune the last layer for a few epochs
        for fold, (train_index,val_index) in enumerate(kf.split(dataset)):
            print(f'Fold {fold + 1}')

            #Create the train and the evaluate subsets from the cross-validation split
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            self.training(self.model, train_loader, val_loader, train_subset, val_subset, criterion, optimizer, num_epochs=self.num_epochs)


        self.create_image_csv()
        self.save_model()


