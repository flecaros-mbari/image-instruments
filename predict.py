import os
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
import torch
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

class Test:
    """_summary_
    """    
    def __init__(self,model, path_test, name, name_instrument, net = "ResNet18"):
        """_summary_

        Args:
            model (_type_): _description_
            path_test (_type_): _description_
            name (_type_): _description_
            name_instrument (_type_): _description_
        """    
        self.path_test = path_test
        self.name = name
        self.name_instrument = name_instrument
        self.naet = net

        self.num_classes, self.labels_map = self.instrument(name_instrument)

    
        model = models.resnet18(weights='DEFAULT')
        #Modify the last layer of the model
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        model = torch.load('Models/{}/{}.pth'.format(self.name, self.net))
        print("Loading the model with the weights of: ", self.name," ", self.net)
        self.model = model

        # Set the device
        print("Cuda Found: " + str(torch.cuda.is_available()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
      
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

        return num_classes, labels_map

    def run(self):
        """_summary_
        """    

        # Set the model to evaluate mode and disable the tensor.backward() call
        self.model.eval()  # Set the model to evaluation mode
        torch.no_grad()

        # Dictionary to keep track of label counts
        label_count = defaultdict(int)

        # List to store logs
        logs = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        transform = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        # Iterate over each image in the folder recursively
        for root, dirs, files in os.walk(self.path_test):
            for img_name in files:
                _, ext = os.path.splitext(img_name)
                if ext.lower() in image_extensions:
                    img_path = os.path.join(root, img_name)
                    image = Image.open(img_path)
                    image_tensor = transform(image).unsqueeze(0).to(self.device)  # Apply transformation and move to device
                    outputs = self.model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_label = predicted.item()

                    # Using the label map to get the label name
                    predicted_label_name = self.labels_map[predicted_label]

                    # Determine the true label from the folder name
                    true_label = os.path.basename(root)

                    # Update label count
                    label_count[predicted_label_name] += 1

                    # Create folder if it doesn't exist
                    label_folder = os.path.join(self.path_test, predicted_label_name)
                    os.makedirs(label_folder, exist_ok=True)

                    # Save the image to the corresponding folder
                    destination_path = os.path.join(label_folder, img_name)
                    os.rename(img_path, destination_path)

                    # Log the image path, predicted label, and true label
                    logs.append({
                        "image_path": img_path,
                        "predicted_label": predicted_label_name,
                        "true_label": true_label
                    })

                    print(f'Saved {img_name} to {label_folder}')

        # Create a DataFrame from the logs and save to CSV
        log_df = pd.DataFrame(logs)
        log_csv_path = os.path.join("Models", self.name)
        os.makedirs(log_csv_path, exist_ok=True)

        # Save the log CSV file
        csv_file_path = os.path.join(log_csv_path, "images_csv")
        log_df.to_csv(csv_file_path, index=False)
        self.model_metrics(csv_file_path)

    def model_metrics(self, csv_file_path, log_csv_path):
        """_summary_

        Args:
            csv_file_path (_type_): _description_
        """        
        # Load the CSV that your code produced
        df = pd.read_csv(csv_file_path)

        # Extract true labels and predicted labels from the DataFrame
        true_labels = df['true_label']
        predicted_labels = df['predicted_label']

        # Calculate the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=list(set(true_labels)))

        # Calculate accuracy, F1-score, and precision
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')  # Use 'weighted' if labels are imbalanced
        precision = precision_score(true_labels, predicted_labels, average='weighted')

        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')
        print(f'Precision: {precision}')

        path = os.path.join("Models", self.name, self.net)
        # Write the metrics to the text file (saving in the same directory as the CSV)
        metrics_file_path = os.path.join(path, 'metrics_report.txt')
        with open(metrics_file_path, 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Precision: {precision}\n')

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=set(true_labels), yticklabels=set(true_labels))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        # Save the confusion matrix as an image (use a separate filename, not args.name)
        confusion_matrix_path = os.path.join(path, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()

        # Count the frequencies of true and predicted labels
        true_label_counts = df['true_label'].value_counts()
        predicted_label_counts = df['predicted_label'].value_counts()

        # Create a DataFrame combining the frequencies of true and predicted labels
        freq_df = pd.DataFrame({
            'True Labels': true_label_counts,
            'Predicted Labels': predicted_label_counts
        }).fillna(0)  # Fill NaN with 0 if some labels are not predicted

        # Create the frequency bar chart
        ax = freq_df.plot(kind='bar', figsize=(10, 7), color=['blue', 'orange'])

        # Add labels and title
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.title('Frequency of True vs Predicted Labels')

        # Save the frequency chart as an image (use a separate filename, not args.name)
        frequency_chart_path = os.path.join(path, 'label_frequencies.png')
        plt.savefig(frequency_chart_path)

        # Close the figure to free memory
        plt.close()
