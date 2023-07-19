import os
import warnings

import torch
import torchvision.transforms.functional as F
import os
import json
import PIL.Image
import xml.etree.ElementTree as ET
import torch
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import json
import PIL.Image

class MaskDetectionDatasetJSON(Dataset):
    """
    This is a custom dataset class for the mask detection dataset suited for the JSON annotations.
    It implements the abstract methods from the PyTorch Dataset class.
    It is suitable for use with the PyTorch DataLoader class and made for the Faster R-CNN model.
    """


    def __init__(self, root_dir,class_label_map, target_size=(512, 512), only_single_faces=False, only_multiple_faces=False ):
        """
        root_dir: The root directory of the dataset
        class_label_map: A dictionary mapping the class names to class labels
        target_size: The target size of the images.
        only_single_faces: If true, only images with a single face will be present in the dataset
        only_multiple_faces: If true, only images with multiple faces will be present in the dataset

        The images are not loaded directly, but only the annotations are loaded.
        When a item is requested in the __getitem__ method, the image is loaded and the annotations are applied to the image.
        """

        print("dataset init")
        self.root_dir = root_dir
        self.class_label_map = class_label_map
        self.annotations = []
        self.target_size = target_size
        self.only_single_faces = only_single_faces
        self.only_multiple_faces = only_multiple_faces
        if(self.only_single_faces and self.only_multiple_faces):
            raise ValueError("only_single_faces and only_multiple_faces cannot both be true\n Only one of them can be true")
        
        self.load_annotations()

    def load_annotations(self):
        """
        load the annotations from the json files in the annotations folder.
        This method is called in the constructor of the class.
        The annotations are stored in the self.annotations list and represent the dataset.
        """

        annotation_files = os.listdir(f"{self.root_dir}/annotations")
        for file_name in annotation_files:
            with open(f"{self.root_dir}/annotations/{file_name}", "r") as f:
                annotation_data = json.load(f)
                annotations = annotation_data["Annotations"]
                file_name = annotation_data["FileName"]
                #get the allowed class names from the keys of the class_label_map dictionary
                allowed_classes = self.class_label_map.values()      
                allowed_classnames = [key for key, value in self.class_label_map.items() if value in allowed_classes]
                #print("allowed classnames: ", allowed_classnames)
                face_classes = ["face_no_mask", "face_with_mask_incorrect", "face_with_mask", "face_other_covering"]
                annotations = [annotation for annotation in annotations if annotation["classname"] in allowed_classnames]

                if self.only_single_faces:
                    #check if multiple of the face_classes are present in the annotations, indicating multiple faces
                    face_annotations = [annotation for annotation in annotations if annotation["classname"] in face_classes]
                    if len(face_annotations) != 1:
                        continue 
                    self.annotations.append((annotations, file_name))

                elif self.only_multiple_faces:
                    #check if multiple of the face_classes are present in the annotations, indicating multiple faces
                    face_annotations = [annotation for annotation in annotations if annotation["classname"] in face_classes]
                    if len(face_annotations) <= 1:
                        continue
                    else:
                        self.annotations.append((annotations, file_name))

                else:
                    if(annotations == []):
                        warnings.warn(f"File {file_name} has no annotations")
                        continue
                    self.annotations.append((annotations, file_name))
                # Check if the boxes are valid
                for annotation in annotations:
                    boxes = annotation["BoundingBox"]
                    if boxes[0] >= boxes[2] or boxes[1] >= boxes[3]:
                        print("Invalid bounding box coordinates in file:", file_name)
                        break

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        method to get a single item from the dataset.
        This method is called by the PyTorch DataLoader class.
        It returns a tuple of the image and the target.
        The target is a dictionary containing the bounding boxes and the class labels.
        Also converts output to tensor to be ready for use in the model.
        """

        annotations = self.annotations[idx][0]
        file_name = self.annotations[idx][1]
        image_path = f"{self.root_dir}/images/{file_name}"
        image = PIL.Image.open(image_path).convert("RGB")
        original_image_width, original_image_height = image.size
        image = F.resize(image, self.target_size)
        image = F.to_tensor(image)

        boxes = []
        labels = []
        for annotation in annotations:
            box = annotation["BoundingBox"]
            if box[0] < box[2] and box[1] < box[3]:
                # Resize the bounding box coordinates
                box_resized = [
                    box[0] * self.target_size[0] / original_image_width,
                    box[1] * self.target_size[1] / original_image_height,
                    box[2] * self.target_size[0] / original_image_width,
                    box[3] * self.target_size[1] / original_image_height
                ]
                boxes.append(box_resized)
                class_name = annotation["classname"]
                # Get the class label based on the class name
                class_label = self.get_class_label(class_name)
                labels.append(class_label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        #could be used for evaluation currently unused
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        return image, target

    
    def get_class_label(self, class_name):
        return self.class_label_map.get(class_name, -1)  # Return -1 if class_name is not found


class MaskDetectionDatasetXML(Dataset):
    """
    This is a custom dataset class for the mask detection dataset suited for the XML annotations.
    It implements the abstract methods from the PyTorch Dataset class.
    It is suitable for use with the PyTorch DataLoader class and made for the Faster R-CNN model.
    """

    def __init__(self, root_dir,class_label_map, target_size=(512,512), use_dark_images=False):
        self.class_label_map = class_label_map
        self.root_dir = root_dir
        self.annotations = []
        self.target_size = target_size
        self.use_dark_images = use_dark_images
        self.load_annotations()

    def load_annotations(self):
        """
        load the annotations from the json files in the annotations folder.
        This method is called in the constructor of the class.
        The annotations are stored in the self.annotations list and represent the dataset.
        """
        
        annotation_files = os.listdir(f"{self.root_dir}/annotations")
        for file_name in annotation_files:
            with open(f"{self.root_dir}/annotations/{file_name}", "r") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                annotations = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    bounding_box = [xmin, ymin, xmax, ymax]
                    annotation = {
                        "BoundingBox": bounding_box,
                        "classname": name
                    }
                    annotations.append(annotation)
                file_name = root.find('filename').text
                self.annotations.append((annotations, file_name))

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        method to get a single item from the dataset.
        This method is called by the PyTorch DataLoader class.
        It returns a tuple of the image and the target.
        The target is a dictionary containing the bounding boxes and the class labels.
        Also converts output to tensor to be ready for use in the model.
        """
                
        annotations = self.annotations[idx][0]
        file_name = self.annotations[idx][1]
        image_path = f"{self.root_dir}/images/{file_name}"

        if(self.use_dark_images):
            image_path = f"{self.root_dir}/dark/images/{file_name}"
        image = PIL.Image.open(image_path).convert("RGB")
        original_image_width, original_image_height = image.size
        image = F.resize(image, self.target_size)
        image = F.to_tensor(image)

        boxes = []
        labels = []
        for annotation in annotations:
            box = annotation["BoundingBox"]
            if box[0] < box[2] and box[1] < box[3]:
                # Resize the bounding box coordinates
                box_resized = [
                    box[0] * self.target_size[0] / original_image_width,
                    box[1] * self.target_size[1] / original_image_height,
                    box[2] * self.target_size[0] / original_image_width,
                    box[3] * self.target_size[1] / original_image_height
                ]
                boxes.append(box_resized)
                class_name = annotation["classname"]
                # Get the class label based on the class name
                class_label = self.get_class_label(class_name)
                labels.append(class_label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        return image, target

    def get_class_label(self, class_name):
        return self.class_label_map.get(class_name, -1)  # Return -1 if class_name is not found