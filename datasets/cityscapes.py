from PIL import Image
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Resize

class CityscapesDataset(Dataset):
    RESOLUTION = (512, 1024)
    def __init__(self):
        super().__init__()
        self.data, self.label, self.labelIds = [], [], []

        with open("datasets/cityscapes.txt", 'r') as f:
            for line in f.readlines():
                data, label = line.strip().split(',')
                self.data.append(data)
                self.label.append(label.replace('labelIds', 'color'))
                self.labelIds.append(label.replace('labelIds', 'labelTrainIds'))
        assert len(self.data) == len(self.label), "Data len is not equal as label len."

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, index):
        data_path = self.data[index]
        label_path = self.label[index]

        img_arr = cv2.imread(data_path)
        label_arr = cv2.imread(label_path)

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        label_arr = cv2.cvtColor(label_arr, cv2.COLOR_BGR2RGB)

        resize = Resize(size=self.RESOLUTION)
        img_arr = resize(torch.tensor(img_arr).permute(2,0,1)).permute(1,2,0).numpy()
        label_arr = resize(torch.tensor(label_arr).permute(2,0,1)).permute(1,2,0).numpy()

        img_arr = (img_arr.astype(np.float32) / 127.5) -  1.0
        label_arr = label_arr.astype(np.float32) / 255.0

        prompt = "City road scenes."

        label_trainId = cv2.imread(self.labelIds[index], cv2.IMREAD_GRAYSCALE)

        return dict(jpg=img_arr, txt=prompt, hint=label_arr, label=resize(torch.tensor(label_trainId)[None, ...])[0].numpy(), orig_img_path=data_path, orig_label_path=self.labelIds[index])
    
if __name__ == "__main__":
    a = CityscapesDataset()
    b = a[0]
    img = b['jpg']
    label = b['hint']
    img = img * 255
    label = (label + 1.0) * 127.5

    img = Image.fromarray(img.astype(np.uint8))
    label = Image.fromarray(label.astype(np.uint8))

    # import IPython; IPython.embed()
    # img.save('img.png')
    # label.save('label.png')    
