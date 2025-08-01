# 🎨 Image-to-Image Translation with CycleGAN: Edges2Shoes

> A deep learning project by **Ayush Vitthal Fande**, Final-Year B.Tech in AI & Data Science  
> 📍 Division: A | Roll No.: FY_AIDS_12 | Branch: AI&DS

---

## 📌 Project Overview

This project focuses on converting **edge sketches of shoes into photorealistic images** using **CycleGAN**, a powerful model for **unpaired image-to-image translation**.

Unlike paired models (like Pix2Pix), CycleGAN doesn’t require aligned image pairs. Instead, it learns to map between domains using **cycle consistency** and **adversarial training**.

---

## 🚀 Key Objectives

- Translate edge-drawn shoe sketches into realistic shoe images.
- Implement a complete training pipeline using the **Edges2Shoes** dataset from **Hugging Face**.
- Visualize results and evaluate generation quality.

---

## 🔧 Technologies Used

| Tool/Library       | Purpose                          |
|--------------------|----------------------------------|
| `datasets`         | Load Edges2Shoes from HuggingFace|
| `torch`, `torchvision` | Model + transforms         |
| `matplotlib`       | Image visualization              |
| `PIL`              | Image format handling            |
| `Google Colab`     | Training & visualization         |

---

## 📂 Dataset

- **Name**: `huggan/edges2shoes`
- **Source**: [HuggingFace Datasets](https://huggingface.co/datasets/huggan/edges2shoes)
- **Type**: Paired data (edges → shoes), used here for unpaired learning via CycleGAN
- **Split**: `train`

---

## 🧠 Model Architecture: CycleGAN

- **Two Generators**:
  - `G_AB`: edges ➝ shoes
  - `G_BA`: shoes ➝ edges
- **Two Discriminators**:
  - `D_A`: distinguishes real vs generated edges
  - `D_B`: distinguishes real vs generated shoes
- **Losses**:
  - Adversarial Loss
  - Cycle Consistency Loss

---

## 🛠️ Implementation Steps

### 📦 Step 1: Install Required Libraries

```bash
pip install datasets torchvision matplotlib
```
### 🧾 Step 2: Import Libraries
```python
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import io
```
🖼️ Step 3: Load Dataset
```python

dataset = load_dataset("nateraw/edges2shoes", split="train")
### 🧹 Step 4: Define Transformations
python
Copy
Edit
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
```
### 🧰 Step 5: Create Custom Dataset Class
```python

class Edges2ShoesDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = Image.open(io.BytesIO(sample['image']))
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
```
### 📦 Step 6: Create DataLoader
```python

shoes_dataset = Edges2ShoesDataset(dataset, transform=transform)
shoes_loader = DataLoader(shoes_dataset, batch_size=1, shuffle=True)
### 🖥️ Step 7: Visualize Sample Output
python
Copy
Edit
def show_image(img_tensor):
    img = img_tensor.squeeze().permute(1, 2, 0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```
### Display 5 images from the dataset
```python
for i, img in enumerate(shoes_loader):
    show_image(img)
    if i == 4:
        break
```
## ✅ Results

<img width="794" height="218" alt="image" src="https://github.com/user-attachments/assets/df1000ba-8bce-42e7-be3c-a96a158e3722" />

<img width="794" height="218" alt="image" src="https://github.com/user-attachments/assets/962ad24d-5659-452a-a923-3d7333bf7620" />
<img width="794" height="218" alt="image" src="https://github.com/user-attachments/assets/f31d3ad7-3925-4572-8cda-8219f1480a6b" />
<img width="794" height="218" alt="image" src="https://github.com/user-attachments/assets/c079340d-ad1c-4480-94aa-a9e53da08f9f" />


