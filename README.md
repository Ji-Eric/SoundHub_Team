# Bioacoustics Classification with Stratified K-Fold Cross-Validation

This project performs binary and multi-class classification of bioacoustic audio embeddings using a **Stratified K-Fold Cross-Validation** approach. It leverages pre-trained embedding models from the **OpenSoundscape** library and evaluates model performance using ROC AUC scores.

---

## 📂 Project Structure

```
├── train_on_embeddings_kfold_copy.ipynb  # Main notebook
├── requirements.txt                      # Dependencies list
├── README.md                             # Project instructions
└── data/                                 
```

---

## Environment Setup

It is recommended to use a Python virtual environment for package management and isolation.

### 1. Create Virtual Environment

Using **conda**:
```bash
conda create -n soundhub python=3.10
```

### 2. Activate Virtual Environment

```bash
conda activate soundhub
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

---

### (Optional) Register Environment as Jupyter Kernel

If you want to run the notebook in this conda environment:

1. Install ipykernel:
   ```bash
   conda install ipykernel
   ```

2. Add the environment as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name soundhub --display-name "Python (soundhub)"
   ```

After this, you can select **Python (soundhub)** as your kernel in Jupyter Notebook.

-

## 📦 Main Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- torch
- opensoundscape
- bioacoustics-model-zoo

The project uses models from the **Bioacoustics Model Zoo**. To install the model zoo package manually (if needed):

```bash
pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo
```
