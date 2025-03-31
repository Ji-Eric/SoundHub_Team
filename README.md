# Bioacoustics Classification with Stratified K-Fold Cross-Validation

This project performs binary and multi-class classification of bioacoustic audio embeddings using a **Stratified K-Fold Cross-Validation** approach. It leverages pre-trained embedding models from the **OpenSoundscape** library and evaluates model performance using ROC AUC scores.

## Project Structure

- **train_on_embeddings_kfold_copy.ipynb** — Main notebook for running classification experiments.
- **requirements.txt** — List of required Python packages.
- **data/** — Folder to store audio data and embeddings (not included, needs to be prepared by the user).

---

## Environment Setup

It is recommended to use a Python virtual environment for package management and isolation.

### 1. Create Virtual Environment

```bash
python3 -m venv soundhub
```

### 2. Activate Virtual Environment

- On macOS/Linux:
  ```bash
  source soundhub/bin/activate 
  ```

- On Windows:
  ```bash
  soundhub\Scripts\activate
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

## Running the Notebook

Once the environment is set up:

```bash
jupyter notebook
```

and open **train_on_embeddings_kfold_copy.ipynb** to run the experiments.



## Main Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- torch
- opensoundscape
- bioacoustics-model-zoo

