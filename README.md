A GPT (Generative Pretrained Transformer) model with a decoder-only architecture using PyTorch.

This project serves as a personal exploration space and reference for future model building. As it was developed for personal use, please excuse any messiness. The exploration folder contains my personal notes.

---
## Example
The notebook in the scripts foders provice demonstration of the model being used. `prep_data.ipynb` handles tokenization and the training / validation split. `train.ipynb` demonstrates a training run of the model. All model and training configuration are handled in the `config.yaml` file for convenience.

---
## Project Structure
The project is structured in a hierarchical and modular fashion according to the original "attention is all you need" paper (Vaswani et al., 2017). As such the code in the components folder contain most of the detail, while code such as the model.py contain the transformer which ties all the pieces together. The config files handles all of the configuration around the model and training

---
## Requirements
- Python 3.10 or later

Dependencies are listed in the `requirements.txt` file. To install these dependencies, navigate to the project directory and run the following command:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```
