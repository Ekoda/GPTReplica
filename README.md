Work in progress.

GPTessence employs the power of PyTorch to implement a GPT (Generative Pretrained Transformer) model with decoder-only architecture. This project is the natural progression from the EssentialTransformer (https://github.com/Ekoda/EssentialTransformer), moving from implementing a transformer from scratch to harnessing the efficiency and optimization of PyTorch.

In other words, EssentialTransformer was an exploration into the intricate mechanics of the transformer model, GPTessence uses the existing tools in the PyTorch library to streamline the implementation.

On a personal note this is a place for me to explore ideas and use for reference for future projects.

---
## Example
The notebook in the scripts foders provice demonstration of the model being used. `prep_data.ipynb` handles tokenization and the training / validation split. `train.ipynb` demonstrates a training run of the model. All model and training configuration are handled in the `config.yaml` file for convenience, enabling quick and painless experimentation. 

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
