# Fashion-MNIST WGAN

Projeto acadêmico de modelagem generativa com WGAN/DCGAN aplicado ao dataset Fashion-MNIST.

## Conteúdo

- `wgan_fashion_mnist_notebook.ipynb`: notebook original da atividade.
- `train_wgan_fashionmnist.py`: script de treinamento em Python.
- `fashionmnist_wgan_outputs/`: imagens geradas durante o treinamento.

## Objetivo

Treinar um modelo generativo capaz de produzir novas imagens semelhantes às peças de roupa do Fashion-MNIST.

## Requisitos

- Python 3.10+
- Jupyter Notebook ou JupyterLab
- `tensorflow`
- `matplotlib`
- `numpy`

## Como executar o notebook

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter tensorflow matplotlib numpy
jupyter notebook wgan_fashion_mnist_notebook.ipynb
```

## Como executar o script

```bash
python3 train_wgan_fashionmnist.py
```

## Observações

- O script salva imagens geradas em `fashionmnist_wgan_outputs/`.
- Checkpoints e artefatos pesados futuros devem ser mantidos fora do versionamento.
