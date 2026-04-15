# ml-fashion-mnist-wgan

Projeto de estudo de redes generativas adversariais usando WGAN para gerar imagens no dominio Fashion-MNIST.

## Conteudo

- `train_wgan_fashionmnist.py`: script de treinamento da WGAN.
- `wgan_fashion_mnist_notebook.ipynb`: notebook de apoio ao experimento.
- `fashionmnist_wgan_outputs/`: exemplos/artefatos gerados durante o experimento, quando presentes.
- `.gitignore`: exclusoes de cache, ambiente virtual e artefatos temporarios.

## Objetivo do Estudo

O repositorio explora conceitos de modelos generativos:

- treinamento adversarial com gerador e discriminador/critic;
- uso de Fashion-MNIST como dominio visual simples;
- geracao de amostras sinteticas;
- acompanhamento qualitativo da evolucao do treinamento;
- organizacao de experimento reprodutivel em script e notebook.

## Base de Dados

O Fashion-MNIST e um dataset publico de imagens de pecas de roupa em escala de cinza. Ele e normalmente baixado automaticamente por bibliotecas de deep learning, entao a base bruta nao precisa ser mantida no repositorio.

## Como Executar

Instale as dependencias necessarias e rode:

```bash
python train_wgan_fashionmnist.py
```

Ou abra o notebook:

```bash
jupyter notebook wgan_fashion_mnist_notebook.ipynb
```

## Limitacoes

GANs sao sensiveis a hiperparametros, inicializacao e tempo de treino. Os resultados devem ser avaliados visualmente e tratados como experimento de estudo, nao como modelo generativo pronto para producao.
