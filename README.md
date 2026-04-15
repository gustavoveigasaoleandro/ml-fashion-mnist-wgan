# ml-fashion-mnist-wgan

Projeto de estudo de redes generativas adversariais usando WGAN para gerar imagens no domínio Fashion-MNIST.

## Conteúdo

- `train_wgan_fashionmnist.py`: script de treinamento da WGAN.
- `wgan_fashion_mnist_notebook.ipynb`: notebook de apoio ao experimento.
- `fashionmnist_wgan_outputs/`: exemplos/artefatos gerados durante o experimento, quando presentes.
- `.gitignore`: exclusões de cache, ambiente virtual e artefatos temporários.

## Objetivo do Estudo

O repositório explora conceitos de modelos generativos:

- treinamento adversarial com gerador e discriminador/critic;
- uso de Fashion-MNIST como domínio visual simples;
- geração de amostras sintéticas;
- acompanhamento qualitativo da evolução do treinamento;
- organização de experimento reprodutível em script e notebook.

## Base de Dados

O Fashion-MNIST e um dataset público de imagens de peças de roupa em escala de cinza. Ele é normalmente baixado automaticamente por bibliotecas de deep learning, entao a base bruta não precisa ser mantida no repositório.

## Como Executar

Instale as dependências necessárias e rode:

```bash
python train_wgan_fashionmnist.py
```

Ou abra o notebook:

```bash
jupyter notebook wgan_fashion_mnist_notebook.ipynb
```

## Limitações

GANs são sensíveis a hiperparâmetros, inicialização e tempo de treino. Os resultados devem ser avaliados visualmente e tratados como experimento de estudo, não como modelo generativo pronto para produção.
