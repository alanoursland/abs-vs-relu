# Abs vs ReLU: A Comparative Study

## Introduction

This project aims to investigate the effectiveness and interpretability of the absolute value activation function (Abs()) compared to the widely-used Rectified Linear Unit (ReLU). The motivation behind this study is to explore whether Abs() can be as effective as ReLU while providing improved interpretability, particularly in the context of neural networks.

## Motivation

The primary motivations for this research are:
1. **Effectiveness:** To demonstrate that Abs() can be a viable alternative to ReLU without compromising the performance of neural networks across various tasks and datasets.
2. **Interpretability:** To show that using Abs() can enhance the interpretability of neural network models by centering hyperplanes on clusters, thus offering a more intuitive understanding of the model's decision-making process.
3. **Homomorphism Between Gaussians and Linear Separators:** To explore the theoretical benefits of using Abs() in the context of the homomorphism between Gaussians and linear separators, potentially leading to more interpretable models.

## Project Structure

```
project_root/
├── datasets/
│   ├── MNIST/
│   ├── CIFAR-10/
│   ├── CIFAR-100/
│   ├── IMDB/
│   ├── AG_News/
│   ├── Adult/
│   └── Wine_Quality/
├── experiments/
│   ├── MNIST_LeNet_ReLU/
│   ├── MNIST_LeNet_Abs/
│   ├── CIFAR10_ResNet18_ReLU/
│   ├── CIFAR10_ResNet18_Abs/
│   ├── CIFAR100_ResNet18_ReLU/
│   ├── CIFAR100_ResNet18_Abs/
│   ├── IMDB_LSTM_ReLU/
│   ├── IMDB_LSTM_Abs/
│   ├── AGNews_LSTM_ReLU/
│   ├── AGNews_LSTM_Abs/
│   ├── Adult_MLP_ReLU/
│   ├── Adult_MLP_Abs/
│   ├── WineQuality_MLP_ReLU/
│   └── WineQuality_MLP_Abs/
├── results/
│   ├── MNIST/
│   ├── CIFAR10/
│   ├── CIFAR100/
│   ├── IMDB/
│   ├── AGNews/
│   ├── Adult/
│   └── WineQuality/
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── utils/
│   ├── main.py
│   └── config.py
└── README.md
```

## Datasets

The following datasets are used in this study:
- **Image Classification:** MNIST, CIFAR-10, CIFAR-100
- **Natural Language Processing:** IMDB (sentiment analysis), AG News (text classification)
- **Tabular Data:** UCI Adult dataset, Wine Quality dataset

## Model Architectures

Different model architectures are employed to cover various complexities and types:
- **Convolutional Neural Networks (CNNs):** LeNet for MNIST, ResNet-18 for CIFAR-10 and CIFAR-100
- **Recurrent Neural Networks (RNNs):** LSTM for IMDB and AG News
- **Fully Connected Networks (FCNs):** 3-layer MLP for UCI Adult and Wine Quality datasets

## Experiments

For each dataset and model architecture, the following experiments are conducted:
1. Train the model using ReLU activation function.
2. Train the model using Abs activation function.
3. Measure and compare the performance using metrics such as accuracy, precision, recall, F1-score, training time, and inference time.
4. Evaluate interpretability through activation distribution, feature importance (for tabular data), and visualizations.

## Results

Results are stored in the `results/` directory, with separate subdirectories for each dataset. Each subdirectory contains aggregated results for each combination of dataset, model, and activation function, along with summary files comparing ReLU and Abs.

## How to Run

1. **Setup:**
   - Clone the repository.
   - Install the required dependencies using `pip install -r requirements.txt`.
   - Download and place the datasets in the appropriate subdirectories under `datasets/`.

2. **Run Experiments:**
   - Use the scripts in the `src/training/` directory to train models with the desired configurations. For example, to train a model on MNIST with ReLU, run:
     ```bash
     python src/training/train_mnist.py --activation relu
     ```
   - Similarly, to train a model with Abs, use:
     ```bash
     python src/training/train_mnist.py --activation abs
     ```

3. **Analyze Results:**
   - Results for each run will be saved in the corresponding `experiments/` subdirectory.
   - Aggregate and analyze results using the scripts provided in the `src/utils/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This repository includes modified code from the PyTorch project, which is licensed under the BSD license. The original code is copyrighted by Facebook, Inc. and other contributors. See the derivative files for the full text of the PyTorch license.

Modifications to the PyTorch LSTM implementation were made by Alan Oursland.

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

We would like to thank the contributors and the open-source community for their invaluable support and resources.
