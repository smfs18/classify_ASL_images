# Treinamento de uma CNN para Reconhecimento de Sinais (ASL)

Este código demonstra o processo completo de construção e treinamento de um modelo de Rede Neural Convolucional (CNN) usando TensorFlow e Keras. O objetivo é classificar imagens de letras do Alfabeto da Língua de Sinais Americana (ASL).

## 1. Importação das Bibliotecas

Primeiro, importamos as bibliotecas necessárias. O `pandas` é utilizado para carregar e manipular os dados, enquanto o `tensorflow.keras` fornece as ferramentas para construir e treinar nosso modelo de deep learning.

```python
import tensorflow.keras as keras
import pandas as pd
```

---

## 2. Carregamento e Preparação dos Dados

Os dados de treinamento e validação são carregados a partir de arquivos CSV. Em seguida, separamos as características (pixels da imagem) dos rótulos (a letra correspondente).

- `train_df` e `valid_df`: DataFrames do pandas contendo os dados.
- `y_train` e `y_valid`: Contêm os rótulos (labels) de cada imagem.
- `x_train` e `x_valid`: Contêm os valores dos pixels de cada imagem.

```python
# Carrega os dados dos arquivos CSV
train_df = pd.read_csv("/content/gdrive/MyDrive/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("/content/gdrive/MyDrive/asl_data/sign_mnist_valid.csv")

# Separa os rótulos (target)
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

# Separa os vetores das imagens (features)
x_train = train_df.values
x_valid = valid_df.values
```

---

## 3. Pré-processamento dos Dados

Antes de treinar o modelo, os dados precisam ser pré-processados:

1.  **Categorização dos Rótulos**: Os rótulos numéricos (0 a 23) são convertidos em um formato de codificação *one-hot* usando `to_categorical`. Isso é necessário para a função de perda `categorical_crossentropy`.
2.  **Normalização das Imagens**: Os valores dos pixels (que variam de 0 a 255) são normalizados para um intervalo entre 0 e 1. Isso ajuda a otimizar o processo de treinamento.
3.  **Redimensionamento das Imagens**: Os vetores de imagem (de 784 pixels) são redimensionados para o formato `(28, 28, 1)`, que é o formato de entrada esperado por uma camada convolucional (`Conv2D`) para imagens em escala de cinza.

```python
# Transforma os rótulos escalares em categorias binárias (one-hot encoding)
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Normaliza os dados da imagem
x_train = x_train / 255
x_valid = x_valid / 255

# Redimensiona as imagens para o formato 28x28x1
x_train = x_train.reshape(-1, 28, 28, 1)
x_valid = x_valid.reshape(-1, 28, 28, 1)
```

---

## 4. Construção do Modelo (Arquitetura da CNN)

A arquitetura do modelo é construída sequencialmente usando a API `Sequential` do Keras.

-   **`Conv2D`**: Camadas convolucionais que aplicam filtros para extrair características das imagens (bordas, texturas, etc.).
-   **`BatchNormalization`**: Normaliza as ativações da camada anterior, estabilizando e acelerando o treinamento.
-   **`MaxPool2D`**: Reduz a dimensionalidade espacial dos mapas de características, mantendo as informações mais importantes.
-   **`Dropout`**: Técnica de regularização que zera aleatoriamente uma fração das unidades de entrada durante o treinamento para prevenir o sobreajuste (*overfitting*).
-   **`Flatten`**: Transforma a matriz 2D de características em um vetor 1D para ser processado pelas camadas densas.
-   **`Dense`**: Camadas totalmente conectadas. A última camada usa a função de ativação `softmax` para produzir as probabilidades de cada uma das 24 classes.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

model = Sequential()

# Bloco 1
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))

# Bloco 2
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))

# Bloco 3
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))

# Camadas de classificação
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))

# Exibe um resumo da arquitetura do modelo
model.summary()
```

---

## 5. Compilação do Modelo

O modelo é compilado definindo a **função de perda** e a **métrica de avaliação**.

-   **`loss="categorical_crossentropy"`**: Ideal para problemas de classificação multiclasse com rótulos em formato *one-hot*.
-   **`metrics=["accuracy"]`**: A acurácia será monitorada durante o treinamento.

```python
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
```

---

## 6. Treinamento do Modelo

Finalmente, o modelo é treinado com o método `fit`.

-   **`x_train, y_train`**: Dados e rótulos de treinamento.
-   **`epochs=20`**: O modelo verá o conjunto de dados de treinamento 20 vezes.
-   **`verbose=1`**: Exibe uma barra de progresso durante o treinamento.
-   **`validation_data=(x_valid, y_valid)`**: Dados usados para validar o desempenho do modelo ao final de cada época.

```python
model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_valid, y_valid))
```
