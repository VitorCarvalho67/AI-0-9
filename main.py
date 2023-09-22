import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import keras
from sklearn.model_selection import train_test_split
from PIL import Image
import pygame
import sys
import time

x = []
y = []
digitos = 10
registros = 200

for digito in range(digitos):
    for registro in range(registros):
        valores = []
        y.append(digito)
        arquivo = f'data/n{digito}/n{digito}_{registro + 1}.png'
        imagem = Image.open(arquivo)
        largura, altura = imagem.size
        for h in range(altura):
            for w in range(largura):
                pixel = imagem.getpixel((w, h))
                valor = '#{0:02x}{1:02x}{2:02x}'.format(pixel[0], pixel[1], pixel[2])
                if valor == '#000000':
                    valor = 0
                else:
                    valor = 1
                valores.append(valor)
        x.append(valores)

# Converter em arrays numpy
x = np.array(x)
y = np.array(y)

# Normalizar os valores dos pixels para o intervalo [0, 1]
y = to_categorical(y, num_classes=10)

# Dividir os dados em treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Reformular os dados de treinamento e teste para terem o formato correto
x_train = x_train.reshape(-1, 64, 64, 1)
x_test = x_test.reshape(-1, 64, 64, 1)

# Certifique-se de que os dados estejam no tipo float32 e normalizados
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# agora usando o ayers.Dropout para evitar overfitting {melhor modelo até agora}
model = Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=200, batch_size=128)

# modelo para evitar overfitting com o melhor numero de epocas e batch_size
model.fit(x_train, y_train, epochs=100, batch_size=128)

model.save('model.h5')

# Avaliar o desempenho do modelo nos dados de teste
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')

SCREEN_WIDTH = 64
SCREEN_HEIGHT = 64
PIXEL_SIZE = 4

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH * PIXEL_SIZE, SCREEN_HEIGHT * PIXEL_SIZE))
pygame.display.set_caption("Desenho")
drawing_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
drawing_surface.fill((0, 0, 0))
BLACK = (255, 255, 255)
drawing = False

PEN_SIZE = 3
last_save_time = 0 

while True:
    current_time = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                x, y = event.pos
                x //= PIXEL_SIZE
                y //= PIXEL_SIZE
                pygame.draw.rect(drawing_surface, BLACK, (x, y, PEN_SIZE, PEN_SIZE))

    pygame.transform.scale(drawing_surface, (SCREEN_WIDTH * PIXEL_SIZE, SCREEN_HEIGHT * PIXEL_SIZE), screen)
    pygame.display.flip()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_s] and current_time - last_save_time >= 2:
        teste = []
        last_save_time = current_time
        pygame.image.save(drawing_surface, 'n.png')
        drawing_surface.fill((0, 0, 0))
        numero = []
        valores = []
        imagem = Image.open('n.png')
        largura, altura = imagem.size
        for h in range(altura):
            for w in range(largura):
                pixel = imagem.getpixel((w, h))
                valor = '#{0:02x}{1:02x}{2:02x}'.format(pixel[0], pixel[1], pixel[2])
                if valor == '#000000':
                    valor = 0
                else:
                    valor = 1
                valores.append(valor)
        numero.append(valores)
        teste.append(numero)

        teste = np.array(teste)
        teste = teste.reshape(-1, 64, 64, 1)

        teste = teste.astype('float32') / 255.0

        predictions = model.predict(np.array([teste[0]]))
        # predicted_digit = np.argmax(predictions)
        # print(f'Dígito previsto: {predicted_digit}')
        # mostrar a porcentagem de cada digito
        print(predictions)
        print(np.argmax(predictions))
        print(f'Dígito previsto: {np.argmax(predictions)}')
        print(f'Acurácia: {np.max(predictions) * 100:.2f}%')
        

    pygame.time.delay(10)