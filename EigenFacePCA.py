import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================================
# 1. FUNÇÕES MATEMÁTICAS (PCA)
# =====================================================================
def treinar_eigenfaces(matriz_imagens):
    rosto_medio = np.mean(matriz_imagens, axis=0)
    X_centralizado = matriz_imagens - rosto_medio
    C = np.dot(X_centralizado, X_centralizado.T)
    
    autovalores, autovetores_aux = np.linalg.eigh(C)
    
    # Ordenar os autovetores do maior para o menor autovalor
    indices_ordenados = np.argsort(autovalores)[::-1]
    autovetores_aux = autovetores_aux[:, indices_ordenados]
    
    eigenfaces = np.dot(X_centralizado.T, autovetores_aux).T
    
    for i in range(eigenfaces.shape[0]):
        norma = np.linalg.norm(eigenfaces[i])
        if norma > 0:
            eigenfaces[i] = eigenfaces[i] / norma
            
    return rosto_medio, eigenfaces, autovalores

def projetar_rosto(rosto_achatado, rosto_medio, eigenfaces):
    rosto_centralizado = rosto_achatado - rosto_medio
    return np.dot(eigenfaces, rosto_centralizado)

# =====================================================================
# 2. CARREGAMENTO DO BANCO DE DADOS (face_dataset)
# =====================================================================
def carregar_dataset(caminho_dataset, largura, altura):
    print("Carregando banco de imagens...")
    imagens = []
    nomes = []
    lista_pastas = os.listdir(caminho_dataset)
    
    for nome_pasta in lista_pastas:
        caminho_pasta_pessoa = os.path.join(caminho_dataset, nome_pasta)
        
        if os.path.isdir(caminho_pasta_pessoa):
            for arquivo_foto in os.listdir(caminho_pasta_pessoa):
                if arquivo_foto.endswith(('.jpg', '.jpeg', '.png')):
                    caminho_foto = os.path.join(caminho_pasta_pessoa, arquivo_foto)
                    
                    # Carregar imagem em escala de cinza
                    img = cv2.imread(caminho_foto, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Garantir o tamanho correto e transformar em vetor 1D
                        img_redimensionada = cv2.resize(img, (largura, altura))
                        imagens.append(img_redimensionada.flatten())
                        nomes.append(nome_pasta)
                        
    return np.array(imagens), nomes

def plotar_variancia(autovalores):
    print("Gerando gráfico de variância...")
    
    # Álgebra: Calculando a porcentagem de informação que cada autovalor carrega
    variancia_explicada = autovalores / np.sum(autovalores)
    variancia_cumulativa = np.cumsum(variancia_explicada)

    # Plotando o gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(variancia_cumulativa, marker='o', linestyle='-', color='b')
    plt.title('Gráfico de Variância Cumulativa (Scree Plot)')
    plt.xlabel('Número de Componentes Principais (Autovetores)')
    plt.ylabel('Proporção da Variância Retida')
    
    # Desenhando uma linha vermelha no marco de 90% de informação
    plt.axhline(y=0.90, color='r', linestyle='--', label='Limiar de 90%')
    plt.grid(True)
    plt.legend()
    plt.show() 
    
def mostrar_eigenfaces(eigenfaces, largura, altura, num_faces=10):
    print("Gerando galeria de Eigenfaces...")
    
    # Cria uma grade de imagens (2 linhas e 5 colunas)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.3, wspace=0.1))
    
    for i, ax in enumerate(axes.flat):
        if i < num_faces and i < len(eigenfaces):
            # Transformação Linear Reversa: Vetor 1D para Matriz 2D
            eigenface_2d = eigenfaces[i].reshape(altura, largura)
            
            # Mostrando a imagem
            ax.imshow(eigenface_2d, cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
            
    plt.suptitle("Galeria de Autovetores (As características de maior variação)", fontsize=16)
    plt.show()
# =====================================================================
# 3. TREINAMENTO E CONFIGURAÇÕES DO SISTEMA
# =====================================================================
LARGURA, ALTURA = 92, 112
CAMINHO_DATASET = "face_dataset"

# Carregar imagens e extrair as Eigenfaces
matriz_treinamento, nomes_treinamento = carregar_dataset(CAMINHO_DATASET, LARGURA, ALTURA)

if len(matriz_treinamento) == 0:
    print("Erro: Nenhuma imagem encontrada na pasta 'face_dataset'.")
    exit()

print("Executando PCA (Álgebra Linear)...")
rosto_medio, eigenfaces, autovalores = treinar_eigenfaces(matriz_treinamento)   

#plotar_variancia(autovalores)
mostrar_eigenfaces(eigenfaces, LARGURA, ALTURA)

# Calcular os 'pesos' de todas as imagens conhecidas no banco de dados
pesos_conhecidos = []
for imagem in matriz_treinamento:
    pesos_conhecidos.append(projetar_rosto(imagem, rosto_medio, eigenfaces))
pesos_conhecidos = np.array(pesos_conhecidos)

# =====================================================================
# 4. RECONHECIMENTO EM TEMPO REAL (WEBCAM)
# =====================================================================
LIMIAR_ACEITACAO = 3000.0  # VOCÊS PRECISARÃO AJUSTAR ESSE NÚMERO NA PRÁTICA!

classificador_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

print("Sistema pronto! Iniciando câmera...")

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        break

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos_detectados = classificador_face.detectMultiScale(
        frame_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    for (x, y, w, h) in rostos_detectados:
        rosto_recortado = frame_cinza[y:y+h, x:x+w]
        rosto_redimensionado = cv2.resize(rosto_recortado, (LARGURA, ALTURA))
        rosto_vetor = rosto_redimensionado.flatten()
        
        # Projetar o rosto da câmera
        pesos_camera = projetar_rosto(rosto_vetor, rosto_medio, eigenfaces)
        
        # Encontrar a menor distância entre o rosto da câmera e o banco de dados
        distancias = np.linalg.norm(pesos_conhecidos - pesos_camera, axis=1)
        indice_menor_distancia = np.argmin(distancias)
        menor_distancia = distancias[indice_menor_distancia]
        
        # Lógica visual de Controle de Acesso
        if menor_distancia < LIMIAR_ACEITACAO:
            # Reconheceu alguém com sucesso! Fica VERDE.
            cor = (0, 255, 0) 
            nome_reconhecido = nomes_treinamento[indice_menor_distancia]
            texto = f"Acesso liberado: {nome_reconhecido}"
        else:
            # Não reconheceu. Fica VERMELHO.
            cor = (0, 0, 255) 
            texto = "Acesso negado / Desconhecido"
            
        # Desenhar na tela
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        # Opcional: mostrar a distância matemática para ajudar a calibrar o Limiar
        cv2.putText(frame, f"Dist: {menor_distancia:.0f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

    cv2.imshow("Sistema de Reconhecimento - Eigenfaces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()