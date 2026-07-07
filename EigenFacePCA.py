import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

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

def dividir_treino_teste(imagens, nomes, proporcao_teste=0.2):
    por_pessoa = defaultdict(list)
    for img, nome in zip(imagens, nomes):
        por_pessoa[nome].append(img)
    
    treino_imgs, treino_nomes = [], []
    teste_imgs, teste_nomes = [], []
    for nome, fotos in por_pessoa.items():
        n_teste = int(len(fotos) * proporcao_teste)
        teste_imgs.extend(fotos[:n_teste])
        teste_nomes.extend([nome] * n_teste)
        treino_imgs.extend(fotos[n_teste:])
        treino_nomes.extend([nome] * (len(fotos) - n_teste))
    
    return np.array(treino_imgs), treino_nomes, np.array(teste_imgs), teste_nomes

def avaliar_acuracia_otimizada(pesos_treino_total, nomes_treinamento, pesos_teste_total, nomes_teste, k):
    # Fatiamos os pesos para pegar apenas as 'k' primeiras colunas
    pesos_treino_k = pesos_treino_total[:, :k]
    pesos_teste_k = pesos_teste_total[:, :k]

    acertos = 0
    for i, pesos_t in enumerate(pesos_teste_k):
        distancias = np.linalg.norm(pesos_treino_k - pesos_t, axis=1)
        indice_mais_proximo = np.argmin(distancias)
        if nomes_treinamento[indice_mais_proximo] == nomes_teste[i]:
            acertos += 1

    return acertos / len(nomes_teste)


def plotar_acuracia_vs_k(eigenfaces, rosto_medio, matriz_treinamento, nomes_treinamento, matriz_teste, nomes_teste):
    valores_k = list(range(1, len(eigenfaces) + 1))
    acuracias = []

    print("\nPré-calculando projeções para acelerar o processo...")
    # Faz a projeção pesada UMA ÚNICA VEZ antes do loop!
    pesos_treino_total = np.array([projetar_rosto(img, rosto_medio, eigenfaces) for img in matriz_treinamento])
    pesos_teste_total = np.array([projetar_rosto(img, rosto_medio, eigenfaces) for img in matriz_teste])

    for k in valores_k:
        print(f"Avaliando k={k}/{len(eigenfaces)}...", end='\r')
        acc = avaliar_acuracia_otimizada(pesos_treino_total, nomes_treinamento, pesos_teste_total, nomes_teste, k)
        acuracias.append(acc)

    print("\nAvaliação concluída!")
    
    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, acuracias, marker='o', linestyle='-', color='g')
    plt.title('Acurácia vs. Número de Componentes Principais (k)')
    plt.xlabel('Número de Componentes (k)')
    plt.ylabel('Acurácia no Conjunto de Teste')
    plt.grid(True)
    plt.show()

    melhor_k = valores_k[int(np.argmax(acuracias))]
    print(f"Melhor k encontrado: {melhor_k} (acurácia = {max(acuracias)*100:.1f}%)")
    return melhor_k, acuracias
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

# Carregar todas as imagens do dataset
todas_imagens, todos_nomes = carregar_dataset(CAMINHO_DATASET, LARGURA, ALTURA)

if len(todas_imagens) == 0:
    print("Erro: Nenhuma imagem encontrada na pasta 'face_dataset'.")
    exit()

# Separar em treino (usado pra calcular eigenfaces) e teste (nunca visto no treino)
matriz_treinamento, nomes_treinamento, matriz_teste, nomes_teste = dividir_treino_teste(
    todas_imagens, todos_nomes, proporcao_teste=0.2
)

print(f"Treino: {len(matriz_treinamento)} fotos | Teste: {len(matriz_teste)} fotos")

print("Executando PCA (Álgebra Linear)...")
rosto_medio, eigenfaces, autovalores = treinar_eigenfaces(matriz_treinamento)

#plotar_variancia(autovalores)
#mostrar_eigenfaces(eigenfaces, LARGURA, ALTURA)

# Calcular os 'pesos' de todas as imagens de TREINO (banco de dados conhecido)
pesos_conhecidos = []
for imagem in matriz_treinamento:
    pesos_conhecidos.append(projetar_rosto(imagem, rosto_medio, eigenfaces))
pesos_conhecidos = np.array(pesos_conhecidos)

# =====================================================================
# 3.1 AVALIAÇÃO FORMAL: ACURÁCIA EM FUNÇÃO DO NÚMERO DE COMPONENTES (k)
# =====================================================================
print("Avaliando acurácia para diferentes valores de k...")
melhor_k_grafico, acuracias_por_k = plotar_acuracia_vs_k(
    eigenfaces, rosto_medio, matriz_treinamento, nomes_treinamento, matriz_teste, nomes_teste
)

# CORREÇÃO: Forçar um k maior para garantir captura de detalhes reais do rosto
# Em PCA facial, k entre 30 e 50 costuma ser o "ponto doce" para precisão e velocidade
melhor_k = 60 

# Se você tiver menos de 40 fotos de treino, limite o k ao máximo que você tem
if melhor_k > len(eigenfaces):
    melhor_k = len(eigenfaces)

print(f"Ignorando k={melhor_k_grafico} do gráfico. Usando k={melhor_k} para maior segurança.")

# A partir daqui, o sistema (webcam) usa o k robusto
eigenfaces_usadas = eigenfaces[:melhor_k]
pesos_conhecidos = np.array([
    projetar_rosto(img, rosto_medio, eigenfaces_usadas) for img in matriz_treinamento
])
# =====================================================================
# 4. RECONHECIMENTO EM TEMPO REAL (WEBCAM)
# =====================================================================
LIMIAR_ACEITACAO = 3000.0  # AJUSTAR ESSE NÚMERO NA PRÁTICA!

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
        pesos_camera = projetar_rosto(rosto_vetor, rosto_medio, eigenfaces_usadas)
        
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