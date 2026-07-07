import cv2
import os

# =====================================================================
# CONFIGURAÇÕES INICIAIS
# =====================================================================
# Nome da pessoa que está na frente da câmera 
print("Digite o nome da pessoa que deseja capturar:")
input_nome = input().strip()
NOME_PESSOA = input_nome

# Tamanho padrão que definimos para a Álgebra Linear
LARGURA, ALTURA = 92, 112 
QUANTIDADE_FOTOS = 80 # 30 fotos é um bom número para o PCA aprender as variações

# Cria a pasta para salvar as fotos
pasta_destino = f"face_dataset/{NOME_PESSOA}"
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Carrega o detector de rostos do OpenCV
classificador_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

print(f"Preparando para capturar fotos de {NOME_PESSOA}.")
print("Olhe para a câmera. Pressione 'c' para começar a capturar ou 'q' para sair.")

contador_fotos = 0
capturando = False

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        break

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar o rosto
    rostos = classificador_face.detectMultiScale(
        frame_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    # Para cada rosto detectado
    for (x, y, w, h) in rostos:
        # Desenha um retângulo verde para você ver onde ele está focando
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if capturando:
            # Recorta, redimensiona e salva
            rosto_recortado = frame_cinza[y:y+h, x:x+w]
            rosto_redimensionado = cv2.resize(rosto_recortado, (LARGURA, ALTURA))
            
            caminho_arquivo = f"{pasta_destino}/foto_{contador_fotos}.jpg"
            cv2.imwrite(caminho_arquivo, rosto_redimensionado)
            
            contador_fotos += 1
            print(f"Foto {contador_fotos}/{QUANTIDADE_FOTOS} capturada!")
            
            # Pausa super rápida para não tirar todas as fotos no mesmo milissegundo
            cv2.waitKey(100) 

    # Mostra o status na tela
    texto_status = f"Capturadas: {contador_fotos}/{QUANTIDADE_FOTOS}" if capturando else "Aperte 'c' para capturar"
    cv2.putText(frame, texto_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Captura de Dataset", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('c'):
        capturando = True
    elif tecla == ord('q') or contador_fotos >= QUANTIDADE_FOTOS:
        break

print("Captura finalizada com sucesso!")
camera.release()
cv2.destroyAllWindows()