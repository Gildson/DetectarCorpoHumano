import mediapipe as mp
import cv2

# Variáveis para criar os objetos do mediapipe que irá detectar os ponto do corpo humano
pose_detection = mp.solutions.pose.Pose()

# Objeto do opencv capaz de ler o vídeo
cap = cv2.VideoCapture(r'C:\Users\Gildson\Documents\PoseRecognizier\Yoga advanced asanas.mp4')


while True:
    # Ler os frames do vídeo
    ret, frame = cap.read()

    # Caso o vídeo termine o loop é finalizado
    if not ret:
        break

    # Esse tratamento de exceção é necessário, pois quando o mediapipe não detecta um corpo em qualquer frame do vídeo
    # ele gerar um erro no processo de iteração dos pontos de marcação, assim para não parar a repredução quando a bi-
    # blioteca não enocntrar um corpo é preciso tratar essa exceção.
    try:
        # Detectar o comprimento e largura do vídeo. Esse valores serão utilizados para transformar os pontos de marcação
        # normalizados para o tamanho real do vídeo.
        comprimento, largura, _ = frame.shape

        # Converte de BGR para RGB, pois o opencv tem um padrão diferente do mediapipe para as cores
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar os pontos de marcação do corpo presentes no vídeo
        results = pose_detection.process(frame)

        # Converte de RGB para BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Variável auxiliar para criar o traçado dos pontos de marcação do corpo
        pontos = []

        # Loop para gerar os pontos detectados em cada frame do vídeo.
        for id, ponto in enumerate(results.pose_landmarks.landmark):
            x = int(ponto.x * largura)
            y = int(ponto.y * comprimento)
            pontos.append((x,y))
            cv2.circle(frame, (x, y), 2, (255,255,0), -1)

        # Conecta os pontos detectados do corpo em cada frame
        ponto_conexoes = [[0,5], [0,2], [2,1], [2,3], [2,7], [4,5], [5,6], [5,8], [9,10], [11,12], [11,13], [13,15], [15,21], [15,17], [15,19], [17,19], 
            [11,23], [12,14], [14,16], [16,22], [16,18], [16,20], [18,20], [12,24], [23,24], [23,25], [25,27], [27,31], [27,29], [29,31], [24,26], [26,28],
            [28,32], [28,30], [30,32]
        ]
        for conexao in ponto_conexoes:
            parteA = conexao[0]
            parteB = conexao[1]
            if pontos[parteA] and pontos[parteB]:
                cv2.line(frame, pontos[parteA], pontos[parteB], (255,0,0))
    except Exception as e:
        pass

    # Mostra o vídeo
    cv2.imshow('video', frame)

    # Para finalizar o vídeo apertando a tecla 'c'
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break
    
# Libera os recursos utilizados pelo OpenCV    
cap.release()
cv2.destroyAllWindows()