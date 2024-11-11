import cv2
import os

# Configurações do classificador de faces
classifier_folder = cv2.data.haarcascades
classifier_file = "haarcascade_frontalface_alt.xml"
face_detector = cv2.CascadeClassifier(os.path.join(classifier_folder, classifier_file))


def capturar_video(centro_callback, video_running):
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)

    while video_running.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Inverter a imagem para facilitar o controle
        frame = cv2.flip(frame, 1)

        # Detecta o rosto na imagem
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        # Processa a detecção de rosto
        if len(faces) > 0:
            # Seleciona o primeiro rosto detectado (pode ajustar para múltiplos rostos se necessário)
            (x, y, w, h) = faces[0]
            centro = (x + w // 2, y + h // 2)

            # Desenha um retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Envia o centro para o callback de controle do paddle
            centro_callback(centro)

        # Exibe a imagem com a detecção de rosto
        cv2.imshow("Detecção de Rosto", frame)

        # Fecha com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running.clear()
            break

    cap.release()
    cv2.destroyAllWindows()
