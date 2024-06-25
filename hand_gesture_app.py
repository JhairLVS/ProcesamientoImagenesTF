import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Inicializar Mediapipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar los clasificadores Haar Cascade para la detección de rostros y sonrisas
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Función para reconocer gestos básicos
def recognize_gesture(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y:
        return "Pulgar Arriba"
    else:
        return "Palma Abierta"

# Función para aplicar filtros de realce de imagen sin distorsionar colores
def enhance_image(image):
    # Aplicar suavizado
    enhanced = cv2.bilateralFilter(image, 9, 75, 75)
    return enhanced

def detect_face_and_smile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            return "Sonriendo"
    return "No Sonriendo"

# Clase principal de la aplicación
class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Gestos de Mano y Detección de Rostros")
        self.root.geometry("800x600")
        self.root.configure(bg='lightblue')
        
        # Estilos de ttk
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12), padding=10)
        style.configure('TLabel', font=('Arial', 14), padding=10)
        
        # Configurar el panel de video
        self.video_panel = ttk.Label(self.root, text="Panel de Video", background='white', relief='sunken', anchor='center')
        self.video_panel.grid(row=0, column=0, padx=10, pady=10, columnspan=2, sticky='nsew')
        
        # Botón para iniciar la captura de video
        self.start_button = ttk.Button(self.root, text="Iniciar Video", command=self.start_video)
        self.start_button.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        
        # Botón para detener la captura de video
        self.stop_button = ttk.Button(self.root, text="Detener Video", command=self.stop_video)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
        
        # Etiqueta para mostrar el estado
        self.status_label = ttk.Label(self.root, text="Estado: Esperando", background='lightblue', anchor='center')
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        # Configuración para expandir el panel de video
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        self.cap = None
        self.running = False
    
    def start_video(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.status_label.config(text="Estado: Iniciando video...")
            self.process_video()
    
    def stop_video(self):
        if self.running:
            self.running = False
            self.cap.release()
            self.video_panel.config(image='')
            self.status_label.config(text="Estado: Video detenido")
    
    def process_video(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Estado: Error en la captura de video")
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:
            
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture = recognize_gesture(hand_landmarks)
                    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Detectar rostro y sonrisa
        face_expression = detect_face_and_smile(frame)
        cv2.putText(frame, face_expression, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        enhanced_frame = enhance_image(frame)
        frame_pil = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        
        self.video_panel.config(image=frame_tk)
        self.video_panel.image = frame_tk
        self.status_label.config(text="Estado: Detectando gestos y sonrisas...")
        
        self.root.after(10, self.process_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
