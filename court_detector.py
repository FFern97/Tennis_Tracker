import cv2
import numpy as np
import torch
import torchvision.transforms as T
from geometry_utils import get_trans_matrix, refer_kps


def get_coords_from_heatmap(heatmap, low_thresh=150,
                           scale_to_original=True, original_w=None, original_h=None, 
                           heatmap_w=640, heatmap_h=360):
    """
    Encuentra las coordenadas (x, y) desde un heatmap usando thresholding y HoughCircles.
    Opcionalmente escala las coordenadas de vuelta a las dimensiones originales de la imagen.
    
    Args:
        heatmap: Heatmap de un solo canal (uint8)
        low_thresh: Umbral para binarización
        scale_to_original: Si True, escala las coordenadas a las dimensiones originales
        original_w: Ancho de la imagen original
        original_h: Alto de la imagen original
        heatmap_w: Ancho del heatmap
        heatmap_h: Alto del heatmap
    
    Returns:
        x_pred, y_pred: Coordenadas detectadas (pueden ser None si no se detectó nada)
    """
    x_pred, y_pred = None, None
    try:
        _, binary_heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(binary_heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                   param1=50, param2=2, 
                                   minRadius=2, maxRadius=20) 
        if circles is not None and len(circles) > 0:
            x_heatmap = circles[0][0][0]
            y_heatmap = circles[0][0][1]
            if scale_to_original and original_w is not None and original_h is not None:
                scale_x = original_w / heatmap_w
                scale_y = original_h / heatmap_h
                x_pred = x_heatmap * scale_x
                y_pred = y_heatmap * scale_y
            else:
                x_pred = x_heatmap
                y_pred = y_heatmap
    except Exception as e:
        print(f"Error processing heatmap: {e}") 
    return x_pred, y_pred


class CourtDetector:
    """
    Clase para detectar y procesar los puntos clave de la cancha de tenis.
    Maneja la detección, promediado y corrección con homografía.
    """
    
    def __init__(self, keypoint_model, device, keypoint_input_width=640, 
                 keypoint_input_height=360, n_frames_to_average=5):
        """
        Args:
            keypoint_model: Modelo entrenado para detectar puntos clave
            device: Dispositivo (cuda o cpu)
            keypoint_input_width: Ancho de entrada para el modelo de puntos clave
            keypoint_input_height: Alto de entrada para el modelo de puntos clave
            n_frames_to_average: Número de frames a promediar antes de aplicar homografía
        """
        self.keypoint_model = keypoint_model
        self.device = device
        self.keypoint_input_width = keypoint_input_width
        self.keypoint_input_height = keypoint_input_height
        self.n_frames_to_average = n_frames_to_average
        
        self.keypoint_preprocess = T.Compose([T.ToTensor()])
        self.keypoint_accumulator = [[] for _ in range(14)]
        self.stored_keypoints = None
        self.homography_matrix = None  # Almacenar la matriz de homografía
        self.frame_count = 0
    
    def detect_keypoints(self, frame, frame_width, frame_height):
        """
        Detecta los puntos clave en un frame y los acumula para promediado.
        
        Args:
            frame: Frame de video (BGR)
            frame_width: Ancho del frame original
            frame_height: Alto del frame original
        
        Returns:
            stored_keypoints: Lista de 14 puntos clave (o None si aún no se han procesado suficientes frames)
        """
        self.frame_count += 1
        
        if self.frame_count <= self.n_frames_to_average:
            print(f"Detectando puntos clave (frame {self.frame_count}/{self.n_frames_to_average})...")
            img_resized = cv2.resize(frame, (self.keypoint_input_width, self.keypoint_input_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_tensor = self.keypoint_preprocess(img_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.keypoint_model(input_tensor)

            heatmaps = torch.sigmoid(output_tensor).squeeze(0).cpu().numpy()

            for i in range(14):
                heatmap_uint8 = (heatmaps[i] * 255).astype(np.uint8)
                x, y = get_coords_from_heatmap(heatmap_uint8, low_thresh=150,
                                               scale_to_original=True,
                                               original_w=frame_width, original_h=frame_height,
                                               heatmap_w=self.keypoint_input_width, 
                                               heatmap_h=self.keypoint_input_height)
                if x is not None and y is not None:
                    self.keypoint_accumulator[i].append((x, y))

            if self.frame_count == self.n_frames_to_average:
                print("Calculando promedio de puntos clave...")
                averaged_keypoints = []
                for i in range(14):
                    points_for_kp = self.keypoint_accumulator[i]
                    if points_for_kp:
                        avg_x = np.mean([p[0] for p in points_for_kp])
                        avg_y = np.mean([p[1] for p in points_for_kp])
                        averaged_keypoints.append((avg_x, avg_y))
                    else:
                        averaged_keypoints.append((None, None))
                
                # Guardar los puntos promediados (sin homografía todavía)
                self.stored_keypoints = averaged_keypoints
                print("Puntos clave promediados calculados.")
        
        return self.stored_keypoints
    
    def get_keypoints(self):
        """
        Retorna los puntos clave almacenados.
        
        Returns:
            stored_keypoints: Lista de 14 puntos clave o None
        """
        return self.stored_keypoints
    
    def apply_homography(self, keypoints):
        """
        Aplica corrección de homografía a los puntos clave dados.
        
        Args:
            keypoints: Lista de 14 puntos clave [(x, y), ...]
        
        Returns:
            final_keypoints: Lista de 14 puntos clave corregidos con homografía
        """
        if keypoints is None:
            return None
        
        # Convertir keypoints a formato de lista de tuplas si viene como array numpy
        formatted_keypoints = []
        for kp in keypoints:
            if isinstance(kp, np.ndarray):
                if len(kp) == 2 and kp[0] is not None and kp[1] is not None:
                    formatted_keypoints.append((float(kp[0]), float(kp[1])))
                else:
                    formatted_keypoints.append((None, None))
            else:
                formatted_keypoints.append(kp)
        
        # --- Aplicar Corrección de Homografía ---
        matrix_trans = get_trans_matrix(formatted_keypoints)
        
        if matrix_trans is not None:
            print("Homografía encontrada. Aplicando corrección global...")
            # Almacenar la matriz de homografía
            self.homography_matrix = matrix_trans
            # Aplicar la matriz a la plantilla de referencia 'refer_kps'
            final_points_transformed = cv2.perspectiveTransform(refer_kps, matrix_trans)
            
            # Formatear el resultado
            final_keypoints = []
            if final_points_transformed is not None:
                for point in final_points_transformed:
                    final_keypoints.append((point[0][0], point[0][1]))
            else:
                print("Error en perspectiveTransform. Usando puntos sin corrección.")
                final_keypoints = formatted_keypoints  # Fallback
        else:
            print("Homografía no encontrada. Usando puntos sin corrección global.")
            final_keypoints = formatted_keypoints  # Fallback
        
        return final_keypoints
    
    def get_homography_matrix(self):
        """
        Retorna la matriz de homografía almacenada.
        
        Returns:
            homography_matrix: Matriz de homografía 3x3 o None
        """
        return self.homography_matrix
    
    def refine_keypoints(self, frame, keypoints, window_size=20):
        """
        Refina los puntos clave usando detección de esquinas en una ventana local.
        
        Args:
            frame: Frame de video (BGR)
            keypoints: Lista de puntos clave a refinar [(x, y), ...]
            window_size: Tamaño de la ventana de búsqueda alrededor de cada punto
        
        Returns:
            refined_kps: Array numpy con los puntos refinados
        """
        refined_kps = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        for x, y in keypoints:
            if x is None or y is None:
                refined_kps.append([None, None])
                continue
                
            x_min = max(0, int(x - window_size))
            x_max = min(frame.shape[1], int(x + window_size))
            y_min = max(0, int(y - window_size))
            y_max = min(frame.shape[0], int(y + window_size))
            roi = binary[y_min:y_max, x_min:x_max]
            
            corners = cv2.goodFeaturesToTrack(roi, maxCorners=1, qualityLevel=0.05, minDistance=5)
            
            if corners is not None and len(corners) > 0:
                refined_x = corners[0][0][0] + x_min
                refined_y = corners[0][0][1] + y_min
                refined_kps.append([refined_x, refined_y])
            else:
                refined_kps.append([x, y])
        
        return np.array(refined_kps)

