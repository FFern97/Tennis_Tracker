import cv2
import numpy as np
import os
from geometry_utils import court_ref


def draw_keypoints(frame, keypoints):
    """
    Dibuja los puntos clave de la cancha en el frame.
    
    Args:
        frame: Frame de video (numpy array)
        keypoints: Array de puntos clave (N, 2) con coordenadas (x, y)
    
    Returns:
        frame: Frame con los puntos clave dibujados
    """
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)
    
    return frame


def draw_ball_detections(frame, ball_results):
    """
    Dibuja las detecciones de la pelota en el frame.
    
    Args:
        frame: Frame de video (numpy array)
        ball_results: Resultados de YOLO para la pelota
    
    Returns:
        frame: Frame con las detecciones de la pelota dibujadas
    """
    if ball_results and ball_results[0].boxes is not None:
        for box in ball_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf.item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame


def draw_person_tracking(frame, person_results):
    """
    Dibuja el tracking de personas con esqueletos en el frame.
    
    Args:
        frame: Frame de video (numpy array)
        person_results: Resultados de YOLO-pose para personas
    
    Returns:
        frame: Frame con el tracking de personas dibujado
    """
    if person_results and len(person_results) > 0:
        result = person_results[0]
        
        # Dibujar bounding boxes
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Dibujar esqueletos (keypoints)
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            # Definir conexiones del esqueleto (muñecas, codos, hombros)
            # Índices de keypoints de COCO: 0=nariz, 1=ojo_izq, 2=ojo_der, 3=oreja_izq, 4=oreja_der,
            # 5=hombro_izq, 6=hombro_der, 7=codo_izq, 8=codo_der, 9=muñeca_izq, 10=muñeca_der
            skeleton_connections = [
                (5, 7),   # Hombro izquierdo -> Codo izquierdo
                (7, 9),   # Codo izquierdo -> Muñeca izquierda
                (6, 8),   # Hombro derecho -> Codo derecho
                (8, 10),  # Codo derecho -> Muñeca derecha
                (5, 6),   # Hombro izquierdo -> Hombro derecho
            ]
            
            cyan_color = (255, 255, 0)  # Cian en BGR
            line_thickness = 2
            
            # Iterar sobre cada persona detectada
            for person_idx in range(len(result.keypoints.data)):
                keypoints = result.keypoints.data[person_idx]
                
                # Dibujar conexiones del esqueleto
                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        start_kp = keypoints[start_idx]
                        end_kp = keypoints[end_idx]
                        
                        # Verificar que los keypoints sean visibles (confianza > 0)
                        if start_kp[2] > 0 and end_kp[2] > 0:
                            start_x, start_y = int(start_kp[0]), int(start_kp[1])
                            end_x, end_y = int(end_kp[0]), int(end_kp[1])
                            cv2.line(frame, (start_x, start_y), (end_x, end_y), cyan_color, line_thickness)
                
                # Dibujar keypoints individuales (muñecas y codos)
                keypoint_indices = [7, 8, 9, 10]  # Codos y muñecas
                for kp_idx in keypoint_indices:
                    if kp_idx < len(keypoints):
                        kp = keypoints[kp_idx]
                        if kp[2] > 0:  # Visible
                            kp_x, kp_y = int(kp[0]), int(kp[1])
                            cv2.circle(frame, (kp_x, kp_y), 5, cyan_color, -1)
    
    return frame


# Conexiones del esqueleto COCO (hombros, codos, muñecas)
SKELETON_CONNECTIONS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]
KEYPOINT_INDICES = [7, 8, 9, 10]
CYAN_BGR = (255, 255, 0)


def draw_players_from_info(frame, players_info):
    """
    Dibuja jugadores a partir de PlayersInfo (active_tracks: px + keypoints).
    Dibuja círculo en posición pies y esqueleto si hay keypoints.
    """
    if players_info is None or not getattr(players_info, "active_tracks", None):
        return frame
    for track_id, data in players_info.active_tracks.items():
        px = data.get("px")
        if px is None:
            continue
        cx, cy = int(px[0]), int(px[1])
        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (cx - 20, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        kps = data.get("keypoints")
        if kps is not None and len(kps) > 0:
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx < len(kps) and end_idx < len(kps):
                    s, e = kps[start_idx], kps[end_idx]
                    if s[2] > 0 and e[2] > 0:
                        cv2.line(frame, (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), CYAN_BGR, 2)
            for kp_idx in KEYPOINT_INDICES:
                if kp_idx < len(kps) and kps[kp_idx][2] > 0:
                    x, y = int(kps[kp_idx][0]), int(kps[kp_idx][1])
                    cv2.circle(frame, (x, y), 5, CYAN_BGR, -1)
    return frame


def draw_minimap(frame, homography_matrix, person_results, ball_results,
                 minimap_size=200, position='top-right', inv_homography=None,
                 ball_position=None, trajectory_history=None,
                 player_positions_px=None, is_ball_interpolated=False):
    """
    Dibuja un mini-mapa 2D de la cancha con jugadores y pelota.
    Si se pasa player_positions_px (lista de (x,y)), se usa en lugar de person_results para posiciones.
    """
    if homography_matrix is None:
        return frame
    
    # Calcular homografía inversa si no se proporciona
    if inv_homography is None:
        inv_homography = np.linalg.inv(homography_matrix)
    
    # Definir márgenes para expandir el área de visión
    # La plantilla tiene coordenadas que van aproximadamente de (286, 561) a (1379, 2935)
    template_min_x = 286
    template_max_x = 1379
    template_min_y = 561
    template_max_y = 2935
    template_width = template_max_x - template_min_x
    template_height = template_max_y - template_min_y
    
    # Márgenes: 25% del ancho lateral, 25% del alto para el fondo
    lateral_margin_percent = 0.25
    back_margin_percent = 0.25
    
    expanded_min_x = template_min_x - (template_width * lateral_margin_percent)
    expanded_max_x = template_max_x + (template_width * lateral_margin_percent)
    expanded_min_y = template_min_y - (template_height * back_margin_percent) # Margen superior
    expanded_max_y = template_max_y + (template_height * back_margin_percent) # Margen inferior
    
    expanded_width = expanded_max_x - expanded_min_x
    expanded_height = expanded_max_y - expanded_min_y
    
    # Función para escalar coordenadas del espacio expandido al tamaño del mini-mapa
    def scale_to_minimap(x, y):
        # Normalizar coordenadas del espacio expandido
        norm_x = (x - expanded_min_x) / expanded_width
        norm_y = (y - expanded_min_y) / expanded_height
        # Escalar al tamaño del mini-mapa y asegurar que esté dentro de los límites visuales
        return int(max(0, min(minimap_size - 1, norm_x * minimap_size))), \
               int(max(0, min(minimap_size - 1, norm_y * minimap_size)))
    
    # Crear el mini-mapa (fondo verde oscuro para el área expandida)
    minimap = np.zeros((minimap_size, minimap_size, 3), dtype=np.uint8)
    minimap[:] = (0, 80, 0)  # Verde oscuro para el área exterior
    
    # Dibujar el área de la cancha con un verde más claro
    court_minimap_x1, court_minimap_y1 = scale_to_minimap(template_min_x, template_min_y)
    court_minimap_x2, court_minimap_y2 = scale_to_minimap(template_max_x, template_max_y)
    cv2.rectangle(minimap, (court_minimap_x1, court_minimap_y1),
                 (court_minimap_x2, court_minimap_y2), (0, 100, 0), -1) # Verde normal para la cancha
    
    # Dibujar líneas de la cancha en el mini-mapa
    line_color = (255, 255, 255)
    line_thickness = 1
    
    # Baseline superior
    pt1 = scale_to_minimap(*court_ref.baseline_top[0])
    pt2 = scale_to_minimap(*court_ref.baseline_top[1])
    cv2.line(minimap, pt1, pt2, line_color, line_thickness)
    
    # Baseline inferior
    pt1 = scale_to_minimap(*court_ref.baseline_bottom[0])
    pt2 = scale_to_minimap(*court_ref.baseline_bottom[1])
    cv2.line(minimap, pt1, pt2, line_color, line_thickness)
    
    # Líneas laterales
    pt1 = scale_to_minimap(*court_ref.left_court_line[0])
    pt2 = scale_to_minimap(*court_ref.left_court_line[1])
    cv2.line(minimap, pt1, pt2, line_color, line_thickness)
    
    pt1 = scale_to_minimap(*court_ref.right_court_line[0])
    pt2 = scale_to_minimap(*court_ref.right_court_line[1])
    cv2.line(minimap, pt1, pt2, line_color, line_thickness)
    
    # Líneas internas
    pt1 = scale_to_minimap(*court_ref.left_inner_line[0])
    pt2 = scale_to_minimap(*court_ref.left_inner_line[1])
    cv2.line(minimap, pt1, pt2, line_color, 1)
    
    pt1 = scale_to_minimap(*court_ref.right_inner_line[0])
    pt2 = scale_to_minimap(*court_ref.right_inner_line[1])
    cv2.line(minimap, pt1, pt2, line_color, 1)
    
    pt1 = scale_to_minimap(*court_ref.top_inner_line[0])
    pt2 = scale_to_minimap(*court_ref.top_inner_line[1])
    cv2.line(minimap, pt1, pt2, line_color, 1)
    
    pt1 = scale_to_minimap(*court_ref.bottom_inner_line[0])
    pt2 = scale_to_minimap(*court_ref.bottom_inner_line[1])
    cv2.line(minimap, pt1, pt2, line_color, 1)
    
    pt1 = scale_to_minimap(*court_ref.middle_line[0])
    pt2 = scale_to_minimap(*court_ref.middle_line[1])
    cv2.line(minimap, pt1, pt2, line_color, 1)
    
    # Red de la cancha
    pt1 = scale_to_minimap(*court_ref.net[0])
    pt2 = scale_to_minimap(*court_ref.net[1])
    cv2.line(minimap, pt1, pt2, line_color, line_thickness)
    
    # Dibujar estela de la pelota (trayectoria de los últimos 10 frames)
    if trajectory_history is not None and len(trajectory_history) >= 2 and inv_homography is not None:
        trail_points = []
        for frame_num, x, y in trajectory_history:
            # Transformar del frame al espacio 2D
            point_frame = np.array([[[x, y]]], dtype=np.float32)
            point_2d = cv2.perspectiveTransform(point_frame, inv_homography)
            
            if point_2d is not None and len(point_2d) > 0:
                x_2d, y_2d = point_2d[0][0]
                # NO FILTRAR: Mostrar todos los puntos de la estela (scale_to_minimap ya hace clip)
                minimap_x, minimap_y = scale_to_minimap(x_2d, y_2d)
                trail_points.append((minimap_x, minimap_y))
        
        # Dibujar línea tenue conectando los puntos de la estela
        if len(trail_points) >= 2:
            for i in range(len(trail_points) - 1):
                # Hacer la línea más tenue (menos opaca) para los puntos más antiguos
                alpha = 0.3 + (i / len(trail_points)) * 0.4  # De 0.3 a 0.7
                color_intensity = int(255 * alpha)
                trail_color = (0, color_intensity, color_intensity)  # Amarillo con transparencia variable
                cv2.line(minimap, trail_points[i], trail_points[i + 1], trail_color, 1)
    
    # Transformar posiciones de jugadores del frame al espacio 2D
    if player_positions_px is not None:
        for (center_x, feet_y) in player_positions_px:
            if center_x is None or feet_y is None:
                continue
            point_frame = np.array([[[center_x, feet_y]]], dtype=np.float32)
            point_2d = cv2.perspectiveTransform(point_frame, inv_homography)
            if point_2d is not None and len(point_2d) > 0:
                x_2d, y_2d = point_2d[0][0]
                minimap_x, minimap_y = scale_to_minimap(x_2d, y_2d)
                cv2.circle(minimap, (minimap_x, minimap_y), 5, (255, 0, 0), -1)
    elif person_results and person_results[0].boxes is not None:
        for box in person_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2.0
            feet_y = y2
            point_frame = np.array([[[center_x, feet_y]]], dtype=np.float32)
            point_2d = cv2.perspectiveTransform(point_frame, inv_homography)
            if point_2d is not None and len(point_2d) > 0:
                x_2d, y_2d = point_2d[0][0]
                minimap_x, minimap_y = scale_to_minimap(x_2d, y_2d)
                cv2.circle(minimap, (minimap_x, minimap_y), 5, (255, 0, 0), -1)
    
    # Transformar posición de la pelota del frame al espacio 2D
    ball_center = None
    if ball_position is not None:
        ball_center = ball_position
    elif ball_results and ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
        # Obtener el centro del bounding box de la detección
        box = ball_results[0].boxes[0]
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        ball_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        is_ball_interpolated = False
    
    if ball_center is not None:
        center_x, center_y = ball_center
        # Transformar del frame al espacio 2D usando la homografía inversa
        point_frame = np.array([[[center_x, center_y]]], dtype=np.float32)
        point_2d = cv2.perspectiveTransform(point_frame, inv_homography)
        
        if point_2d is not None and len(point_2d) > 0:
            x_2d, y_2d = point_2d[0][0]
            # NO FILTRAR: Mostrar todos los puntos, incluso fuera de la cancha (scale_to_minimap ya hace clip)
            minimap_x, minimap_y = scale_to_minimap(x_2d, y_2d)
            # Dibujar pelota (círculo amarillo, más claro si es interpolada)
            if is_ball_interpolated:
                cv2.circle(minimap, (minimap_x, minimap_y), 4, (0, 200, 255), -1)  # Amarillo más claro
            else:
                cv2.circle(minimap, (minimap_x, minimap_y), 4, (0, 255, 255), -1)  # Amarillo normal
    
    # Superponer el mini-mapa en el frame original
    frame_height, frame_width = frame.shape[:2]
    
    # Calcular posición del mini-mapa
    if position == 'top-right':
        x_offset = frame_width - minimap_size - 10
        y_offset = 10
    elif position == 'top-left':
        x_offset = 10
        y_offset = 10
    elif position == 'bottom-right':
        x_offset = frame_width - minimap_size - 10
        y_offset = frame_height - minimap_size - 10
    elif position == 'bottom-left':
        x_offset = 10
        y_offset = frame_height - minimap_size - 10
    else:
        x_offset = frame_width - minimap_size - 10
        y_offset = 10
    
    # Extraer región de interés del frame
    roi = frame[y_offset:y_offset+minimap_size, x_offset:x_offset+minimap_size]
    
    # Mezclar el mini-mapa con el frame (transparencia)
    alpha = 0.7  # Opacidad del mini-mapa
    beta = 1.0 - alpha
    blended = cv2.addWeighted(roi, beta, minimap, alpha, 0)
    
    # Colocar el mini-mapa en el frame
    frame[y_offset:y_offset+minimap_size, x_offset:x_offset+minimap_size] = blended
    
    # Dibujar un borde alrededor del mini-mapa
    cv2.rectangle(frame, (x_offset-1, y_offset-1), 
                  (x_offset+minimap_size, y_offset+minimap_size), 
                  (255, 255, 255), 2)
    
    return frame


def save_final_heatmap(homography_matrix, inv_homography, bounce_history, video_out_path):
    """
    Genera un heatmap final (mapa de tiros) con todos los piques detectados.
    
    Args:
        homography_matrix: Matriz de homografía (de plantilla 2D a frame)
        inv_homography: Matriz de homografía inversa
        bounce_history: Lista de posiciones (x, y) de piques detectados
        video_out_path: Ruta del video de salida (para generar nombre del heatmap)
    """
    if homography_matrix is None or inv_homography is None:
        print("[Heatmap] No se puede generar heatmap: falta matriz de homografía")
        return
    
    if not bounce_history:
        print("[Heatmap] No hay piques detectados para generar el heatmap")
        return
    
    # Tamaño del heatmap (más grande que el mini-mapa para mejor resolución)
    heatmap_size = 800
    
    # Obtener las dimensiones de la plantilla de referencia
    template_min_x = 286
    template_max_x = 1379
    template_min_y = 561
    template_max_y = 2935
    template_width = template_max_x - template_min_x
    template_height = template_max_y - template_min_y
    
    # Definir márgenes expandidos (25% del ancho/alto de la cancha)
    MARGIN_PERCENTAGE = 0.25
    lateral_margin = template_width * MARGIN_PERCENTAGE
    depth_margin = template_height * MARGIN_PERCENTAGE
    
    # Calcular límites expandidos
    expanded_min_x = template_min_x - lateral_margin
    expanded_max_x = template_max_x + lateral_margin
    expanded_min_y = template_min_y - depth_margin
    expanded_max_y = template_max_y + depth_margin
    expanded_width = expanded_max_x - expanded_min_x
    expanded_height = expanded_max_y - expanded_min_y
    
    # Crear el heatmap con área expandida
    heatmap = np.zeros((heatmap_size, heatmap_size, 3), dtype=np.uint8)
    
    # Dibujar área exterior (márgenes) con verde más oscuro
    heatmap[:] = (0, 80, 0)
    
    # Dibujar área interior (cancha) con verde normal
    court_min_x_norm = (template_min_x - expanded_min_x) / expanded_width
    court_max_x_norm = (template_max_x - expanded_min_x) / expanded_width
    court_min_y_norm = (template_min_y - expanded_min_y) / expanded_height
    court_max_y_norm = (template_max_y - expanded_min_y) / expanded_height
    
    court_min_x_px = int(court_min_x_norm * heatmap_size)
    court_max_x_px = int(court_max_x_norm * heatmap_size)
    court_min_y_px = int(court_min_y_norm * heatmap_size)
    court_max_y_px = int(court_max_y_norm * heatmap_size)
    
    heatmap[court_min_y_px:court_max_y_px, court_min_x_px:court_max_x_px] = (0, 100, 0)
    
    # Función para escalar coordenadas al heatmap
    def scale_to_heatmap(x, y):
        norm_x = (x - expanded_min_x) / expanded_width
        norm_y = (y - expanded_min_y) / expanded_height
        heatmap_x = int(norm_x * heatmap_size)
        heatmap_y = int(norm_y * heatmap_size)
        heatmap_x = max(0, min(heatmap_size - 1, heatmap_x))
        heatmap_y = max(0, min(heatmap_size - 1, heatmap_y))
        return heatmap_x, heatmap_y
    
    # Dibujar las líneas de la cancha
    line_color = (255, 255, 255)
    line_thickness = 3
    
    # Baseline superior
    pt1 = scale_to_heatmap(*court_ref.baseline_top[0])
    pt2 = scale_to_heatmap(*court_ref.baseline_top[1])
    cv2.line(heatmap, pt1, pt2, line_color, line_thickness)
    
    # Baseline inferior
    pt1 = scale_to_heatmap(*court_ref.baseline_bottom[0])
    pt2 = scale_to_heatmap(*court_ref.baseline_bottom[1])
    cv2.line(heatmap, pt1, pt2, line_color, line_thickness)
    
    # Líneas laterales
    pt1 = scale_to_heatmap(*court_ref.left_court_line[0])
    pt2 = scale_to_heatmap(*court_ref.left_court_line[1])
    cv2.line(heatmap, pt1, pt2, line_color, line_thickness)
    
    pt1 = scale_to_heatmap(*court_ref.right_court_line[0])
    pt2 = scale_to_heatmap(*court_ref.right_court_line[1])
    cv2.line(heatmap, pt1, pt2, line_color, line_thickness)
    
    # Líneas internas
    pt1 = scale_to_heatmap(*court_ref.left_inner_line[0])
    pt2 = scale_to_heatmap(*court_ref.left_inner_line[1])
    cv2.line(heatmap, pt1, pt2, line_color, 2)
    
    pt1 = scale_to_heatmap(*court_ref.right_inner_line[0])
    pt2 = scale_to_heatmap(*court_ref.right_inner_line[1])
    cv2.line(heatmap, pt1, pt2, line_color, 2)
    
    pt1 = scale_to_heatmap(*court_ref.top_inner_line[0])
    pt2 = scale_to_heatmap(*court_ref.top_inner_line[1])
    cv2.line(heatmap, pt1, pt2, line_color, 2)
    
    pt1 = scale_to_heatmap(*court_ref.bottom_inner_line[0])
    pt2 = scale_to_heatmap(*court_ref.bottom_inner_line[1])
    cv2.line(heatmap, pt1, pt2, line_color, 2)
    
    pt1 = scale_to_heatmap(*court_ref.middle_line[0])
    pt2 = scale_to_heatmap(*court_ref.middle_line[1])
    cv2.line(heatmap, pt1, pt2, line_color, 2)
    
    # Red de la cancha
    pt1 = scale_to_heatmap(*court_ref.net[0])
    pt2 = scale_to_heatmap(*court_ref.net[1])
    cv2.line(heatmap, pt1, pt2, line_color, line_thickness)
    
    # Dibujar todos los piques detectados
    bounce_count = 0
    for bounce_x, bounce_y in bounce_history:
        # Transformar del frame al espacio 2D
        point_frame = np.array([[[bounce_x, bounce_y]]], dtype=np.float32)
        point_2d = cv2.perspectiveTransform(point_frame, inv_homography)
        
        if point_2d is not None and len(point_2d) > 0:
            x_2d, y_2d = point_2d[0][0]
            # Escalar al heatmap
            heatmap_x, heatmap_y = scale_to_heatmap(x_2d, y_2d)
            # Dibujar círculo blanco para cada pique
            cv2.circle(heatmap, (heatmap_x, heatmap_y), 5, (255, 255, 255), -1)
            bounce_count += 1
    
    # Generar nombre del archivo de salida
    base_name = os.path.splitext(video_out_path)[0]
    heatmap_path = f"{base_name}_heatmap.png"
    
    # Guardar el heatmap
    cv2.imwrite(heatmap_path, heatmap)
    print(f"[Heatmap] Mapa de tiros guardado: {heatmap_path} ({bounce_count} piques detectados)")
