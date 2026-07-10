import cv2
import numpy as np
from scipy.spatial import distance


class CourtReference:
    """
    Court reference model (plantilla 2D ideal)
    """
    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        
        # Estos son los 14 puntos de la plantilla
        self.key_points = [*self.baseline_top, *self.baseline_bottom, 
                           *self.left_inner_line, *self.right_inner_line,
                           *self.top_inner_line, *self.bottom_inner_line,
                           *self.middle_line]
        
        # Configuraciones de 4 puntos para probar la homografía
        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                               self.right_inner_line[1]],
                           3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                               self.right_court_line[1]],
                           4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                               self.right_inner_line[1]],
                           5: [*self.top_inner_line, *self.bottom_inner_line],
                           6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
                           7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
                           8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                               self.right_court_line[1]],
                           9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
                               self.left_inner_line[1]],
                           10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0],
                                self.middle_line[1]],
                           11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1],
                                self.bottom_inner_line[1]],
                           12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}


# Inicializar la referencia de la cancha y los puntos clave
court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

# Pre-calcular los índices de las configuraciones
court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i+1]
    inds = []
    for j in range(4):
        inds.append(court_ref.key_points.index(conf[j]))
    court_conf_ind[i+1] = inds


def get_trans_matrix(points):
    """
    Calcula la matriz de transformación de homografía que mejor ajusta los puntos detectados
    a la plantilla de referencia de la cancha.
    
    Args:
        points: Lista de 14 puntos detectados (pueden contener None si no se detectaron)
    
    Returns:
        matrix_trans: Matriz de homografía 3x3 o None si no se pudo calcular
    """
    matrix_trans = None
    dist_max = np.inf
    
    # Asegurarse de que 'points' tenga el formato correcto (lista de tuplas)
    # y manejar los (None, None)
    formatted_points = []
    for p in points:
        if p[0] is not None and p[1] is not None:
            formatted_points.append((float(p[0]), float(p[1])))
        else:
            formatted_points.append((None, None))  # Mantener None si no se detectó

    for conf_ind in range(1, 13):  # Iterar sobre las 12 configuraciones
        conf_template = court_ref.court_conf[conf_ind]  # Puntos de la plantilla
        inds = court_conf_ind[conf_ind]  # Índices de esos puntos

        # Obtener los 4 puntos detectados correspondientes
        inters_detected = [formatted_points[inds[0]], formatted_points[inds[1]], 
                           formatted_points[inds[2]], formatted_points[inds[3]]]
        
        # Si alguno de los 4 puntos clave falta, no se puede usar esta configuración
        if not any([None in x for x in inters_detected]):
            # Calcular la matriz de homografía
            matrix, _ = cv2.findHomography(np.float32(conf_template), np.float32(inters_detected), method=0)
            
            # Transformar todos los puntos de la plantilla usando esta matriz
            trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
            
            # Calcular el error (distancia) para los otros puntos (no los 4 que usamos)
            dists = []
            for i in range(14):  # Nota: el original decía 12, pero hay 14 puntos
                if i not in inds and formatted_points[i][0] is not None:
                    # Calcular distancia entre el punto detectado y el punto transformado por homografía
                    dists.append(distance.euclidean(formatted_points[i], np.squeeze(trans_kps[i])))
            
            if dists:  # Asegurarse de que haya distancias para calcular
                dist_median = np.mean(dists)
                # Si esta matriz da un error menor, guardarla como la mejor
                if dist_median < dist_max:
                    matrix_trans = matrix
                    dist_max = dist_median
    return matrix_trans

