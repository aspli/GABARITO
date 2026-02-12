import streamlit as st
import cv2
import numpy as np
import imutils
from imutils import contours
import pandas as pd

# --- Configurações ---
QUESTION_RANGES = {
    "left": range(27, 40),  # 27-39
    "right": range(40, 53)  # 40-52
}
OPTIONS = ['A', 'B', 'C', 'D', 'E']

def pre_process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny suave para contornos
    edged = cv2.Canny(blurred, 30, 150) 
    return gray, edged

def find_bubbles(edged_image, img_area):
    # RETR_TREE para pegar contornos internos
    cnts = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    questionCnts = []
    
    # Filtros de área
    min_area = img_area * 0.00015 
    max_area = img_area * 0.008

    for c in cnts:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if min_area < area < max_area and 0.7 <= ar <= 1.3:
            questionCnts.append(c)
            
    return questionCnts

def smart_filter_columns(cnts):
    """
    ALGORITMO DE CLUSTERIZAÇÃO:
    Em vez de cortar a imagem em %, agrupa os contornos por coordenadas X.
    Esperamos encontrar 5 grandes grupos verticais (A, B, C, D, E).
    O que sobrar (números à esquerda) é descartado.
    """
    if not cnts: return []
    
    # 1. Pega o centro X de cada contorno
    items = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            items.append((cX, c))
    
    # 2. Ordena por X
    items.sort(key=lambda x: x[0])
    
    # 3. Agrupa em clusters (se a distância X for pequena, é a mesma coluna)
    clusters = []
    current_cluster = [items[0]]
    
    for i in range(1, len(items)):
        # Se a distância horizontal for maior que 15px, é uma nova coluna
        if items[i][0] - current_cluster[-1][0] > 15: 
            clusters.append(current_cluster)
            current_cluster = [items[i]]
        else:
            current_cluster.append(items[i])
    clusters.append(current_cluster)
    
    # 4. Filtra: Queremos as 5 colunas com mais elementos (as bolhas)
    # Os números (27, 28...) formam uma coluna, mas geralmente com menos elementos ou posição errada.
    # Mas o jeito mais seguro é: As 5 colunas mais à DIREITA são A, B, C, D, E.
    
    # Filtra clusters muito pequenos (ruído)
    valid_clusters = [cl for cl in clusters if len(cl) > 3]
    
    if len(valid_clusters) < 5:
        return [] # Falha na detecção
        
    # Pega os 5 clusters mais à direita (maior X)
    # Isso garante que ignoramos os números na esquerda
    valid_clusters.sort(key=lambda cl: cl[0][0]) # Ordena por posição X
    final_5_cols = valid_clusters[-5:] # Pega os 5 últimos
    
    # Reconstrói a lista de contornos limpa
    clean_cnts = []
    for cl in final_5_cols:
        for item in cl:
            clean_cnts.append(item[1])
            
    return clean_cnts

def grade_column_with_grid(gray, all_cnts, q_range, key_dict, out_img):
    results = []
    score = 0
    total = 0
    
    # 1. Limpeza Inteligente (Clusterização)
    bubbles = smart_filter_columns(all_cnts)
    
    if not bubbles:
        return [], 0, 0
        
    # Estatísticas para a Grade Virtual
    boxes = [cv2.boundingRect(c) for c in bubbles]
    min_x = min([b[0] for b in boxes])
    max_x = max([b[0] for b in boxes])
    avg_w = np.mean([b[2] for b in boxes])
    
    # Passo horizontal (distância entre centros de colunas)
    step_x = (max_x - min_x) / 4.0
    
    # 2. Agrupar em Linhas (Y)
    bubbles = contours.sort_contours(bubbles, method="top-to-bottom")[0]
    
    rows = []
    (_, _, _, h_ref) = cv2.boundingRect(bubbles[0])
    threshold_y = h_ref * 0.6
    
    current_row = [bubbles[0]]
    (_, prev_y, _, _) = cv2.boundingRect(bubbles[0])
    
    for i in range(1, len(bubbles)):
        c = bubbles[i]
        (_, y, _, _) = cv2.boundingRect(c)
        if abs(y - prev_y) < threshold_y:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
            (_, prev_y, _, _) = cv2.boundingRect(c)
    rows.append(current_row)

    # Threshold para leitura de pixels
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    q_nums = sorted(list(q_range))
    limit = min(len(rows), len(q_nums))
    
    for i in range(limit):
        q_num = q_nums[i]
        row_cnts = rows[i]
        
        # Y médio da linha
        ys = [cv2.boundingRect(c)[1] for c in row_cnts]
        avg_y = int(np.mean(ys))
        center_y_line = avg_y + int(h_ref/2)
        
        # --- GRADE VIRTUAL ---
        pixel_counts = []
        found_contours = []
        
        for col_idx in range(5):
            expected_x = int(min_x + (col_idx * step_x))
            
            # Tenta casar contorno existente
            matched = None
            for c in row_cnts:
                cx = cv2.boundingRect(c)[0]
                if abs(cx - expected_x) < (step_x * 0.4):
                    matched = c
                    break
            
            mask = np.zeros(gray.shape, dtype="uint8")
            if matched is not None:
                cv2.drawContours(mask, [matched], -1, 255, -1)
                found_contours.append(matched)
            else:
                # Cria bolha virtual se não achou (para casos preenchidos demais)
                radius = int(avg_w / 2)
                cv2.circle(mask, (expected_x + radius, center_y_line), radius, 255, -1)
                found_contours.append(None)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total_pixels = cv2.countNonZero(mask)
            pixel_counts.append(total_pixels)

        # Decisão
        max_pixels = max(pixel_counts)
        if max_pixels > 150:
            bubbled_idx = pixel_counts.index(max_pixels)
        else:
            bubbled_idx = None
            
        correct_char = key_dict.get(q_num, 'A')
        student_char = OPTIONS[bubbled_idx] if bubbled_idx is not None else "N/A"
        is_correct = (student_char == correct_char)
        
        if is_correct: score += 1
        total += 1
        
        # --- DESENHO ---
        # 1. Desenha resposta do aluno
        if bubbled_idx is not None:
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            cnt = found_contours[bubbled_idx]
            if cnt is not None:
                cv2.drawContours(out_img, [cnt], -1, color, 3)
            else:
                # Desenho da grade virtual
                expected_x = int(min_x + (bubbled_idx * step_x))
                radius = int(avg_w / 2)
                cv2.circle(out_img, (expected_x + radius, center_y_line), radius+2, color, 3)

        # 2. Se errou, mostra a correta (Caixa Azul)
        if not is_correct:
            correct_idx = OPTIONS.index(correct_char)
            expected_x_corr = int(min_x + (correct_idx * step_x))
            
            # Desenha retângulo azul
            cv2.rectangle(out_img, 
                          (expected_x_corr - 5, center_y_line - 15), 
                          (expected_x_corr + int(avg_w) + 5, center_y_line + 15), 
                          (255, 0, 0), 2)

        results.append({
            "Questão": q_num,
            "Aluno": student_char,
            "Correto": correct_char,
            "Status": "✅" if is_correct else "❌"
        })

    return results, score, total

# --- Interface ---
st.set_page_config(page_title="Corretor OMR V6 Final", layout="wide")
st.title("📝 Corretor de Gabarito (V6 - Clusterização)")

# Sidebar com Cache Fix
st.sidebar.header("Gabarito")
key_left = {}
key_right = {}

with st.sidebar.form("key_form"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("27-39")
        for q in QUESTION_RANGES["left"]:
            # key alterada para forçar reset do cache
            key_left[q] = st.selectbox(f"Q{q}", OPTIONS, index=0, key=f"L_v6_{q}") 
    with c2:
        st.subheader("40-52")
        for q in QUESTION_RANGES["right"]:
            # Index=1 (B) e key alterada para forçar reset
            key_right[q] = st.selectbox(f"Q{q}", OPTIONS, index=1, key=f"R_v6_{q}") 
    st.form_submit_button("Atualizar Gabarito")

# Main
uploaded = st.file_uploader("Carregar Imagem", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.image(img, caption="Original", channels="BGR", width=500)
    
    if st.button("Corrigir Prova"):
        gray, edged = pre_process_image(img)
        img_area = img.shape[0] * img.shape[1]
        
        # 1. Encontrar Candidatos
        raw_cnts = find_bubbles(edged, img_area)
        
        if len(raw_cnts) > 0:
            # 2. Separar Colunas (Mediana X)
            all_xs = [cv2.boundingRect(c)[0] for c in raw_cnts]
            mid_x = np.median(all_xs)
            
            left_cnts = [c for c in raw_cnts if cv2.boundingRect(c)[0] < mid_x]
            right_cnts = [c for c in raw_cnts if cv2.boundingRect(c)[0] >= mid_x]
            
            out_img = img.copy()
            
            # 3. Processar
            res_l, sc_l, tot_l = grade_column_with_grid(gray, left_cnts, QUESTION_RANGES["left"], key_left, out_img)
            res_r, sc_r, tot_r = grade_column_with_grid(gray, right_cnts, QUESTION_RANGES["right"], key_right, out_img)
            
            final_score = sc_l + sc_r
            final_total = tot_l + tot_r
            
            st.divider()
            st.header(f"Nota Final: {final_score} / {final_total}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(pd.DataFrame(res_l + res_r), height=600, use_container_width=True)
            with c2:
                st.image(out_img, caption="Correção Final", channels="BGR")
        else:
            st.error("Não foi possível detectar bolhas.")