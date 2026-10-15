import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
import cv2
import colorsys

# ==========================================
# KONFIGURASI WARNA & NAMA KELAS
# ==========================================
custom_names = {
    0: "00_dcw_1plus_madu_1000g", 1: "01_dcw_1plus_madu_800g", 2: "02_dcw_1plus_madu_400g",
    3: "03_dcw_1plus_madu_200g", 4: "04_dcw_1plus_van_1000g", 5: "05_dcw_1plus_van_800g",
    6: "06_dcw_1plus_van_400g", 7: "07_dcw_1plus_van_200g", 8: "08_dcw_3plus_madu_1000g",
    9: "09_dcw_3plus_madu_800g", 10: "10_dcw_3plus_madu_400g", 11: "11_dcw_3plus_van_1000g",
    12: "12_dcw_3plus_van_800g", 13: "13_dcw_3plus_van_400g", 14: "14_dcw_5plus_madu_1000g",
    15: "15_dcw_5plus_madu_800g", 16: "16_dcw_5plus_madu_400g", 17: "17_lact_1_happy_1kg",
    18: "18_lact_1_happy_750g", 19: "19_lact_1_happy_350g", 20: "20_lact_1_happy_180g",
    21: "21_lact_2_happy_1kg", 22: "22_lact_2_happy_750g", 23: "23_lact_2_happy_350g",
    24: "24_lact_2_happy_180g", 25: "25_lact_pro_0_6_1kg", 26: "26_lact_pro_0_6_735g",
    27: "27_lact_pro_0_6_350g", 28: "28_lact_pro_0_6_180g", 29: "29_lact_pro_6_12_1kg",
    30: "30_lact_pro_6_12_735g", 31: "31_lact_pro_6_12_350g", 32: "32_lact_pro_6_12_180g",
    33: "33_lactg_3_happy_madu_1kg", 34: "34_lactg_3_happy_madu_750g", 35: "35_lactg_3_happy_madu_350g",
    36: "36_lactg_3_happy_madu_180g", 37: "37_lactg_3_happy_van_1kg", 38: "38_lactg_3_happy_van_750g",
    39: "39_lactg_3_happy_van_350g", 40: "40_lactg_pro_1plus_madu_1kg", 41: "41_lactg_pro_1plus_madu_735g",
    42: "42_lactg_pro_1plus_madu_350g", 43: "43_lactg_pro_1plus_madu_145g", 44: "44_lactg_pro_1plus_van_1kg",
    45: "45_lactg_pro_1plus_van_735g", 46: "46_lactg_pro_1plus_van_350g", 47: "47_other_variant",
    48: "48_other_brand"
}

# Membuat warna kelas secara dinamis
class_colors = {}
base_groups = {}

for class_id, class_name in custom_names.items():
    parts = class_name.split('_')
    if len(parts) >= 4 and parts[-1] not in ['variant', 'brand']:
        base_name = "_".join(parts[1:-1])
    else:
        base_name = "other"
    
    if base_name not in base_groups:
        base_groups[base_name] = []
    base_groups[base_name].append(class_name)

num_groups = len(base_groups)
group_hues = {base_name: i / num_groups for i, base_name in enumerate(base_groups.keys())}

for base_name, members in base_groups.items():
    hue = group_hues[base_name]
    num_members = len(members)
    for i, class_name in enumerate(members):
        brightness = 0.6 + (i * (0.4 / (num_members - 1))) if num_members > 1 else 0.8
        saturation = 0.85
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        # Simpan dalam format RGB (karena PIL Image / Streamlit memakai RGB)
        class_colors[class_name] = (int(r * 255), int(g * 255), int(b * 255))

# Fungsi untuk menggambar bounding box transparan yang elegan
def plot_box_better(img, bbox, label, color, line_thickness=3):
    x1, y1, x2, y2 = map(int, bbox)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.25  
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

    font_scale = 0.6
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    text_x = x1
    text_y = y1 - 10 
    if text_y - text_h < 0: 
        text_y = y1 + text_h + 10 

    p = 5 
    cv2.rectangle(img, (text_x - p, text_y - text_h - p), (text_x + text_w + p, text_y + p), color, -1)
    
    # Warna teks hitam atau putih tergantung kecerahan kotak label
    text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
    cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img

# Konfigurasi Halaman
st.set_page_config(page_title="Demo YOLOv12 SSL - Ritel", layout="wide")

st.title("Demo Deteksi Objek Ritel: YOLOv12 + Noisy Student")
st.markdown("Eksplorasi dan bandingkan hasil deteksi Model Student serta grafik pelatihan dari berbagai skenario data berlabel dan sumber dataset.")
st.markdown("Geser slider untuk membandingkan hasil deteksi dan grafik pelatihan dari berbagai skenario data berlabel.")

# ==========================================
# PENGATURAN PARAMETER (SIDEBAR / TOP)
# ==========================================
col_param1, col_param2 = st.columns(2)

with col_param1:
    # Pilihan Sumber Dataset
    jenis_dataset = st.radio(
        "Pilih Sumber Dataset:",
        ("Dataset Parsial", "Dataset Menyeluruh"),
        horizontal=True,
        help="Pilih dataset mana yang model/metriknya ingin ditampilkan."
    )

with col_param2:
    # Slider untuk memilih rasio data berlabel
    rasio_label = st.slider("Rasio Data Berlabel (%)", min_value=10, max_value=100, step=10)

# Menentukan folder berdasarkan pilihan dataset
if jenis_dataset == "Dataset Parsial":
    folder_weights = "weights"
    folder_metrics = "metrics"
else:
    folder_weights = "weightsImproved"
    folder_metrics = "metricsImproved"

# Mengamankan jalur direktori aktif
current_dir = os.path.dirname(os.path.abspath(__file__))

# Fungsi untuk memuat model dengan caching yang diperbaiki
@st.cache_resource
def load_model(rasio, nama_folder):
    # Sekarang kita menggunakan parameter 'nama_folder' alih-alih hardcode "weights"
    model_path = os.path.join(current_dir, nama_folder, f"best_{rasio}.pt")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Memuat model dengan menyertakan variabel folder_weights dari pilihan radio button
model = load_model(rasio_label, folder_weights)

st.divider()

# Membuat Tab agar UI rapi
tab1, tab2, tab3 = st.tabs([
    "🖼️ Demo Inferensi Gambar", 
    "📊 Grafik Metrik Pelatihan", 
    "📈 Perbandingan Semua Skenario"
])

# ==========================================
# TAB 1: DEMO INFERENSI
# ==========================================
with tab1:
    st.header(f"Pengujian Visual ({jenis_dataset} | Model {rasio_label}%)")
    uploaded_file = st.file_uploader("Unggah gambar rak ritel (jpg/png)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if model is None:
            st.error(f"File model untuk rasio {rasio_label}% tidak ditemukan di folder '{folder_weights}'!")
        else:
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file).convert("RGB")
            
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Hasil Deteksi")
                # Indikator teks agar penguji tahu parameter apa yang digunakan
                st.markdown("**Parameter Inferensi:** `conf=0.7`, `iou=0.6`")
                
                if st.button("Jalankan Inferensi", key="btn_infer"):
                    with st.spinner("Model sedang memproses gambar..."):
                        
                        # 1. Jalankan inferensi
                        results = model.predict(source=image, conf=0.7, iou=0.6)
                        
                        # 2. Konversi PIL Image ke NumPy Array (format RGB)
                        img_array = np.array(image)
                        
                        # 3. Iterasi setiap objek yang terdeteksi dan gambar manual
                        boxes = results[0].boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls_id = int(box.cls[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            
                            class_name = custom_names.get(cls_id, "UNKNOWN")
                            color = class_colors.get(class_name, (200, 200, 200))
                            
                            # Tampilkan persentase confidence beserta namanya
                            label = f"{class_name} {conf:.2f}"
                            
                            img_array = plot_box_better(img_array, (x1, y1, x2, y2), label, color)
                        
                        # 4. Tampilkan gambar ke Streamlit
                        st.image(img_array, use_container_width=True)
                        st.success(f"Mendeteksi {len(boxes)} objek.")

# ==========================================
# TAB 2: METRIK PELATIHAN
# ==========================================
with tab2:
    st.header(f"Performa Model selama Pelatihan ({jenis_dataset} | {rasio_label}%)")
    
    # Membaca CSV dari folder yang dinamis
    csv_path = os.path.join(current_dir, folder_metrics, f"results_{rasio_label}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        if 'epoch' in df.columns:
            df.set_index('epoch', inplace=True)
        
        # --- BAGIAN 1: METRIK PERFORMA ---
        st.markdown("### 🎯 Metrik Performa Deteksi")
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.write("**Grafik mAP (Mean Average Precision)**")
            kolom_map = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']
            if all(k in df.columns for k in kolom_map):
                st.line_chart(df[kolom_map])
            else:
                st.warning("Kolom mAP tidak ditemukan.")
                
        with col_perf2:
            st.write("**Grafik Precision & Recall**")
            kolom_pr = ['metrics/precision(B)', 'metrics/recall(B)']
            if all(k in df.columns for k in kolom_pr):
                st.line_chart(df[kolom_pr])
            else:
                st.warning("Kolom Precision/Recall tidak ditemukan.")

        st.divider()

        # --- BAGIAN 2: ANALISIS LOSS ---
        st.markdown("### 📉 Analisis Loss (Train vs Validation)")
        col_loss1, col_loss2, col_loss3 = st.columns(3)
        
        with col_loss1:
            st.write("**Box Loss**")
            kolom_box = ['train/box_loss', 'val/box_loss']
            if all(k in df.columns for k in kolom_box):
                st.line_chart(df[kolom_box])
                
        with col_loss2:
            st.write("**Classification (Cls) Loss**")
            kolom_cls = ['train/cls_loss', 'val/cls_loss']
            if all(k in df.columns for k in kolom_cls):
                st.line_chart(df[kolom_cls])
                
        with col_loss3:
            st.write("**Distribution Focal (DFL) Loss**")
            kolom_dfl = ['train/dfl_loss', 'val/dfl_loss']
            if all(k in df.columns for k in kolom_dfl):
                st.line_chart(df[kolom_dfl])
        
        st.divider()
        
        
            
    else:
        st.error(f"File CSV tidak ditemukan di jalur: {csv_path}. Pastikan struktur folder sudah benar.")

# # ==========================================
# # TAB 3: PERBANDINGAN SEMUA SKENARIO (SWEET SPOT ANALYSIS)
# # ==========================================
# with tab3:
#     st.header(f"Analisis Komparatif 'Sweet Spot' SSL ({jenis_dataset})")
#     st.markdown("""
#     Bagian ini membandingkan metrik mAP dari seluruh skenario rasio data berlabel (10% - 100%) 
#     untuk melihat pengaruh metode *Noisy Student* secara keseluruhan, serta mencari **Sweet Spot**-nya.
#     """)

#     # Daftar semua rasio yang ingin dibandingkan
#     semua_rasio = list(range(10, 110, 10))
    
#     # List untuk menampung data gabungan
#     data_map50 = {}
#     data_map50_95 = {}

#     with st.spinner("Mengompilasi data dari seluruh skenario..."):
#         for rasio in semua_rasio:
#             csv_path_all = os.path.join(current_dir, "metricsImproved", f"results_{rasio}.csv")
            
#             if os.path.exists(csv_path_all):
#                 df_temp = pd.read_csv(csv_path_all)
#                 df_temp.columns = df_temp.columns.str.strip()
                
#                 # Gunakan epoch sebagai index agar sinkron di grafik
#                 if 'epoch' in df_temp.columns:
#                     # Ambil kolom mAP50 dan mAP50-95
#                     data_map50[f"Rasio {rasio}%"] = df_temp['metrics/mAP50(B)']
#                     data_map50_95[f"Rasio {rasio}%"] = df_temp['metrics/mAP50-95(B)']

#     if data_map50:
#         # Konversi dictionary ke DataFrame
#         df_comparison_50 = pd.DataFrame(data_map50)
#         df_comparison_95 = pd.DataFrame(data_map50_95)
        
#         col_comp1, col_comp2 = st.columns(2)

#         with col_comp1:
#             st.subheader("Grafik Perbandingan mAP50")
#             # Membuat grafik dengan Plotly
#             fig_50 = px.line(df_comparison_50, labels={'value': 'mAP50', 'index': 'Epoch', 'variable': 'Rasio'})
#             # Memposisikan legenda di sebelah kanan agar rapi
#             fig_50.update_layout(
#                 legend_title_text='Skenario Data',
#                 legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
#                 margin=dict(r=150) # Memberi ruang ekstra di kanan untuk legenda
#             )
#             st.plotly_chart(fig_50, use_container_width=True)
#             st.caption("Tren kenaikan mAP50 untuk setiap rasio data berlabel.")

#         with col_comp2:
#             st.subheader("Grafik Perbandingan mAP50-95")
#             # Membuat grafik dengan Plotly
#             fig_95 = px.line(df_comparison_95, labels={'value': 'mAP50-95', 'index': 'Epoch', 'variable': 'Rasio'})
#             # Memposisikan legenda di sebelah kanan agar rapi
#             fig_95.update_layout(
#                 legend_title_text='Skenario Data',
#                 legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
#                 margin=dict(r=150)
#             )
#             st.plotly_chart(fig_95, use_container_width=True)
#             st.caption("Tren stabilitas deteksi pada berbagai tingkat IoU (mAP50-95).")

#         st.divider()
        
#         # Ringkasan Akhir & Perhitungan DELTA untuk Bab 4/5
#         st.success("### 🎓 Analisis 'Sweet Spot' (Tabel Delta)")
        
#         summary_data = []
#         prev_map50 = 0.0
#         prev_map50_95 = 0.0
        
#         # Variabel untuk menampung ringkasan otomatis
#         max_delta_val = -999.0
#         max_delta_scenario = ""
#         sweet_spot_scenario = "Belum Ditemukan"
#         best_map_val = -1.0
#         best_map_scenario = ""
#         found_sweet_spot = False
        
#         # Memastikan kolom terurut secara numerik (10%, 20%, dst.)
#         sorted_cols = sorted(df_comparison_50.columns, key=lambda x: int(x.split(' ')[1].replace('%', '')))
        
#         for i, col in enumerate(sorted_cols):
#             current_map50 = df_comparison_50[col].max()
#             current_map50_95 = df_comparison_95[col].max()
            
#             # Hitung delta absolut dan persentase
#             if i == 0:
#                 delta_50 = 0.0 
#                 delta_95 = 0.0
#                 delta_50_pct = 0.0
#                 delta_95_pct = 0.0
#             else:
#                 delta_50 = current_map50 - prev_map50
#                 delta_95 = current_map50_95 - prev_map50_95
                
#                 # Menghitung persentase kenaikan (mencegah pembagian dengan nol)
#                 delta_50_pct = (delta_50 / prev_map50 * 100) if prev_map50 > 0 else 0.0
#                 delta_95_pct = (delta_95 / prev_map50_95 * 100) if prev_map50_95 > 0 else 0.0
                
#                 # 1. Mencari Kenaikan Terbesar (Max Delta berdasarkan persentase)
#                 if delta_50_pct > max_delta_val:
#                     max_delta_val = delta_50_pct
#                     max_delta_scenario = f"{sorted_cols[i-1]} ke {col}"
                
#                 # 2. Mencari Sweet Spot (Misal batasnya kenaikan di bawah 1%)
#                 if not found_sweet_spot and delta_50_pct < 1.0:
#                     sweet_spot_scenario = sorted_cols[i-1]
#                     found_sweet_spot = True

#             # 3. Mencari Performa Puncak
#             if current_map50 > best_map_val:
#                 best_map_val = current_map50
#                 best_map_scenario = col
                
#             summary_data.append({
#                 "Skenario": col,
#                 "Max mAP50": round(current_map50, 4),
#                 "Δ Absolut mAP50": f"+{round(delta_50, 4)}" if delta_50 > 0 else round(delta_50, 4),
#                 "Δ Persen mAP50 (%)": f"+{round(delta_50_pct, 2)}%" if delta_50_pct > 0 else f"{round(delta_50_pct, 2)}%",
#                 "Max mAP50-95": round(current_map50_95, 4),
#                 "Δ Absolut mAP50-95": f"+{round(delta_95, 4)}" if delta_95 > 0 else round(delta_95, 4),
#                 "Δ Persen mAP50-95 (%)": f"+{round(delta_95_pct, 2)}%" if delta_95_pct > 0 else f"{round(delta_95_pct, 2)}%"
#             })
            
#             # Update nilai previous untuk iterasi selanjutnya
#             prev_map50 = current_map50
#             prev_map50_95 = current_map50_95
            
#         # Jika grafik terus naik dan tidak pernah landai (delta % selalu > 1%)
#         if not found_sweet_spot:
#             sweet_spot_scenario = sorted_cols[-1]
        
#         df_summary = pd.DataFrame(summary_data)
        
#         # Menampilkan tabel interaktif
#         st.dataframe(df_summary, use_container_width=True)
        
#         # Menampilkan Ringkasan Otomatis untuk Laporan Skripsi
#         st.info(f"""
#         **📝 Ringkasan Analisis:**
        
#         1. **Pencapaian Puncak (Best Performance):** Model mencapai performa deteksi tertinggi pada **{best_map_scenario}** dengan nilai mAP50 sebesar **{best_map_val:.4f}**. Ini menunjukkan batas atas kemampuan model dalam mengenali objek ritel pada dataset ini.
#         2. **Lonjakan Pembelajaran Terbesar:** Kenaikan akurasi paling signifikan terjadi pada fase penambahan data dari **{max_delta_scenario}**, yaitu sebesar **+{max_delta_val:.4f}%**. Ini mengindikasikan bahwa pada titik tersebut, model *Student* mendapatkan informasi fitur yang paling krusial dari *pseudo-labels* yang dihasilkan model *Teacher*.
#         """)
#     else:
#         st.warning("Data CSV untuk perbandingan tidak ditemukan. Pastikan file 'results_10.csv' hingga 'results_90.csv' tersedia di folder metrics.")

# ==========================================
# TAB 3: PERBANDINGAN SEMUA SKENARIO
# ==========================================
with tab3:
    st.header(f"Analisis Komparatif 'Sweet Spot' SSL ({jenis_dataset})")
    st.markdown("""
    Analisis ini membandingkan mAP dari rasio data berlabel (10% - 100%) 
    untuk melihat pengaruh metode *Noisy Student* dan mencari **Sweet Spot** optimal.
    """)

    semua_rasio = list(range(10, 110, 10))
    data_map50 = {}
    data_map50_95 = {}

    with st.spinner("Mengompilasi data dari seluruh skenario..."):
        for rasio in semua_rasio:
            csv_path_all = os.path.join(current_dir, folder_metrics, f"results_{rasio}.csv")
            
            if os.path.exists(csv_path_all):
                df_temp = pd.read_csv(csv_path_all)
                df_temp.columns = df_temp.columns.str.strip()
                
                if 'epoch' in df_temp.columns:
                    data_map50[f"Rasio {rasio}%"] = df_temp['metrics/mAP50(B)']
                    data_map50_95[f"Rasio {rasio}%"] = df_temp['metrics/mAP50-95(B)']

    if data_map50:
        df_comparison_50 = pd.DataFrame(data_map50)
        df_comparison_95 = pd.DataFrame(data_map50_95)
        
        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            st.subheader("Grafik Perbandingan mAP50")
            fig_50 = px.line(df_comparison_50, labels={'value': 'mAP50', 'index': 'Epoch', 'variable': 'Rasio'})
            fig_50.update_layout(legend_title_text='Skenario Data', legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), margin=dict(r=150))
            st.plotly_chart(fig_50, use_container_width=True)

        with col_comp2:
            st.subheader("Grafik Perbandingan mAP50-95")
            fig_95 = px.line(df_comparison_95, labels={'value': 'mAP50-95', 'index': 'Epoch', 'variable': 'Rasio'})
            fig_95.update_layout(legend_title_text='Skenario Data', legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), margin=dict(r=150))
            st.plotly_chart(fig_95, use_container_width=True)

        st.divider()
        
        st.success(f"### 🎓 Analisis 'Sweet Spot' (Tabel Delta - {jenis_dataset})")
        
        summary_data = []
        prev_map50, prev_map50_95 = 0.0, 0.0
        max_delta_val, best_map_val = -999.0, -1.0
        max_delta_scenario, best_map_scenario, sweet_spot_scenario = "", "", "Belum Ditemukan"
        found_sweet_spot = False
        
        sorted_cols = sorted(df_comparison_50.columns, key=lambda x: int(x.split(' ')[1].replace('%', '')))
        
        for i, col in enumerate(sorted_cols):
            current_map50 = df_comparison_50[col].max()
            current_map50_95 = df_comparison_95[col].max()
            
            if i == 0:
                delta_50, delta_95, delta_50_pct, delta_95_pct = 0.0, 0.0, 0.0, 0.0
            else:
                delta_50 = current_map50 - prev_map50
                delta_95 = current_map50_95 - prev_map50_95
                delta_50_pct = (delta_50 / prev_map50 * 100) if prev_map50 > 0 else 0.0
                delta_95_pct = (delta_95 / prev_map50_95 * 100) if prev_map50_95 > 0 else 0.0
                
                if delta_50_pct > max_delta_val:
                    max_delta_val = delta_50_pct
                    max_delta_scenario = f"{sorted_cols[i-1]} ke {col}"
                
                if not found_sweet_spot and delta_50_pct < 1.0:
                    sweet_spot_scenario = sorted_cols[i-1]
                    found_sweet_spot = True

            if current_map50 > best_map_val:
                best_map_val = current_map50
                best_map_scenario = col
                
            summary_data.append({
                "Skenario": col,
                "Max mAP50": round(current_map50, 4),
                "Δ Absolut mAP50": f"+{delta_50:.4f}" if delta_50 > 0 else f"{delta_50:.4f}",
                "Δ Persen mAP50 (%)": f"+{delta_50_pct:.2f}%" if delta_50_pct > 0 else f"{delta_50_pct:.2f}%",
                "Max mAP50-95": round(current_map50_95, 4),
                "Δ Absolut mAP50-95": f"+{delta_95:.4f}" if delta_95 > 0 else f"{delta_95:.4f}",
                "Δ Persen mAP50-95 (%)": f"+{delta_95_pct:.2f}%" if delta_95_pct > 0 else f"{delta_95_pct:.2f}%"
            })
            
            prev_map50, prev_map50_95 = current_map50, current_map50_95
            
        if not found_sweet_spot:
            sweet_spot_scenario = sorted_cols[-1]
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        st.info(f"""
        **📝 Insight:**
        Pada **{jenis_dataset}**, performa puncak berada di **{best_map_scenario}** (mAP50: **{best_map_val:.4f}**). 
        Lonjakan pemahaman fitur (*Teacher to Student*) terbesar terjadi pada transisi **{max_delta_scenario}** (**+{max_delta_val:.2f}%**).
        """)
    else:
        st.warning(f"Data CSV untuk perbandingan tidak ditemukan di folder '{folder_metrics}'.")