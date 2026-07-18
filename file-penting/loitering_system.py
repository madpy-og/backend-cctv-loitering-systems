"""
loitering_system.py — Sistem deteksi loitering (standalone script version).

Ini adalah versi .py dari notebook CCTV_Try_revised.ipynb, dengan tambahan:
  - Snapshot evidence: begitu alert loitering terpicu, frame (lengkap dengan
    overlay bounding box + zona) otomatis disimpan sebagai file gambar bukti,
    dan path-nya dicatat di kolom baru pada CSV log.

Fitur yang SUDAH ADA dari notebook sebelumnya (tetap dipertahankan):
  - Input video, preprocessing (resize sesuai imgsz training)
  - Deteksi person (YOLOv8 custom)
  - Tracking (DeepSORT)
  - Behavior analysis: zona bahaya + durasi loitering, logika AKUMULATIF
    dengan grace period (tahan terhadap noise tracking singkat)
  - Overlay visual (bounding box, track ID, timer, zona)
  - Video output (.mp4) dan log CSV

Fitur yang BELUM ditambahkan (sengaja, sesuai permintaan saat ini):
  - Notifikasi real-time (Telegram bot / webhook) -> menyusul di iterasi berikutnya

Cara pakai:
  1. Sesuaikan bagian KONFIGURASI di bawah (path model, video, output, parameter).
  2. Jalankan: python loitering_system.py
  3. Cek folder OUTPUT_VIDEO_PATH, LOITERING_LOG_CSV, dan SNAPSHOT_DIR setelah selesai.
"""

import os
import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# =============================================================================
# KONFIGURASI (edit bagian ini sesuai kebutuhan)
# =============================================================================

# --- Path dan File ---
MODEL_PATH = '/content/drive/MyDrive/Magang/Uji/Model/best.pt'
VIDEO_PATH = '/content/drive/MyDrive/Magang/Uji/Video/Video-01.mp4'
OUTPUT_VIDEO_PATH = '/content/drive/MyDrive/Magang/Uji/Output/output_loitering.mp4'
LOITERING_LOG_CSV = '/content/drive/MyDrive/Magang/Uji/Output/loitering_log.csv'
SNAPSHOT_DIR = '/content/drive/MyDrive/Magang/Uji/Output/snapshots'  # <-- BARU: folder bukti visual

# --- Parameter Deteksi YOLO ---
CONF_THRESHOLD = 0.4   # Confidence threshold untuk deteksi objek (0.0 - 1.0)
IOU_THRESHOLD = 0.5    # IOU threshold untuk Non-Maximum Suppression (NMS)

# --- Parameter Video Processing ---
N_FRAMES_TO_PROCESS = None  # None = proses seluruh video
# PENTING: samakan dengan imgsz TERBAIK dari hasil eksperimen TAHAP 1 (train.py / runs/summary.csv)
IMG_SIZE = 640

# --- Konfigurasi Tracking (DeepSORT) ---
MAX_AGE = 30   # Berapa frame objek bisa hilang sebelum ID-nya dihapus
MIN_HITS = 3   # Berapa banyak deteksi berturut-turut untuk membuat track baru

# --- Konfigurasi Zona Bahaya (Loitering) ---
# TODO: sesuaikan dengan area rawan yang sebenarnya di video kamu.
# Bisa juga pakai interactive_zone_selector() dari notebook untuk klik manual.
HAZARD_ZONE_POLYGON = np.array([[
    (0, 0),
    (160, 0),
    (160, 240),
    (0, 240)
]], np.int32)

# Ambang batas waktu loitering dalam detik
LOITERING_THRESHOLD_SECONDS = 4

# Toleransi waktu (detik) seseorang boleh sempat keluar dari zona akibat noise
# tracking TANPA menghitung ulang durasi loitering dari nol (logika AKUMULATIF).
# Set ke 0 untuk kembali ke perilaku kontinu (reset total begitu keluar zona).
GRACE_PERIOD_SECONDS = 2

# --- Output Visual ---
BBOX_COLOR_NORMAL = (0, 255, 0)     # Hijau (BGR)
BBOX_COLOR_LOITERING = (0, 0, 255)  # Merah (BGR)
ZONE_COLOR = (255, 0, 0)            # Biru (BGR)
ZONE_ALPHA = 0.3


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_video_properties(video_path):
    """Mendapatkan properti video seperti lebar, tinggi, dan FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka video di {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, total_frames


def point_in_polygon(point, polygon):
    """Mengecek apakah suatu titik (centroid) berada di dalam poligon."""
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0


def get_centroid(bbox):
    """Menghitung titik tengah (centroid) dari bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_overlays(frame, tracks, loitering_state, hazard_zone_polygon, fps,
                   current_frame_number, bbox_color_normal, bbox_color_loitering,
                   zone_color, zone_alpha):
    """Menggambar overlay pada frame: bounding box, track ID, timer loitering, dan zona bahaya."""
    overlay = frame.copy()

    if hazard_zone_polygon is not None:
        cv2.fillPoly(overlay, [hazard_zone_polygon], zone_color)
        frame = cv2.addWeighted(overlay, zone_alpha, frame, 1 - zone_alpha, 0)
        cv2.polylines(frame, [hazard_zone_polygon], True, zone_color, 2)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        current_color = bbox_color_normal
        loitering_info = loitering_state.get(track_id)
        if loitering_info and loitering_info['is_loitering']:
            current_color = bbox_color_loitering

        cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

        if loitering_info and loitering_info['in_zone'] and loitering_info['start_frame'] != -1:
            elapsed_frames = current_frame_number - loitering_info['start_frame']
            elapsed_time_sec = elapsed_frames / fps
            timer_text = f"{elapsed_time_sec:.1f}s"
            cv2.putText(frame, timer_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

    return frame


def save_snapshot(frame, track_id, current_frame_number, snapshot_dir):
    """BARU: Simpan frame (dengan overlay) sebagai bukti visual saat alert loitering terpicu.
    Return path file snapshot yang disimpan."""
    os.makedirs(snapshot_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"alert_track{track_id}_frame{current_frame_number}_{timestamp_str}.jpg"
    snapshot_path = os.path.join(snapshot_dir, filename)
    cv2.imwrite(snapshot_path, frame)
    return snapshot_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== Memulai Sistem Deteksi Loitering ===")

    # --- Inisialisasi model dan tracker ---
    print(f"[INFO] Memuat model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    tracker = DeepSort(max_age=MAX_AGE, n_init=MIN_HITS)

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOITERING_LOG_CSV), exist_ok=True)

    # --- Inisialisasi Video Capture dan Writer ---
    video_width, video_height, video_fps, video_total_frames = get_video_properties(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Gagal membuka video: {VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (video_width, video_height))

    print(f"[INFO] Memproses video: {VIDEO_PATH}")
    print(f"[INFO] Resolusi video: {video_width}x{video_height}, FPS: {video_fps:.2f}")
    print(f"[INFO] Zona Bahaya: {HAZARD_ZONE_POLYGON.tolist()}")
    print(f"[INFO] Threshold loitering: {LOITERING_THRESHOLD_SECONDS}s, grace period: {GRACE_PERIOD_SECONDS}s\n")

    current_frame_number = 0
    progress_interval = max(1, video_total_frames // 10)

    loitering_state = {}
    loitering_events_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_number += 1
        if N_FRAMES_TO_PROCESS is not None and current_frame_number > N_FRAMES_TO_PROCESS:
            print(f"[INFO] Berhenti setelah memproses {N_FRAMES_TO_PROCESS} frame.")
            break

        if current_frame_number % progress_interval == 0:
            print(f"[INFO] Memproses frame {current_frame_number}/{video_total_frames}...")

        # --- 3. Deteksi Person (YOLO) ---
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, classes=0, imgsz=IMG_SIZE, verbose=False)

        detections = []
        for r in results:
            for *xyxy, conf, cls in r.boxes.data.tolist():
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        # --- 4. Tracking (DeepSORT) ---
        tracks = tracker.update_tracks(detections, frame=frame)

        # Menampung entri log yang baru saja dibuat frame ini, supaya bisa
        # diisi 'snapshot_path'-nya setelah frame overlay selesai digambar.
        alerts_this_frame = []

        # --- 5. Behavior Analysis (Rule-based: Zone + Dwell Time, akumulatif + grace period) ---
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            centroid = get_centroid((x1, y1, x2, y2))

            if track_id not in loitering_state:
                loitering_state[track_id] = {
                    'in_zone': False,
                    'start_frame': -1,
                    'last_in_zone_frame': -1,
                    'alert_triggered': False,
                    'last_centroid': centroid,
                    'is_loitering': False
                }

            is_in_hazard_zone = point_in_polygon(centroid, HAZARD_ZONE_POLYGON)
            state = loitering_state[track_id]

            if is_in_hazard_zone:
                if state['start_frame'] == -1:
                    state['start_frame'] = current_frame_number
                    state['alert_triggered'] = False
                    state['is_loitering'] = False

                state['in_zone'] = True
                state['last_in_zone_frame'] = current_frame_number

                elapsed_frames = current_frame_number - state['start_frame']
                elapsed_time_sec = elapsed_frames / video_fps

                if elapsed_time_sec >= LOITERING_THRESHOLD_SECONDS and not state['alert_triggered']:
                    alert_timestamp = str(datetime.datetime.now())
                    print(f"[ALERT] Track ID {track_id} loitering di Zona Bahaya selama "
                          f"{elapsed_time_sec:.2f} detik pada frame {current_frame_number}")

                    state['is_loitering'] = True
                    state['alert_triggered'] = True

                    log_entry = {
                        'track_id': track_id,
                        'timestamp_masuk_zona': (datetime.datetime.now() - datetime.timedelta(seconds=elapsed_time_sec)).strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp_alert_triggered': alert_timestamp,
                        'durasi_detik': f"{elapsed_time_sec:.2f}",
                        'koordinat_centroid_terakhir': str(centroid),
                        'nama_zona': 'Zona Bahaya',
                        'snapshot_path': '',  # diisi setelah overlay frame ini selesai digambar
                    }
                    loitering_events_log.append(log_entry)
                    alerts_this_frame.append((track_id, log_entry))
            else:
                state['in_zone'] = False
                if state['start_frame'] != -1:
                    frames_since_last_seen = current_frame_number - state['last_in_zone_frame']
                    grace_period_frames = GRACE_PERIOD_SECONDS * video_fps
                    if frames_since_last_seen > grace_period_frames:
                        state['start_frame'] = -1
                        state['last_in_zone_frame'] = -1
                        state['is_loitering'] = False
                        state['alert_triggered'] = False

        # --- 6. Overlay Visual pada Video Output ---
        output_frame = draw_overlays(
            frame.copy(), tracks, loitering_state, HAZARD_ZONE_POLYGON, video_fps, current_frame_number,
            BBOX_COLOR_NORMAL, BBOX_COLOR_LOITERING, ZONE_COLOR, ZONE_ALPHA
        )
        out.write(output_frame)

        # --- BARU: Simpan snapshot evidence untuk tiap alert yang terpicu frame ini ---
        for track_id, log_entry in alerts_this_frame:
            snapshot_path = save_snapshot(output_frame, track_id, current_frame_number, SNAPSHOT_DIR)
            log_entry['snapshot_path'] = snapshot_path
            print(f"[INFO] Snapshot bukti tersimpan: {snapshot_path}")

    # --- Finalisasi ---
    cap.release()
    out.release()

    print(f"\n[INFO] Processing selesai. Video output disimpan di: {OUTPUT_VIDEO_PATH}")

    if loitering_events_log:
        df_loitering = pd.DataFrame(loitering_events_log)
        df_loitering.to_csv(LOITERING_LOG_CSV, index=False)
        print(f"[INFO] Log loitering disimpan di: {LOITERING_LOG_CSV}")
        print(f"[INFO] Total event loitering: {len(loitering_events_log)}")
        print(f"[INFO] Snapshot bukti tersimpan di folder: {SNAPSHOT_DIR}")
    else:
        print("[INFO] Tidak ada event loitering yang tercatat.")


if __name__ == '__main__':
    main()
