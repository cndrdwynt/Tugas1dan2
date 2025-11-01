# utils.py
# Berisi fungsi-fungsi utilitas/pembantu.

import cv2
import numpy as np
import math
import mediapipe as mp

import config as cfg

mp_hands = mp.solutions.hands

# ==================== FILTER FUNCTIONS (TUGAS 1) ====================

def create_gaussian_kernel(size, sigma):
    """
    Membuat kernel Gaussian secara manual untuk pemahaman konvolusi.
    Menggunakan cv2.getGaussianKernel() sesuai spesifikasi tugas.
    """
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = kernel_1d @ kernel_1d.T  # Outer product untuk 2D kernel
    return kernel_2d

def apply_average_blur(frame, kernel_size):
    """
    Menerapkan Average Blurring dengan kernel size tertentu.
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        return cv2.blur(frame, (kernel_size, kernel_size))
    except Exception as e:
        print(f"Error applying average blur: {e}")
        return frame

def apply_gaussian_blur_custom(frame, kernel_size, sigma):
    """
    Menerapkan Gaussian Blurring menggunakan cv2.filter2D
    untuk menunjukkan pemahaman konvolusi (WAJIB sesuai tugas).
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        # Membuat kernel Gaussian sendiri
        kernel = create_gaussian_kernel(kernel_size, sigma)
        # Menerapkan konvolusi dengan cv2.filter2D
        return cv2.filter2D(frame, -1, kernel)
    except Exception as e:
        print(f"Error applying Gaussian blur: {e}")
        return frame

def apply_sharpening(frame):
    """
    Menerapkan filter Sharpening menggunakan kernel yang ditentukan.
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        return cv2.filter2D(frame, -1, cfg.SHARPEN_KERNEL)
    except Exception as e:
        print(f"Error applying sharpening: {e}")
        return frame

def apply_image_filter(frame, filter_mode):
    """
    Menerapkan filter pada frame berdasarkan mode yang dipilih.
    
    Args:
        frame: Input frame (BGR)
        filter_mode: Mode filter (0-4)
    
    Returns:
        Filtered frame
    """
    if frame is None or frame.size == 0:
        return frame
    
    if filter_mode == cfg.FILTER_MODE_NORMAL:
        return frame
    elif filter_mode == cfg.FILTER_MODE_AVG_5:
        return apply_average_blur(frame, 5)
    elif filter_mode == cfg.FILTER_MODE_AVG_9:
        return apply_average_blur(frame, 9)
    elif filter_mode == cfg.FILTER_MODE_GAUSSIAN:
        return apply_gaussian_blur_custom(frame, cfg.GAUSSIAN_KERNEL_SIZE, cfg.GAUSSIAN_SIGMA)
    elif filter_mode == cfg.FILTER_MODE_SHARPEN:
        return apply_sharpening(frame)
    else:
        return frame

def get_filter_mode_name(filter_mode):
    """
    Mendapatkan nama mode filter untuk ditampilkan di UI.
    """
    if filter_mode == cfg.FILTER_MODE_NORMAL:
        return "Normal"
    elif filter_mode == cfg.FILTER_MODE_AVG_5:
        return "Average Blur 5x5"
    elif filter_mode == cfg.FILTER_MODE_AVG_9:
        return "Average Blur 9x9"
    elif filter_mode == cfg.FILTER_MODE_GAUSSIAN:
        return "Gaussian Blur"
    elif filter_mode == cfg.FILTER_MODE_SHARPEN:
        return "Sharpening"
    else:
        return "Unknown"

# ==================== ORIGINAL UTILITY FUNCTIONS ====================

def create_morphology_kernel(size=5):
    """
    Membuat kernel untuk operasi morfologi.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def detect_color_hsv(frame, color_mode):
    """
    Mendeteksi warna tertentu dalam ruang HSV (TUGAS 2).
    
    Args:
        frame: Input frame dalam BGR
        color_mode: 'green', 'blue', 'red', 'yellow'
    
    Returns:
        mask: Binary mask dari warna yang terdeteksi
        cleaned_mask: Mask setelah operasi morfologi
        contour: Kontur terbesar yang terdeteksi
        area: Luas kontur
        center: Koordinat pusat kontur (x, y)
    """
    if frame is None or frame.size == 0:
        return None, None, None, 0, None
    
    try:
        # 1. Konversi BGR ke HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Thresholding Warna berdasarkan mode
        if color_mode == cfg.COLOR_DETECT_GREEN:
            mask = cv2.inRange(hsv, cfg.HSV_GREEN_LOWER, cfg.HSV_GREEN_UPPER)
        elif color_mode == cfg.COLOR_DETECT_BLUE:
            mask = cv2.inRange(hsv, cfg.HSV_BLUE_LOWER, cfg.HSV_BLUE_UPPER)
        elif color_mode == cfg.COLOR_DETECT_RED:
            # Merah memiliki 2 rentang
            mask1 = cv2.inRange(hsv, cfg.HSV_RED_LOWER1, cfg.HSV_RED_UPPER1)
            mask2 = cv2.inRange(hsv, cfg.HSV_RED_LOWER2, cfg.HSV_RED_UPPER2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_mode == cfg.COLOR_DETECT_YELLOW:
            mask = cv2.inRange(hsv, cfg.HSV_YELLOW_LOWER, cfg.HSV_YELLOW_UPPER)
        else:
            # Default: green
            mask = cv2.inRange(hsv, cfg.HSV_GREEN_LOWER, cfg.HSV_GREEN_UPPER)
        
        # 3. Pembersihan mask dengan operasi morfologi
        kernel = create_morphology_kernel(cfg.MORPH_KERNEL_SIZE)
        
        # Opening: menghapus noise kecil (false positives)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing: menutup lubang kecil (false negatives)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. Temukan kontur
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return mask, cleaned_mask, None, 0, None
        
        # Cari kontur terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Hitung pusat kontur
        center = None
        if area > 0:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
        
        return mask, cleaned_mask, largest_contour, area, center
        
    except Exception as e:
        print(f"Error in color detection: {e}")
        return None, None, None, 0, None

def draw_color_detection_overlay(frame, contour, center, color_name, area):
    """
    Menggambar overlay untuk visualisasi deteksi warna.
    
    Args:
        frame: Frame untuk ditampilkan
        contour: Kontur yang terdeteksi
        center: Koordinat pusat objek
        color_name: Nama warna yang terdeteksi
        area: Luas area yang terdeteksi
    """
    if frame is None:
        return frame
    
    try:
        # Gambar kontur
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Gambar pusat
        if center is not None:
            cv2.circle(frame, center, 10, (0, 255, 0), -1)
            cv2.circle(frame, center, 15, (255, 255, 255), 2)
            
            # Tampilkan koordinat
            coord_text = f"({center[0]}, {center[1]})"
            cv2.putText(frame, coord_text, (center[0] + 20, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Tampilkan info deteksi
        detect_text = f"{color_name.upper()} Terdeteksi!"
        cv2.putText(frame, detect_text, (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        area_text = f"Area: {int(area)}"
        cv2.putText(frame, area_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    except Exception as e:
        print(f"Error drawing overlay: {e}")
        return frame

def get_color_name_indonesian(color_mode):
    """Mendapatkan nama warna dalam bahasa Indonesia."""
    color_map = {
        cfg.COLOR_DETECT_GREEN: "Hijau",
        cfg.COLOR_DETECT_BLUE: "Biru",
        cfg.COLOR_DETECT_RED: "Merah",
        cfg.COLOR_DETECT_YELLOW: "Kuning"
    }
    return color_map.get(color_mode, "Unknown")

# ==================== ORIGINAL UTILITY FUNCTIONS ====================

def overlay_png(background, overlay, x, y):
    """Menempelkan gambar PNG transparan ke background."""
    try:
        if background is None:
            print("Error: Background is None.")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        if overlay is None:
            return background
        bg_h, bg_w, *bg_channels_list = background.shape
        bg_channels = bg_channels_list[0] if bg_channels_list else 1
        overlay_h, overlay_w, *overlay_channels_list = overlay.shape
        overlay_channels = overlay_channels_list[0] if overlay_channels_list else 1
        if bg_channels == 1:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
            bg_h, bg_w, bg_channels = background.shape
        if overlay_channels < 4:
            if overlay_channels == 3:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
                overlay[:, :, 3] = 255
                overlay_h, overlay_w, overlay_channels = overlay.shape
            else:
                return background
        if x >= bg_w or y >= bg_h or x + overlay_w <= 0 or y + overlay_h <= 0:
            return background
        y1, y2 = max(0, y), min(bg_h, y + overlay_h)
        x1, x2 = max(0, x), min(bg_w, x + overlay_w)
        alpha_h_start = max(0, -y)
        alpha_h_end = y2 - y1 + alpha_h_start
        alpha_w_start = max(0, -x)
        alpha_w_end = x2 - x1 + alpha_w_start
        if y1 >= y2 or x1 >= x2:
            return background
        roi = background[y1:y2, x1:x2]
        if alpha_h_start >= alpha_h_end or alpha_w_start >= alpha_w_end or alpha_h_start < 0 or alpha_h_end > overlay_h or alpha_w_start < 0 or alpha_w_end > overlay_w:
            return background
        overlay_area = overlay[alpha_h_start:alpha_h_end, alpha_w_start:alpha_w_end]
        if roi.size == 0 or overlay_area.size == 0:
            return background
        if roi.shape[:2] != overlay_area.shape[:2]:
            try:
                overlay_area = cv2.resize(overlay_area, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
            except Exception as resize_error:
                print(f"Resize Error: {resize_error}")
                return background
        alpha = overlay_area[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            if roi.shape[2] > c:
                roi[:, :, c] = (alpha * overlay_area[:, :, c] + alpha_inv * roi[:, :, c])
        background[y1:y2, x1:x2] = roi
        return background
    except Exception as e:
        print(f"Overlay Error: {e}")
        return background if background is not None else np.zeros((100, 100, 3), dtype=np.uint8)

def remove_greenscreen(frame):
    """Hapus background hijau dari video dan kembalikan frame BGRA dengan alpha channel."""
    if frame is None or frame.size == 0:
        return None
    
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cfg.CHROMA_KEY_COLOR_LOWER, cfg.CHROMA_KEY_COLOR_UPPER)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.GaussianBlur(mask_inv, (5, 5), 0)
        b, g, r = cv2.split(frame)
        frame_bgra = cv2.merge([b, g, r, mask_inv])
        return frame_bgra
    except Exception as e:
        print(f"Greenscreen removal error: {e}")
        return None

def overlay_video_effect(background, video_frame, position='fullscreen', frame_w=None, frame_h=None):
    """Overlay video effect dengan chroma key ke background."""
    if background is None or video_frame is None:
        return background
    
    bg_h, bg_w = background.shape[:2]
    
    effect_with_alpha = remove_greenscreen(video_frame)
    if effect_with_alpha is None:
        return background
    
    if position == 'fullscreen':
        effect_resized = cv2.resize(effect_with_alpha, (bg_w, bg_h))
        return overlay_png(background, effect_resized, 0, 0)
    
    elif position == 'right':
        effect_h, effect_w = effect_with_alpha.shape[:2]
        target_w = int(bg_w * 0.5)
        target_h = int(effect_h * (target_w / effect_w))
        
        if target_h > bg_h:
            target_h = bg_h
            target_w = int(effect_w * (target_h / effect_h))
        
        effect_resized = cv2.resize(effect_with_alpha, (target_w, target_h))
        
        x_pos = bg_w - target_w - 20
        y_pos = (bg_h - target_h) // 2
        
        return overlay_png(background, effect_resized, x_pos, y_pos)
    
    return background

def draw_fancy_text(img, text, pos, font, scale, color, thickness, bg_color, alpha, padding):
    """Menggambar teks dengan background."""
    try:
        if img is None:
            return img
        if not isinstance(text, str):
            text = str(text)
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = int(pos[0]), int(pos[1])
        top_left = (x, y)
        bottom_right = (x + w + padding * 2, y + h + baseline + padding * 2)
        text_pos = (x + padding, y + h + padding)
        img_h, img_w, _ = img.shape
        y1, y2 = max(0, top_left[1]), min(img_h, bottom_right[1])
        x1, x2 = max(0, top_left[0]), min(img_w, bottom_right[0])
        if y1 >= y2 or x1 >= x2:
            return img
        sub_img = img[y1:y2, x1:x2]
        if sub_img.size == 0:
            return img
        bg_rect = np.full(sub_img.shape, bg_color, dtype=sub_img.dtype)
        if sub_img.shape != bg_rect.shape:
            return img
        res = cv2.addWeighted(sub_img, 1 - alpha, bg_rect, alpha, 0)
        img[y1:y2, x1:x2] = res
        cv2.putText(img, text, text_pos, font, scale, color, thickness, cv2.LINE_AA)
        return img
    except Exception as e:
        print(f"Draw Text Error ('{text}'): {e}")
        return img

def get_aspect_ratio(face_landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """Hitung rasio aspek."""
    if not face_landmarks:
        return 0.0
    lm = face_landmarks.landmark
    def get_p(idx):
        try:
            return (lm[idx].x, lm[idx].y) if idx < len(lm) and lm[idx].visibility > 0.1 else None
        except IndexError:
            return None
    top_p, bottom_p, left_p, right_p = get_p(top_idx), get_p(bottom_idx), get_p(left_idx), get_p(right_idx)
    if None in [top_p, bottom_p, left_p, right_p]:
        return 0.0
    try:
        ver = math.hypot(top_p[0] - bottom_p[0], top_p[1] - bottom_p[1])
        hor = math.hypot(left_p[0] - right_p[0], left_p[1] - right_p[1])
        return ver / hor if hor > 1e-6 else 0.0
    except Exception as e:
        print(f"Aspect Ratio Error: {e}")
        return 0.0

def get_finger_status(hand_landmarks):
    """Deteksi jari lurus."""
    status = {'THUMB': False, 'INDEX': False, 'MIDDLE': False, 'RING': False, 'PINKY': False}
    
    if not hand_landmarks:
        return status
    
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        if thumb_tip.y < thumb_ip.y:
            status['THUMB'] = True
        
        finger_tips_ids = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_pips_ids = [
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP
        ]
        
        finger_names = ['INDEX', 'MIDDLE', 'RING', 'PINKY']
        
        for i, (tip_id, pip_id) in enumerate(zip(finger_tips_ids, finger_pips_ids)):
            finger_name = finger_names[i]
            tip = hand_landmarks.landmark[tip_id]
            pip = hand_landmarks.landmark[pip_id]
            
            if tip.y < pip.y:
                status[finger_name] = True
    
    except (IndexError, AttributeError) as e:
        print(f"Warning: Error di get_finger_status: {e}")
    
    return status