#!/usr/bin/env python3
"""
Color Calibration Tool untuk VTuber
Digunakan untuk mencari nilai HSV yang tepat untuk objek berwarna Anda.

Cara pakai:
1. Jalankan: python color_calibration.py
2. KLIK pada objek berwarna di webcam
3. Lihat nilai HSV di terminal dan di layar
4. Adjust slider H, S, V untuk fine-tune detection
5. Copy nilai Lower dan Upper ke config.py
"""

import cv2
import numpy as np

# Global variables
current_hsv = None
clicked_point = None

# Default HSV ranges (will be updated based on clicks)
h_lower, h_upper = 0, 180
s_lower, s_upper = 50, 255
v_lower, v_upper = 50, 255

def mouse_callback(event, x, y, flags, param):
    """Callback untuk mouse click - mendapatkan nilai HSV dari pixel yang diklik."""
    global current_hsv, clicked_point, h_lower, h_upper, s_lower, s_upper, v_lower, v_upper
    
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param['hsv']
        if hsv_frame is not None:
            # Ambil nilai HSV dari pixel yang diklik
            h, s, v = hsv_frame[y, x]
            current_hsv = (h, s, v)
            clicked_point = (x, y)
            
            # Auto-adjust range berdasarkan nilai yang diklik
            # Hue: ±10 dari nilai yang diklik
            h_lower = max(0, h - 15)
            h_upper = min(180, h + 15)
            
            # Saturation: dari 50% nilai diklik sampai 255
            s_lower = max(30, s - 30)
            s_upper = 255
            
            # Value: dari 50% nilai diklik sampai 255
            v_lower = max(30, v - 30)
            v_upper = 255
            
            print("\n" + "="*60)
            print(f"📍 Pixel ({x}, {y}) diklik!")
            print(f"🎨 HSV Value: H={h}, S={s}, V={v}")
            print(f"📊 Suggested Range:")
            print(f"   Lower: [{h_lower}, {s_lower}, {v_lower}]")
            print(f"   Upper: [{h_upper}, {s_upper}, {v_upper}]")
            print("="*60)
            
            # Update trackbars
            cv2.setTrackbarPos('H Lower', 'Controls', h_lower)
            cv2.setTrackbarPos('H Upper', 'Controls', h_upper)
            cv2.setTrackbarPos('S Lower', 'Controls', s_lower)
            cv2.setTrackbarPos('S Upper', 'Controls', s_upper)
            cv2.setTrackbarPos('V Lower', 'Controls', v_lower)
            cv2.setTrackbarPos('V Upper', 'Controls', v_upper)

def nothing(x):
    """Dummy callback untuk trackbar."""
    pass

def main():
    global h_lower, h_upper, s_lower, s_upper, v_lower, v_upper
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka webcam!")
        return
    
    # Buat window
    cv2.namedWindow('Original Frame')
    cv2.namedWindow('HSV Frame')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Controls')
    
    # Buat trackbars untuk adjust range
    cv2.createTrackbar('H Lower', 'Controls', h_lower, 180, nothing)
    cv2.createTrackbar('H Upper', 'Controls', h_upper, 180, nothing)
    cv2.createTrackbar('S Lower', 'Controls', s_lower, 255, nothing)
    cv2.createTrackbar('S Upper', 'Controls', s_upper, 255, nothing)
    cv2.createTrackbar('V Lower', 'Controls', v_lower, 255, nothing)
    cv2.createTrackbar('V Upper', 'Controls', v_upper, 255, nothing)
    
    print("\n" + "="*60)
    print("🎨 COLOR CALIBRATION TOOL")
    print("="*60)
    print("📌 Instruksi:")
    print("  1. KLIK pada objek berwarna di window 'Original Frame'")
    print("  2. Lihat nilai HSV yang terdeteksi di terminal")
    print("  3. Adjust slider di window 'Controls' untuk fine-tune")
    print("  4. Lihat hasil deteksi di window 'Mask' (putih = terdeteksi)")
    print("  5. Tekan 'q' untuk keluar")
    print("  6. Tekan 's' untuk save konfigurasi ke file")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error membaca frame dari webcam")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Konversi ke HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Setup mouse callback dengan parameter HSV frame
        cv2.setMouseCallback('Original Frame', mouse_callback, {'hsv': hsv})
        
        # Get current trackbar positions
        h_lower = cv2.getTrackbarPos('H Lower', 'Controls')
        h_upper = cv2.getTrackbarPos('H Upper', 'Controls')
        s_lower = cv2.getTrackbarPos('S Lower', 'Controls')
        s_upper = cv2.getTrackbarPos('S Upper', 'Controls')
        v_lower = cv2.getTrackbarPos('V Lower', 'Controls')
        v_upper = cv2.getTrackbarPos('V Upper', 'Controls')
        
        # Buat mask berdasarkan range HSV
        lower = np.array([h_lower, s_lower, v_lower])
        upper = np.array([h_upper, s_upper, v_upper])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations untuk cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw pada original frame
        display_frame = frame.copy()
        
        # Draw crosshair di clicked point
        if clicked_point:
            x, y = clicked_point
            cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2)
            cv2.line(display_frame, (x-20, y), (x+20, y), (0, 255, 0), 2)
            cv2.line(display_frame, (x, y-20), (x, y+20), (0, 255, 0), 2)
            
            if current_hsv:
                h, s, v = current_hsv
                text = f"H:{h} S:{s} V:{v}"
                cv2.putText(display_frame, text, (x+15, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw contours dan info
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 500:  # Minimum area threshold
                cv2.drawContours(display_frame, [largest_contour], -1, (0, 255, 0), 2)
                
                # Calculate center
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_frame, (cx, cy), 7, (255, 0, 0), -1)
                    
                    # Display area
                    cv2.putText(display_frame, f"Area: {int(area)}", (cx-50, cy-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display current range
        range_text = f"Range: H[{h_lower}-{h_upper}] S[{s_lower}-{s_upper}] V[{v_lower}-{v_upper}]"
        cv2.putText(display_frame, range_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show windows
        cv2.imshow('Original Frame', display_frame)
        cv2.imshow('HSV Frame', hsv)
        cv2.imshow('Mask', mask)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Keluar dari program...")
            break
        elif key == ord('s'):
            # Save configuration
            print("\n" + "="*60)
            print("💾 KONFIGURASI UNTUK config.py:")
            print("="*60)
            print(f"HSV_YOUR_COLOR_LOWER = np.array([{h_lower}, {s_lower}, {v_lower}])")
            print(f"HSV_YOUR_COLOR_UPPER = np.array([{h_upper}, {s_upper}, {v_upper}])")
            print("="*60)
            print("✅ Copy kode di atas ke config.py Anda!")
            print("="*60 + "\n")
            
            # Save to file
            with open('color_config_result.txt', 'w') as f:
                f.write("# Hasil Kalibrasi Warna\n")
                f.write("# Copy kode berikut ke config.py\n\n")
                f.write(f"HSV_YOUR_COLOR_LOWER = np.array([{h_lower}, {s_lower}, {v_lower}])\n")
                f.write(f"HSV_YOUR_COLOR_UPPER = np.array([{h_upper}, {s_upper}, {v_upper}])\n")
            
            print("📝 Konfigurasi juga disimpan ke: color_config_result.txt")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✨ Program selesai!")

if __name__ == "__main__":
    main()