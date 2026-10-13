import subprocess
import time
from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

end_sign_detected = False
last_state = "idle"  # "idle" -> "ready" (tangan terbuka) -> trigger -> "idle"

CONFIRM_FRAMES = 10  # jumlah frame berturut-turut yang harus sepakat sebelum trigger
sign_buffer = deque(maxlen=CONFIRM_FRAMES)

appDict = {
    "steam": "C:\\Program Files (x86)\\Steam\\steam.exe",
    "krita": "C:\\Program Files\\Krita (x64)\\bin\\krita.exe",
    "blender": "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Blender\\blender.exe",
    "vscode": "C:\\Users\\muh4l\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
    "roblox": "C:\\Users\\muh4l\\AppData\\Local\\Roblox\\Versions\\version-c5aecda2245e4fae\\RobloxPlayerBeta.exe",
    "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "thunderbird": "C:\\Program Files\\Mozilla Thunderbird\\thunderbird.exe"
}

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ibu jari
    (0, 5), (5, 6), (6, 7), (7, 8),        # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),   # Jari tengah
    (9, 13), (13, 14), (14, 15), (15, 16), # Jari manis
    (13, 17), (17, 18), (18, 19), (19, 20),# Kelingking
    (0, 17)                                 # Pergelangan ke pangkal kelingking
]

# Index landmark MediaPipe (referensi):
# 0=wrist, 4=thumb_tip, 3=thumb_ip, 2=thumb_mcp
# 8=index_tip, 6=index_pip
# 12=middle_tip, 10=middle_pip
# 16=ring_tip, 14=ring_pip
# 20=pinky_tip, 18=pinky_pip

EXTEND_MARGIN = 0.02   # toleransi untuk jari (selain jempol), koordinat ternormalisasi 0-1
THUMB_MARGIN = 0.02    # toleransi khusus jempol (berbasis jarak, bukan sumbu Y)

def finger_state(tip, ref, margin=EXTEND_MARGIN):
    """
    Bandingkan tip terhadap titik referensi (pip) untuk jari selain jempol.
    Return 1 = terentang, 0 = tertekuk, -1 = ambigu (terlalu dekat garis batas -> diabaikan).
    """
    diff = ref.y - tip.y
    if diff > margin:
        return 1
    elif diff < -margin:
        return 0
    return -1


def distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def thumb_state(landmarks, margin=THUMB_MARGIN):
    """
    Jempol melipat menyilang ke arah telapak (bukan turun secara vertikal seperti jari lain),
    jadi dicek lewat JARAK ujung jempol ke pangkal kelingking (landmark 17):
    - Menjauh dari pangkal kelingking (dibanding posisi pangkal jempol) -> terentang (1)
    - Mendekat ke pangkal kelingking (menempel telapak)                -> tertekuk (0)
    """
    thumb_tip, thumb_mcp, pinky_mcp = landmarks[4], landmarks[2], landmarks[17]
    diff = distance(thumb_tip, pinky_mcp) - distance(thumb_mcp, pinky_mcp)
    if diff > margin:
        return 1
    elif diff < -margin:
        return 0
    return -1


def is_open(landmarks):
    # Gestur "arming": semua jari terentang (tangan terbuka), tanpa ambigu
    states = [
        thumb_state(landmarks),
        finger_state(landmarks[8], landmarks[6]),   # index
        finger_state(landmarks[12], landmarks[10]), # middle
        finger_state(landmarks[16], landmarks[14]), # ring
        finger_state(landmarks[20], landmarks[18]), # pinky
    ]
    return all(s == 1 for s in states)


# Vektor status: (thumb, index, middle, ring, pinky) -> 1 = terentang, 0 = tertekuk
# Dicocokkan secara EXACT: kalau tangan tidak persis membentuk salah satu pola ini,
# hasilnya "unknown" -- tidak akan "jatuh" ke gestur lain yang kebetulan mirip sebagian.
GESTURE_MAP = {
    (1, 0, 0, 0, 0): "steam",        # jempol lurus ke atas saja
    (0, 1, 0, 0, 1): "krita",        # rock sign: telunjuk + kelingking
    (0, 1, 1, 0, 0): "blender",      # peace sign: telunjuk + tengah
    (0, 1, 1, 1, 0): "vscode",       # tiga jari tengah: telunjuk+tengah+manis
    (1, 1, 0, 0, 1): "chrome",       # ASL "I Love You": jempol+telunjuk+kelingking
    (1, 1, 0, 0, 0): "roblox",       # bentuk L: jempol + telunjuk
    (0, 0, 1, 1, 1): "thunderbird",  # bentuk ok: tengah + manis + kelingking
}


def rec_sign(landmarks):
    thumb_tip, thumb_mcp = landmarks[4], landmarks[2]
    index_tip, index_pip = landmarks[8], landmarks[6]
    middle_tip, middle_pip = landmarks[12], landmarks[10]
    ring_tip, ring_pip = landmarks[16], landmarks[14]
    pinky_tip, pinky_pip = landmarks[20], landmarks[18]

    # END dicek terpisah dulu: shaka sign (jempol menyamping + kelingking naik)
    index_curl = finger_state(index_tip, index_pip) == 0
    middle_curl = finger_state(middle_tip, middle_pip) == 0
    ring_curl = finger_state(ring_tip, ring_pip) == 0
    pinky_ext = finger_state(pinky_tip, pinky_pip) == 1

    if thumb_tip.x > thumb_mcp.x and pinky_ext and index_curl and middle_curl and ring_curl:
        return "end_sign"

    vector = (
        thumb_state(landmarks),
        finger_state(index_tip, index_pip),
        finger_state(middle_tip, middle_pip),
        finger_state(ring_tip, ring_pip),
        finger_state(pinky_tip, pinky_pip),
    )

    if -1 in vector:
        return "unknown"  # posisi jari ambigu, jangan tebak

    return GESTURE_MAP.get(vector, "unknown")


def detect_hand_gesture(image):
    global last_state, end_sign_detected

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = image.shape
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)
            for x, y in points:
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

            if is_open(hand_landmarks):
                last_state = "ready"
                sign_buffer.clear()  # tangan terbuka = mulai ulang pengumpulan konfirmasi
                cv2.putText(image, "Idle (ready)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            elif last_state == "ready":
                detected_sign = rec_sign(hand_landmarks)
                sign_buffer.append(detected_sign)

                is_confirmed = (
                    len(sign_buffer) == CONFIRM_FRAMES
                    and len(set(sign_buffer)) == 1
                    and sign_buffer[0] != "unknown"
                )

                if is_confirmed:
                    confirmed_sign = sign_buffer[0]
                    sign_buffer.clear()

                    if confirmed_sign in appDict:
                        subprocess.Popen([appDict[confirmed_sign]])
                        print(f"Membuka: {confirmed_sign}")
                        last_state = "idle"  # wajib buka tangan lagi sebelum trigger berikutnya
                        cv2.putText(image, f"Trigger: {confirmed_sign}", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    elif confirmed_sign == "end_sign":
                        end_sign_detected = True
                else:
                    # progres konfirmasi, mis. "krita (6/10)"
                    stable_count = 0
                    for s in reversed(sign_buffer):
                        if s == detected_sign:
                            stable_count += 1
                        else:
                            break
                    cv2.putText(image, f"{detected_sign} ({stable_count}/{CONFIRM_FRAMES})", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            else:
                cv2.putText(image, "Buka tangan dulu", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    return image


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: kamera tidak dpt terbuka")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        continue

    frame = detect_hand_gesture(frame)

    cv2.imshow("Hand Gesture Recognition", frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or end_sign_detected:
        break

cap.release()
cv2.destroyAllWindows()