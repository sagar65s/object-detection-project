import cv2
import face_recognition
from ultralytics import YOLO
import winsound
import time
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from twilio.rest import Client

# ------------------------------
# 1. YOLOv8 model
# ------------------------------
model = YOLO("yolov8s.pt")  # make sure yolov8s.pt is in your folder

# ------------------------------
# 2. Known Face Encodings
# ------------------------------
known_face_encodings = []
known_face_names = []

owners = [
    ("me.jpg", "Owner1"),
    ("me.jpg", "Owner2"),
    ("me.jpg", "Owner3"),
    ("me.jpg", "Owner4")
]

for image_path, name in owners:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# ------------------------------
# 3. Non-blocking beep
# ------------------------------
def play_beep():
    winsound.Beep(1000, 50)  # 1000Hz for 5 seconds

# ------------------------------
# 4. Email Notification
# ------------------------------
def send_email_notification(image_path):
    sender_email = "myselfuse12355@gmail.com"
    receiver_email = "devimugesh1@gmail.com"
    password = "aaaabbbbccccdddd"  # Gmail App Password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "⚠ Unknown Person Detected!"

    body = "An unknown person has been detected. See the attached image."
    msg.attach(MIMEText(body, 'plain'))

    attachment = open(image_path, "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload(attachment.read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', f"attachment; filename= {image_path}")
    msg.attach(p)
    attachment.close()

    try:
        server = smtplib.SMTP('smtp.gmail.com', 465)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# ------------------------------
# 5. SMS Notification
# ------------------------------
def send_sms_notification():
    account_sid = 'AC3275ae477816c6075212574e69dcd607'
    auth_token = 'bc34c062c79c14f37f9b8a83a9747f73'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body="⚠ Unknown person detected by your security system!",
        from_='+15088599511',  # Your Twilio number
        to='+916379083994'    # Your phone number
    )
    print("SMS sent successfully!")

# ------------------------------
# 6. Camera Loop
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Night Vision Boost
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Face Detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    unknown_detected = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if name == "Unknown":
            unknown_detected = True

    # YOLO Detection + Notifications
    if unknown_detected:
        threading.Thread(target=play_beep, daemon=True).start()
        results = model(frame, conf=0.25)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save frame
        filename = f"unknown_detected_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"⚠ Unknown detected! Frame saved: {filename}")

        # Send notifications in threads
        threading.Thread(target=send_email_notification, args=(filename,), daemon=True).start()
        threading.Thread(target=send_sms_notification, daemon=True).start()

    # Show video
    cv2.imshow("Face + YOLOv8 Security", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
