import os
import cv2
import json
import base64
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import face_recognition
import sqlite3
import threading
import time

# ── Twilio (SMS) ──────────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️  Twilio not installed. SMS disabled. Run: pip install twilio")

app = Flask(__name__)
CORS(app)

DB_PATH        = "database/missing_persons.db"
ENCODINGS_PATH = "database/encodings.pkl"
KNOWN_FACES_DIR = "known_faces"

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  —  Fill these in before running
# ══════════════════════════════════════════════════════════════
TWILIO_ACCOUNT_SID  = "YOUR_TWILIO_ACCOUNT_SID"
TWILIO_AUTH_TOKEN   = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_FROM_NUMBER  = "+1XXXXXXXXXX"
ALERT_TO_NUMBER     = "+91XXXXXXXXXX"

SMS_ENABLED = (
    TWILIO_AVAILABLE and
    TWILIO_ACCOUNT_SID  != "YOUR_TWILIO_ACCOUNT_SID" and
    TWILIO_AUTH_TOKEN   != "YOUR_TWILIO_AUTH_TOKEN"
)
# ══════════════════════════════════════════════════════════════


def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS missing_persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            age         INTEGER,
            last_seen   TEXT,
            contact     TEXT,
            description TEXT,
            photo_path  TEXT,
            date_added  TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER,
            person_name TEXT,
            detected_at TEXT,
            confidence  REAL,
            location    TEXT,
            gps_lat     REAL,
            gps_lon     REAL,
            sms_sent    INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def encode_all_known_faces():
    known_encodings, known_names, known_ids = [], [], []
    conn    = get_db()
    persons = conn.execute("SELECT * FROM missing_persons").fetchall()
    conn.close()
    for person in persons:
        photo_path = person["photo_path"]
        if photo_path and os.path.exists(photo_path):
            image = face_recognition.load_image_file(photo_path)
            encs  = face_recognition.face_encodings(image)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(person["name"])
                known_ids.append(person["id"])
    data = {"encodings": known_encodings, "names": known_names, "ids": known_ids}
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)
    return data


def load_encodings():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": [], "ids": []}


def send_sms_alert(person_name, confidence, location, gps_lat=None, gps_lon=None):
    if not SMS_ENABLED:
        print(f"[SMS DISABLED] Would have alerted for: {person_name}")
        return False
    try:
        client    = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
        maps_link = ""
        if gps_lat and gps_lon:
            maps_link = f"\nGPS Map: https://maps.google.com/?q={gps_lat},{gps_lon}"
        body = (
            f"GUARDIAN ALERT - MISSING PERSON FOUND\n"
            f"Name      : {person_name}\n"
            f"Confidence: {confidence}%\n"
            f"Location  : {location}\n"
            f"Time      : {timestamp}"
            f"{maps_link}\n"
            f"Contact authorities immediately."
        )
        msg = client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=ALERT_TO_NUMBER)
        print(f"SMS sent! SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"SMS failed: {e}")
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/persons", methods=["GET"])
def get_persons():
    conn    = get_db()
    persons = conn.execute("SELECT * FROM missing_persons ORDER BY date_added DESC").fetchall()
    conn.close()
    result  = []
    for p in persons:
        d = dict(p)
        if d["photo_path"] and os.path.exists(d["photo_path"]):
            with open(d["photo_path"], "rb") as f:
                d["photo_b64"] = base64.b64encode(f.read()).decode()
        else:
            d["photo_b64"] = None
        result.append(d)
    return jsonify(result)


@app.route("/api/persons", methods=["POST"])
def add_person():
    name        = request.form.get("name")
    age         = request.form.get("age")
    last_seen   = request.form.get("last_seen")
    contact     = request.form.get("contact")
    description = request.form.get("description")
    photo       = request.files.get("photo")
    photo_path  = None
    if photo:
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        filename   = f"{name.replace(' ', '_')}_{int(time.time())}.jpg"
        photo_path = os.path.join(KNOWN_FACES_DIR, filename)
        photo.save(photo_path)
    conn = get_db()
    conn.execute(
        "INSERT INTO missing_persons (name, age, last_seen, contact, description, photo_path, date_added) VALUES (?,?,?,?,?,?,?)",
        (name, age, last_seen, contact, description, photo_path, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    encode_all_known_faces()
    return jsonify({"success": True, "message": f"{name} added to database."})


@app.route("/api/persons/<int:person_id>", methods=["DELETE"])
def delete_person(person_id):
    conn = get_db()
    conn.execute("DELETE FROM missing_persons WHERE id=?", (person_id,))
    conn.commit()
    conn.close()
    encode_all_known_faces()
    return jsonify({"success": True})


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    conn   = get_db()
    alerts = conn.execute("SELECT * FROM alerts ORDER BY detected_at DESC LIMIT 50").fetchall()
    conn.close()
    return jsonify([dict(a) for a in alerts])


@app.route("/api/alerts", methods=["DELETE"])
def clear_alerts():
    conn = get_db()
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/scan_image", methods=["POST"])
def scan_image():
    data_url = request.json.get("image")
    location = request.json.get("location", "Unknown Location")
    gps_lat = request.json.get("gps_lat")
    gps_lon = request.json.get("gps_lon")

    # Decode base64 image
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Load saved face encodings
    encodings_data = load_encodings()

    # Detect faces (CNN model works better for group photos)
    face_locations = face_recognition.face_locations(rgb, model="cnn")

    # Generate encodings for detected faces
    face_encs = face_recognition.face_encodings(rgb, face_locations)

    matches_found = []
    annotated_frame = frame.copy()

    # Compare each detected face
    for (top, right, bottom, left), face_enc in zip(face_locations, face_encs):
        name = "Unknown"
        person_id = None
        confidence = 0
        sms_sent = False

        if encodings_data["encodings"]:
            distances = face_recognition.face_distance(
                encodings_data["encodings"],
                face_enc
            )

            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            print(f"Best match: {encodings_data['names'][best_idx]}, Distance: {best_dist:.4f}")
            confidence = round((1 - best_dist) * 100, 1)

            # Match threshold
            if best_dist < 0.48:
                name = encodings_data["names"][best_idx]
                person_id = encodings_data["ids"][best_idx]

                # Send SMS alert in background
                t = threading.Thread(
                    target=send_sms_alert,
                    args=(name, confidence, location, gps_lat, gps_lon),
                    daemon=True
                )
                t.start()
                sms_sent = SMS_ENABLED

                # Get person details
                conn = get_db()
                person_row = conn.execute(
                    "SELECT * FROM missing_persons WHERE id=?",
                    (person_id,)
                ).fetchone()

                last_known = person_row["last_seen"] if person_row else "Unknown"
                age = person_row["age"] if person_row else "Unknown"
                contact = person_row["contact"] if person_row else "Unknown"
                description = person_row["description"] if person_row else ""

                # Save alert to database
                conn.execute(
                    """
                    INSERT INTO alerts
                    (person_id, person_name, detected_at, confidence,
                     location, gps_lat, gps_lon, sms_sent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person_id,
                        name,
                        datetime.now().isoformat(),
                        confidence,
                        location,
                        gps_lat,
                        gps_lon,
                        int(sms_sent)
                    )
                )
                conn.commit()
                conn.close()

                # Store match details
                matches_found.append({
                    "name": name,
                    "person_id": person_id,
                    "confidence": confidence,
                    "box": [top, right, bottom, left],
                    "last_seen": last_known,
                    "age": age,
                    "contact": contact,
                    "description": description,
                    "sms_sent": sms_sent,
                    "current_location": location,
                    "gps_lat": gps_lat,
                    "gps_lon": gps_lon
                })

        # Draw rectangle around face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(
            annotated_frame,
            (left, top),
            (right, bottom),
            color,
            3
        )

        label = (
            f"{name} ({confidence}%)"
            if name != "Unknown"
            else "Unknown"
        )

        cv2.putText(
            annotated_frame,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # Convert annotated image back to base64
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    annotated_b64 = (
        "data:image/jpeg;base64,"
        + base64.b64encode(buffer).decode()
    )

    # Return results
    return jsonify({
        "matches": matches_found,
        "faces_detected": len(face_locations),
        "annotated_image": annotated_b64,
        "sms_enabled": SMS_ENABLED
    })

@app.route("/api/rebuild_encodings", methods=["POST"])
def rebuild_encodings():
    data = encode_all_known_faces()
    return jsonify({"success": True, "encoded": len(data["encodings"])})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    conn           = get_db()
    total_missing  = conn.execute("SELECT COUNT(*) FROM missing_persons").fetchone()[0]
    total_alerts   = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    recent_alerts  = conn.execute("SELECT COUNT(*) FROM alerts WHERE detected_at > datetime('now', '-24 hours')").fetchone()[0]
    sms_sent_count = conn.execute("SELECT COUNT(*) FROM alerts WHERE sms_sent=1").fetchone()[0]
    conn.close()
    return jsonify({
        "total_missing": total_missing,
        "total_alerts":  total_alerts,
        "alerts_24h":    recent_alerts,
        "sms_sent":      sms_sent_count,
        "sms_enabled":   SMS_ENABLED,
    })


@app.route("/api/sms_status", methods=["GET"])
def sms_status():
    return jsonify({
        "sms_enabled":      SMS_ENABLED,
        "twilio_available": TWILIO_AVAILABLE,
        "alert_to":         ALERT_TO_NUMBER if SMS_ENABLED else "Not configured",
    })


if __name__ == "__main__":
    init_db()
    encode_all_known_faces()
    print("\n" + "="*50)
    print("  GUARDIAN — Missing Person Alert System")
    print("="*50)
    print(f"  SMS Alerts : {'ENABLED' if SMS_ENABLED else 'DISABLED (configure Twilio keys)'}")
    print(f"  Dashboard  : http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
