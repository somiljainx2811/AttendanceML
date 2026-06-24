import os
import face_recognition
import numpy as np


class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", tolerance=0.5):
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        self.known_encodings = []
        self.known_names = []

        if not os.path.isdir(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)
            return

        for person_name in sorted(os.listdir(self.known_faces_dir)):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for filename in sorted(os.listdir(person_dir)):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                image_path = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        # if a reference photo has multiple faces, use the first
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(person_name)
                    else:
                        print(f"[WARN] No face found in {image_path}, skipping.")
                except Exception as exc:
                    print(f"[WARN] Could not process {image_path}: {exc}")

        print(
            f"[INFO] Loaded {len(self.known_encodings)} face encodings "
            f"for {len(set(self.known_names))} people."
        )

    def recognize_faces(self, frame_rgb_small):
        face_locations = face_recognition.face_locations(frame_rgb_small)
        face_encodings = face_recognition.face_encodings(frame_rgb_small, face_locations)

        results = []
        for location, encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0.0

            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_idx = int(np.argmin(distances))
                best_distance = distances[best_idx]

                if best_distance <= self.tolerance:
                    name = self.known_names[best_idx]
                    confidence = max(0.0, 1.0 - best_distance)

            results.append((name, location, confidence))

        return results
