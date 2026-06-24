import csv
import os
from datetime import datetime


class AttendanceLogger:
    def __init__(self, csv_path="attendance.csv"):
        self.csv_path = csv_path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])

    def _read_all_rows(self):
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def is_marked_today(self, name):
        today = datetime.now().strftime("%Y-%m-%d")
        return any(row["Name"] == name and row["Date"] == today for row in self._read_all_rows())

    def mark_attendance(self, name):
        if name == "Unknown" or not name:
            return False

        if self.is_marked_today(name):
            return False

        now = datetime.now()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
        return True

    def get_today_records(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return [row for row in self._read_all_rows() if row["Date"] == today]

    def get_all_records(self):
        return self._read_all_rows()
