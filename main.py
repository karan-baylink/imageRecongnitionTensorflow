import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from collections import deque
from moviepy.editor import VideoFileClip
from wide_resnet import WideResNet
import threading
import time

# Video paths dictionary remains the same
video_paths = {
    ('Male', '(0-2)'): 'male_0-2.mp4',
    ('Male', '(4-6)'): 'male_4-6.mp4',
    ('Male', '(8-12)'): 'male_8-12.mp4',
    ('Male', '(15-20)'): 'male_15-20.mp4',
    ('Male', '(25-32)'): 'male_25-32.mp4',
    ('Male', '(38-43)'): 'male_38-43.mp4',
    ('Male', '(48-53)'): 'male_60+.mp4',
    ('Male', '(60+)'): 'male_60+.mp4',
    ('Female', '(0-2)'): 'female_0-2.mp4',
    ('Female', '(4-6)'): 'female_4-6.mp4',
    ('Female', '(8-12)'): 'female_8-12.mp4',
    ('Female', '(15-20)'): 'female_15-20.mp4',
    ('Female', '(25-32)'): 'female_25-32.mp4',
    ('Female', '(38-43)'): 'female_38-43.mp4',
    ('Female', '(48-53)'): 'female_60+.mp4',
    ('Female', '(60+)'): 'female_60+.mp4'
}

class AdManager:
    def __init__(self):
        self.ad_queue = deque()
        self.current_ad = None
        self.is_playing = False
        self.ad_window = None

    def add_ad(self, ad_path):
        if ad_path not in self.ad_queue and ad_path != self.current_ad:
            self.ad_queue.append(ad_path)
        if not self.is_playing and self.ad_window and self.ad_window.is_active:
            self.play_next_ad()

    def play_next_ad(self):
        if self.ad_queue and self.ad_window and self.ad_window.is_active:
            self.current_ad = self.ad_queue.popleft()
            self.is_playing = True
            self.ad_window.play_ad(self.current_ad)

    def ad_finished(self):
        self.is_playing = False
        self.current_ad = None
        self.play_next_ad()

    def set_ad_window(self, window):
        self.ad_window = window

class AdPlayerWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel()
        self.window.title("Ad Player")
        
        self.canvas = tk.Canvas(self.window, width=400, height=300)
        self.canvas.pack()
        
        self.controls_frame = ttk.Frame(self.window)
        self.controls_frame.pack(pady=10)
        
        self.btn_start = ttk.Button(self.controls_frame, text="Start Ad Player", command=self.start_player)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(self.controls_frame, text="Stop Ad Player", command=self.stop_player, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.is_active = False
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_player(self):
        self.is_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

    def stop_player(self):
        self.is_active = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def play_ad(self, ad_path):
        def play_video():
            try:
                clip = VideoFileClip(ad_path).resize((400, 300)).set_duration(10)
                for frame in clip.iter_frames():
                    if not self.is_active:
                        break
                    photo = ImageTk.PhotoImage(Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.canvas.image = photo
                    self.window.update()
                    time.sleep(1/clip.fps)
                clip.close()
                ad_manager.ad_finished()
            except Exception as e:
                print(f"Error playing video: {e}")

        if self.is_active:
            threading.Thread(target=play_video, daemon=True).start()

    def on_closing(self):
        self.stop_player()
        self.window.destroy()
        self.parent.ad_window = None
        ad_manager.set_ad_window(None)

class FaceRecognitionWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel()
        self.window.title("Face Recognition System")

        self.face_size = 64
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(self.window, width=640, height=480)
        self.canvas.pack()

        self.controls_frame = ttk.Frame(self.window)
        self.controls_frame.pack(pady=10)

        self.btn_start = ttk.Button(self.controls_frame, text="Start Camera", command=self.start_recognition)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(self.controls_frame, text="Stop Camera", command=self.stop_recognition, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        self.model = self.load_model(depth=16, width=8)

        self.is_running = False
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_model(self, depth, width):
        model = WideResNet(self.face_size, depth=depth, k=width)()
        model_dir = os.path.dirname(os.path.abspath(__file__))
        fpath = os.path.join(model_dir, "weights.18-4.06.hdf5")
        model.load_weights(fpath)
        return model

    def crop_face(self, imgarray, section, margin=40, size=64):
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
            
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def draw_label(self, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def start_recognition(self):
        if not self.vid.isOpened():
            self.vid = cv2.VideoCapture(self.video_source)
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.recognize_face()

    def stop_recognition(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def recognize_face(self):
        if self.is_running and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(self.face_size, self.face_size)
                )

                if len(faces) > 0:
                    face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                    for i, face in enumerate(faces):
                        face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                        (x, y, w, h) = cropped
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                        face_imgs[i, :, :, :] = face_img

                    if len(face_imgs) > 0:
                        results = self.model.predict(face_imgs)
                        predicted_genders = results[0]
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()

                        for i, face in enumerate(faces):
                            age = int(predicted_ages[i])
                            gender = "Female" if predicted_genders[i][0] > 0.5 else "Male"
                            age_range = self.get_age_range(age)
                            label = f"{age}, {gender}"
                            self.draw_label(frame, (face[0], face[1]), label)

                            ad_key = (gender, age_range)
                            if ad_key in video_paths:
                                ad_path = os.path.abspath(video_paths[ad_key])
                                ad_manager.add_ad(ad_path)

                # Convert the frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            self.window.after(10, self.recognize_face)

    def get_age_range(self, age):
        if age <= 2: return "(0-2)"
        elif age <= 6: return "(4-6)"
        elif age <= 12: return "(8-12)"
        elif age <= 20: return "(15-20)"
        elif age <= 32: return "(25-32)"
        elif age <= 43: return "(38-43)"
        elif age <= 53: return "(48-53)"
        else: return "(60+)"

    def on_closing(self):
        self.stop_recognition()
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()
        self.parent.face_window = None

class MainApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition Ad System - Control Panel")
        self.root.geometry("300x150")

        ttk.Button(self.root, text="Open Face Recognition", 
                  command=self.open_face_recognition).pack(pady=10)
        ttk.Button(self.root, text="Open Ad Player", 
                  command=self.open_ad_player).pack(pady=10)
        ttk.Button(self.root, text="Exit", 
                  command=self.on_exit).pack(pady=10)

        self.face_window = None
        self.ad_window = None

    def open_face_recognition(self):
        if not self.face_window:
            self.face_window = FaceRecognitionWindow(self)

    def open_ad_player(self):
        if not self.ad_window:
            self.ad_window = AdPlayerWindow(self)
            ad_manager.set_ad_window(self.ad_window)

    def on_exit(self):
        if self.face_window:
            self.face_window.on_closing()
        if self.ad_window:
            self.ad_window.on_closing()
        self.root.quit()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    ad_manager = AdManager()
    app = MainApplication()
    app.run()