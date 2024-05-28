import tkinter as tk
from tkinter import filedialog
import cv2

class ObjectDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Reconnaissance et détection d'objets")

        self.video_running = False
        self.video_capture = None
        self.stop_video_loop = False

        self.label = tk.Label(master, text="Menu:", font=("Arial", 20), fg="blue")
        self.label.pack(pady=(20, 10))

        self.img_button = tk.Button(master, text="Importer une image", command=self.import_image, font=("Arial", 16), bg="green", fg="white", padx=20, pady=10)
        self.img_button.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.video_button = tk.Button(master, text="Importer une vidéo", command=self.import_video, font=("Arial", 16), bg="blue", fg="white", padx=20, pady=10)
        self.video_button.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.cam_button = tk.Button(master, text="Démarrer la caméra", command=self.start_camera, font=("Arial", 16), bg="orange", fg="white", padx=20, pady=10)
        self.cam_button.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.quit_button = tk.Button(master, text="Fermer l'application", command=self.close_app, font=("Arial", 16), bg="red", fg="white", padx=20, pady=10)
        self.quit_button.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.stop_video_button = tk.Button(master, text="Arrêter la détection vidéo", command=self.stop_video, font=("Arial", 16), bg="red", fg="white", padx=20, pady=10)

        self.center_window()

    def center_window(self):
        # Centre la fenêtre principale sur l'écran
        window_width = self.master.winfo_reqwidth()
        window_height = self.master.winfo_reqheight()

        position_right = int(self.master.winfo_screenwidth() / 2 - window_width / 2)
        position_down = int(self.master.winfo_screenheight() / 2 - window_height / 2)

        self.master.geometry("+{}+{}".format(position_right, position_down))

    def import_image(self):
        img_path = filedialog.askopenfilename(title="Select an image")
        if img_path:
            image = cv2.imread(img_path)
            self.detect_objects(image)
            self.master.geometry("{}x{}".format(image.shape[1], image.shape[0]))
            self.show_menu()

    def import_video(self):
        video_path = filedialog.askopenfilename(title="Select a video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if video_path:
            video_capture = cv2.VideoCapture(video_path)
            if video_capture.isOpened():
                width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.master.geometry("{}x{}".format(width, height))
                self.detect_and_show_video(video_capture)
                self.show_menu()

    def start_camera(self):
        video_capture = cv2.VideoCapture(0)
        if video_capture.isOpened():
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.master.geometry("{}x{}".format(width, height))
            self.detect_and_show_video(video_capture)
            self.show_menu()

    def stop_video(self):
        self.stop_video_loop = True

    def detect_and_show_video(self, video_capture):
        self.stop_video_loop = False
        while not self.stop_video_loop and video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                self.detect_objects(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        video_capture.release()
        cv2.destroyAllWindows()

    def detect_objects(self, img):
        classNames = []
        classFile = 'object.names'

        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightpath = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightpath, configPath)
        net.setInputSize(320, 230)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        cv2.imshow('Output', img)

    def show_menu(self):
        self.label.config(text="Menu:")
        self.img_button.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.video_button.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.cam_button.pack(fill=tk.X, padx=20, pady=(0, 10))
        if self.video_running:
            self.stop_video_button.pack(fill=tk.X, padx=20, pady=(0, 10))
        else:
            self.stop_video_button.pack_forget()

    def close_app(self):
        self.stop_video()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
