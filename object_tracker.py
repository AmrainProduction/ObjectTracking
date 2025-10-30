import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImprovedObjectTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Improved Object Tracker")
        self.root.geometry("1500x800")
        
        # Переменные
        self.template = None
        self.template_gray = None
        self.cap = None
        self.is_tracking = False
        self.tracker = None
        self.current_frame = None
        self.detection_method = "Feature Matching"  # По умолчанию
        
        self.setup_ui()
        
    def setup_ui(self):
        # Основной фрейм
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Верхняя панель управления
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.load_btn = tk.Button(control_frame, text="Загрузить фото", command=self.load_template)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = tk.Button(control_frame, text="Запуск", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="Остановить", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Выбор метода обнаружения
        tk.Label(control_frame, text="Метод обнаружения:").pack(side=tk.LEFT, padx=10)
        self.method_var = tk.StringVar(value=self.detection_method)
        methods = ["Feature Matching", "Template Matching", "Color Detection"]
        self.method_menu = ttk.Combobox(control_frame, textvariable=self.method_var, 
                                       values=methods, state="readonly", width=15)
        self.method_menu.pack(side=tk.LEFT, padx=5)
        self.method_menu.bind('<<ComboboxSelected>>', self.on_method_change)
        
        self.status_label = tk.Label(control_frame, text="Статус: Загрузите шаблон и запустите камеру")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Основная область контента
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель - видео
        left_frame = tk.LabelFrame(content_frame, text="Вебкамера", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_canvas = tk.Canvas(left_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_text = self.video_canvas.create_text(320, 240, 
                                                        text="Камера начнет работу здесь\n\nНажмите 'Запуск'", 
                                                        fill="white", font=("Arial", 14), justify=tk.CENTER)
        
        # Правая панель - шаблон и информация
        right_frame = tk.LabelFrame(content_frame, text="Шаблон и информация", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        self.template_label = tk.Label(right_frame, text="Шаблон не загружен\n\nНажмите 'Загрузить фото'", 
                                      bg="lightgray", width=200, height=200)
        self.template_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Информация о детекции
        info_frame = tk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=30, pady=2, font=("Arial", 10))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.insert(tk.END, "Информация о детекции:\n\n")
        self.info_text.config(state=tk.DISABLED)
        
        # События мыши
        self.video_canvas.bind("<Button-1>", self.on_mouse_down)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.temp_rect = None
        
        # Для feature matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.kp_template = None
        self.des_template = None
        
    def on_method_change(self, event):
        self.detection_method = self.method_var.get()
        self.status_label.config(text=f"Метод изменен на: {self.detection_method}")
        
    def load_template(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
        )
        if file_path:
            try:
                # Загружаем изображение
                self.template = cv2.imread(file_path)
                if self.template is None:
                    messagebox.showerror("Ошибка", "Не удалось загрузить изображение!")
                    return
                
                # Конвертируем в grayscale для трекинга
                self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                
                # Для feature matching
                self.kp_template, self.des_template = self.orb.detectAndCompute(self.template_gray, None)
                
                # Показываем шаблон
                template_rgb = cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(template_rgb)
                img = img.resize((250, 200), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                
                self.template_label.configure(image=img_tk, text="")
                self.template_label.image = img_tk
                
                # Обновляем информацию
                self.update_info(f"Шаблон загружен:\nРазмер: {self.template.shape[1]}x{self.template.shape[0]}\nОсобенности: {len(self.kp_template) if self.kp_template else 0}")
                
                self.status_label.config(text="Шаблон загружен - Нажмите 'Запустить камеру'")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")
    
    def update_info(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"Информация о детекции:\n\n{text}")
        self.info_text.config(state=tk.DISABLED)
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось получить доступ к веб-камере!")
                return
        
        # Ждем стабилизации камеры
        for _ in range(10):
            ret, frame = self.cap.read()
        
        self.is_tracking = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.video_canvas.delete(self.canvas_text)
        
        if self.template is not None:
            self.status_label.config(text=f"Камера запущена - Используется {self.detection_method}")
        else:
            self.status_label.config(text="Камера запущена - Загрузите шаблон для трекинга")

        self.update_frame()
    
    def stop_camera(self):
        self.is_tracking = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Камера остановлена")
        
        self.video_canvas.delete("all")
        self.canvas_text = self.video_canvas.create_text(320, 240,
                                                        text="Камера остановлена\n\nНажмите 'Запустить камеру'",
                                                        fill="white", font=("Arial", 14), justify=tk.CENTER)
    
    def create_tracker(self):
        """Создает трекер с обработкой ошибок"""
        try:
            # Пробуем разные трекеры
            tracker = cv2.TrackerCSRT_create()
            return tracker, "CSRT"
        except:
            try:
                tracker = cv2.TrackerKCF_create()
                return tracker, "KCF"
            except:
                try:
                    tracker = cv2.TrackerMIL_create()
                    return tracker, "MIL"
                except:
                    return None, "None"
    
    def detect_with_features(self, frame):
        """Обнаружение через feature matching (лучше для лиц и текстур)"""
        if self.des_template is None:
            return False, None, 0
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Находим ключевые точки в текущем кадре
        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)
        
        if des_frame is None:
            return False, None, 0
        
        # Сопоставляем features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des_template, des_frame)
        
        if len(matches) < 10:  # Слишком мало совпадений
            return False, None, len(matches)
        
        # Сортируем matches по качеству
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Берем лучшие matches
        good_matches = matches[:30]
        
        # Получаем точки для homography
        src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        try:
            # Находим homography матрицу
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Получаем bounding box из homography
                h, w = self.template_gray.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Преобразуем в прямоугольник
                x_coords = dst[:, 0, 0]
                y_coords = dst[:, 0, 1]
                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                
                # Проверяем валидность bbox
                if (x2 - x1 > 20 and y2 - y1 > 20 and 
                    x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]):
                    
                    bbox = (x1, y1, x2-x1, y2-y1)
                    confidence = len(good_matches) / 30.0  # Нормализованная уверенность
                    
                    return True, bbox, confidence
        except:
            pass
        
        return False, None, len(good_matches)
    
    def detect_with_template(self, frame):
        """Простое template matching"""
        if self.template_gray is None:
            return False, None, 0
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale template matching
        found = None
        for scale in np.linspace(0.5, 1.5, 5):
            width = int(self.template_gray.shape[1] * scale)
            height = int(self.template_gray.shape[0] * scale)
            
            if width > gray.shape[1] or height > gray.shape[0]:
                continue
                
            resized = cv2.resize(self.template_gray, (width, height))
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if found is None or max_val > found[0]:
                found = (max_val, max_loc, scale)
        
        if found and found[0] > 0.6:
            max_val, max_loc, scale = found
            w = int(self.template_gray.shape[1] * scale)
            h = int(self.template_gray.shape[0] * scale)
            
            # Проверяем валидность bbox
            if (max_loc[0] >= 0 and max_loc[1] >= 0 and 
                max_loc[0] + w < frame.shape[1] and max_loc[1] + h < frame.shape[0]):
                
                bbox = (max_loc[0], max_loc[1], w, h)
                return True, bbox, max_val
        
        return False, None, found[0] if found else 0
    
    def detect_with_color(self, frame):
        """Обнаружение по цвету (для объектов с distinct color)"""
        if self.template is None:
            return False, None, 0
            
        # Анализируем доминирующий цвет в шаблоне
        hsv_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        
        # Усредняем цвет шаблона
        avg_color = np.mean(hsv_template, axis=(0, 1))
        hue, saturation, value = avg_color
        
        # Создаем маску по цвету в текущем кадре
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Диапазон цветов вокруг доминирующего цвета шаблона
        lower_color = np.array([max(0, hue-20), 50, 50])
        upper_color = np.array([min(179, hue+20), 255, 255])
        
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Берем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Минимальный размер
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Проверяем валидность bbox
                if (w > 20 and h > 20 and x >= 0 and y >= 0 and 
                    x + w < frame.shape[1] and y + h < frame.shape[0]):
                    
                    bbox = (x, y, w, h)
                    confidence = min(cv2.contourArea(largest_contour) / 10000, 1.0)
                    return True, bbox, confidence
        
        return False, None, 0
    
    def auto_detect_object(self, frame):
        """Автоматическое обнаружение выбранным методом"""
        if self.template is None:
            return False, None, 0
            
        if self.detection_method == "Feature Matching":
            return self.detect_with_features(frame)
        elif self.detection_method == "Template Matching":
            return self.detect_with_template(frame)
        elif self.detection_method == "Color Detection":
            return self.detect_with_color(frame)
        else:
            return self.detect_with_features(frame)  # По умолчанию
    
    def on_mouse_down(self, event):
        if self.is_tracking and self.current_frame is not None:
            self.drawing = True
            self.ix, self.iy = event.x, event.y
            
    def on_mouse_drag(self, event):
        if self.drawing:
            if self.temp_rect:
                self.video_canvas.delete(self.temp_rect)
            self.temp_rect = self.video_canvas.create_rectangle(
                self.ix, self.iy, event.x, event.y, outline="blue", width=2
            )
    
    def on_mouse_up(self, event):
        if self.drawing and self.current_frame is not None:
            self.drawing = False
            x, y = event.x, event.y
            
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            
            if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    scale_x = self.current_frame.shape[1] / canvas_width
                    scale_y = self.current_frame.shape[0] / canvas_height
                    
                    cv_roi = (
                        int(x1 * scale_x), int(y1 * scale_y),
                        int((x2 - x1) * scale_x), int((y2 - y1) * scale_y)
                    )
                    
                    # Проверяем валидность ROI
                    if (cv_roi[2] > 10 and cv_roi[3] > 10 and 
                        cv_roi[0] >= 0 and cv_roi[1] >= 0 and 
                        cv_roi[0] + cv_roi[2] < self.current_frame.shape[1] and 
                        cv_roi[1] + cv_roi[3] < self.current_frame.shape[0]):
                        
                        self.tracker, tracker_name = self.create_tracker()
                        
                        if self.tracker is not None:
                            try:
                                success = self.tracker.init(self.current_frame, cv_roi)
                                if success:
                                    self.status_label.config(text=f"Трекинг: Ручной выбор начат ({tracker_name})")
                                    self.update_info(f"Ручной выбор начат\nТрекер: {tracker_name}")
                                else:
                                    self.status_label.config(text="Ошибка инициализации трекера")
                                    self.update_info("Ошибка: Не удалось инициализировать трекер")
                            except Exception as e:
                                self.status_label.config(text=f"Ошибка трекера: {str(e)}")
                                self.update_info(f"Ошибка инициализации: {str(e)}")
            
            if self.temp_rect:
                self.video_canvas.delete(self.temp_rect)
                self.temp_rect = None
    
    def update_frame(self):
        if not self.is_tracking or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return
        
        self.current_frame = frame.copy()
        display_frame = frame.copy()
        
        # Автоматическое обнаружение
        if self.tracker is None and self.template is not None:
            detected, bbox, confidence = self.auto_detect_object(frame)
            if detected:
                # Рисуем красный прямоугольник обнаружения
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Detected", (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                self.status_label.config(text=f"Трекинг: {self.detection_method} обнаружен (уверенность: {confidence:.2f})")
                self.update_info(f"Автообнаружение с помощью {self.detection_method}\nУверенность: {confidence:.2f}")

                # Создаем трекер
                self.tracker, tracker_name = self.create_tracker()
                
                if self.tracker is not None:
                    try:
                        success = self.tracker.init(frame, bbox)
                        if success:
                            self.update_info(f"Трекер {tracker_name} инициализирован")
                        else:
                            self.update_info("Ошибка инициализации трекера")
                            self.tracker = None
                    except Exception as e:
                        self.update_info(f"Ошибка инициализации: {str(e)}")
                        self.tracker = None
            else:
                info_text = f"{self.detection_method} не удался\nУверенность: {confidence:.2f}\nНарисуйте прямоугольник вручную"
                self.update_info(info_text)
        
        # Обновление трекера
        if self.tracker is not None:
            try:
                success, bbox = self.tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    # Проверяем валидность bbox
                    if (w > 10 and h > 10 and x >= 0 and y >= 0 and 
                        x + w < frame.shape[1] and y + h < frame.shape[0]):
                        
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(display_frame, "Tracking", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        self.status_label.config(text="Трекинг: Объект отслеживается")
                    else:
                        # Невалидный bbox - сбрасываем трекер
                        self.tracker = None
                        self.status_label.config(text="Трекинг: Невалидный bounding box")
                else:
                    self.status_label.config(text="Трекинг потерян - нарисуйте прямоугольник снова")
                    self.tracker = None
            except Exception as e:
                self.update_info(f"Ошибка трекинга: {str(e)}")
                self.tracker = None
        
        # Конвертируем для отображения
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        
        # Масштабируем для canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        else:
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            
        img_tk = ImageTk.PhotoImage(img)
        
        # Обновляем Canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.video_canvas.image = img_tk
        
        self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedObjectTracker(root)
    root.mainloop()