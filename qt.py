import sys
import cv2
import time
import base64
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QSpinBox, QLineEdit, QGroupBox, QFileDialog) # 新增 QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from openai import OpenAI

# ================= 配置区域 =================
API_URL = "xxx" 
MODEL_NAME = "qwen3-vl-4b" 
# ===========================================

class LLMWorker(QThread):
    result_signal = pyqtSignal(str, float)

    def __init__(self, frame, prompt):
        super().__init__()
        self.frame = frame
        self.prompt = prompt
        self.client = OpenAI(base_url=API_URL, api_key="EMPTY")

    def encode_image(self, cv_image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', cv_image, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

    def run(self):
        try:
            start_time = time.time()
            base64_img = self.encode_image(self.frame)
            
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                            }
                        ],
                    }
                ],
                max_tokens=128,
                temperature=0.01
            )
            
            content = response.choices[0].message.content
            cost_time = time.time() - start_time
            self.result_signal.emit(content, cost_time)
            
        except Exception as e:
            self.result_signal.emit(f"API Error: {str(e)}", 0)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    trigger_detection_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str) # 新增错误信号

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.interval = 5
        self.last_check_time = 0
        self.source = 0 # 默认源

    def set_source(self, source):
        """设置视频源，自动判断是int还是str"""
        if isinstance(source, str) and source.isdigit():
            self.source = int(source) # 摄像头索引
        else:
            self.source = source # 文件路径

    def update_interval(self, val):
        self.interval = val

    def run(self):
        # 使用动态设置的 source 初始化
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            self.error_signal.emit(f"无法打开视频源: {self.source}")
            self._run_flag = False
            return

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                current_time = time.time()
                if current_time - self.last_check_time >= self.interval:
                    self.last_check_time = current_time
                    self.trigger_detection_signal.emit(cv_img)
            else:
                # 视频播放结束或摄像头断开
                self.error_signal.emit("视频播放结束" if isinstance(self.source, str) else "摄像头连接断开")
                break
            
            # 简单的帧率控制，避免MP4播放过快
            time.sleep(0.03)
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多模态大模型智能监控台 - Qwen3-VL")
        self.setFixedSize(960, 760) # 稍微增加高度以容纳新控件
        
        # 样式表
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox { color: #ddd; font-weight: bold; border: 1px solid #555; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QLabel { color: #eee; font-size: 13px; }
            QTextEdit { background-color: #1e1e1e; color: #00ff00; border: 1px solid #444; font-family: Consolas; font-size: 13px; }
            QPushButton { background-color: #0078d7; color: white; border-radius: 4px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #198ce6; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QLineEdit, QSpinBox { padding: 5px; border-radius: 3px; border: 1px solid #555; background: #333; color: white; }
        """)
        
        self.init_ui()
        
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.trigger_detection_signal.connect(self.start_llm_detection)
        self.thread.error_signal.connect(self.on_video_error) # 连接错误信号
        self.llm_worker = None

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # === 上部分：视频区 ===
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("background-color: #111; border: 2px solid #444; border-radius: 5px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("系统就绪\n请选择源并点击 [启动监控]")
        main_layout.addWidget(self.image_label, stretch=60)

        # === 下部分：控制与日志 ===
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        bottom_widget.setLayout(bottom_layout)

        # 控制面板
        settings_group = QGroupBox("🛠️ 参数设置")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(10)
        
        # 1. 视频源选择区域 (新增)
        settings_layout.addWidget(QLabel("视频源 (0为摄像头):"))
        h_source = QHBoxLayout()
        self.txt_source = QLineEdit("0") # 默认摄像头
        self.btn_file = QPushButton("📂")
        self.btn_file.setFixedWidth(40)
        self.btn_file.clicked.connect(self.select_file)
        h_source.addWidget(self.txt_source)
        h_source.addWidget(self.btn_file)
        settings_layout.addLayout(h_source)

        # 2. 间隔设置
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("检测间隔(s):"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 600)
        self.spin_interval.setValue(5)
        self.spin_interval.valueChanged.connect(self.on_interval_change)
        h1.addWidget(self.spin_interval)
        settings_layout.addLayout(h1)
        
        # 3. Prompt
        settings_layout.addWidget(QLabel("Prompt (提示词):"))
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setText("请描述画面中人物的动作。")
        settings_layout.addWidget(self.txt_prompt)
        
        # 4. 按钮
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 启动")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_video)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setStyleSheet("background-color: #d9534f; border: none;") 
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        settings_layout.addLayout(btn_layout)
        
        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)
        settings_group.setFixedWidth(320) # 稍微加宽

        # 日志面板
        log_group = QGroupBox("📝 实时日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        bottom_layout.addWidget(settings_group)
        bottom_layout.addWidget(log_group)

        main_layout.addWidget(bottom_widget, stretch=40)

    def select_file(self):
        """打开文件选择器"""
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '.', "Video files (*.mp4 *.avi *.mkv)")
        if fname:
            self.txt_source.setText(fname)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def start_video(self):
        # 获取用户输入的源
        source_input = self.txt_source.text().strip()
        self.thread.set_source(source_input) # 设置源
        self.thread.interval = self.spin_interval.value()
        self.thread._run_flag = True
        self.thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.txt_source.setEnabled(False) # 运行时锁定输入框
        self.btn_file.setEnabled(False)
        self.append_log(f">>> 系统启动... 源: {source_input}")

    def stop_video(self):
        self.thread.stop()
        self.reset_ui_state()
        self.image_label.clear()
        self.image_label.setText("停止")
        self.append_log(">>> 系统手动停止。")

    def on_video_error(self, msg):
        """处理视频线程报错（如文件读完）"""
        self.thread.stop()
        self.reset_ui_state()
        self.append_log(f"<span style='color:orange;'>[提示] {msg}</span>")
        self.image_label.setText(msg)

    def reset_ui_state(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.txt_source.setEnabled(True)
        self.btn_file.setEnabled(True)

    def on_interval_change(self):
        self.thread.update_interval(self.spin_interval.value())
        self.append_log(f"--- 间隔更新: {self.spin_interval.value()}s ---")

    def start_llm_detection(self, frame):
        prompt = self.txt_prompt.text()
        self.append_log(f"🔍 分析中... ({prompt})")
        self.llm_worker = LLMWorker(frame, prompt)
        self.llm_worker.result_signal.connect(self.handle_llm_result)
        self.llm_worker.start()

    def handle_llm_result(self, result_text, cost_time):
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        if cost_time > 0:
            self.append_log(
                f"<div style='border-bottom:1px solid #444; padding-bottom:5px; margin-bottom:5px;'>"
                f"<span style='color:#888;'>[{timestamp}]</span> "
                f"<span style='color:#00aaff;'>耗时 {cost_time:.2f}s</span><br>"
                f"<span style='color:#00ff00; font-weight:bold;'>{result_text}</span>"
                f"</div>"
            )
        else:
            self.append_log(f"<span style='color:#ff4444;'>[{timestamp}] 错误: {result_text}</span>")
        self.llm_worker = None

    def append_log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())