import sys
import cv2
import time
import base64
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QSpinBox, QLineEdit, QFrame, QFileDialog, QSizePolicy, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from openai import OpenAI

# ================= 配置区域 =================
API_URL = "http://xxx/v1" 
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
        # 压缩图片以加快传输 (质量 70)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
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
                max_tokens=64, # 限制回复长度，提高速度
                temperature=0.01
            )
            
            content = response.choices[0].message.content
            cost_time = time.time() - start_time
            self.result_signal.emit(content, cost_time)
            
        except Exception as e:
            self.result_signal.emit(f"API Error: {str(e)}", 0)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)   # UI显示
    trigger_detection_signal = pyqtSignal(np.ndarray) # AI检测
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.interval = 5
        self.source = 0
        
        # === 拼接模式相关变量 ===
        self.stitching_mode = False 
        self.frame_buffer = deque(maxlen=4)
        self.sample_rate = 1.0 
        
        self.last_check_time = 0
        self.last_sample_time = 0

    def set_config(self, source, interval, stitching_mode):
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source
        self.interval = interval
        
        if self.stitching_mode != stitching_mode:
            self.frame_buffer.clear()
        self.stitching_mode = stitching_mode

    def stitch_images_2x2(self, frames):
        if len(frames) < 4: return frames[-1]
        
        h, w = frames[0].shape[:2]
        resized = [cv2.resize(f, (w, h)) for f in frames]
        
        top = np.hstack((resized[0], resized[1]))
        bottom = np.hstack((resized[2], resized[3]))
        grid = np.vstack((top, bottom))
        
        h_g, w_g = grid.shape[:2]
        return cv2.resize(grid, (int(w_g*0.8), int(h_g*0.8)))

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_signal.emit(f"无法打开视频源: {self.source}")
            self._run_flag = False
            return

        self.last_check_time = time.time()
        self.last_sample_time = time.time()
        self.frame_buffer.clear()

        while self._run_flag:
            ret, cv_img = cap.read()
            if not ret:
                self.error_signal.emit("视频流结束")
                break
            
            self.change_pixmap_signal.emit(cv_img)
            
            curr_time = time.time()

            if self.stitching_mode:
                if curr_time - self.last_sample_time >= self.sample_rate:
                    self.frame_buffer.append(cv_img.copy())
                    self.last_sample_time = curr_time
                
                if curr_time - self.last_check_time >= self.interval:
                    self.last_check_time = curr_time
                    if len(self.frame_buffer) == 4:
                        stitched_img = self.stitch_images_2x2(list(self.frame_buffer))
                        self.trigger_detection_signal.emit(stitched_img)
            else:
                if curr_time - self.last_check_time >= self.interval:
                    self.last_check_time = curr_time
                    self.trigger_detection_signal.emit(cv_img.copy())

            time.sleep(0.02) 
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartStream-AI | 多模态视觉语义智能监控终端")
        
        screen = QApplication.primaryScreen().geometry()
        self.win_w = int(screen.width() * 0.8)
        self.win_h = int(screen.height() * 0.8)
        self.setFixedSize(self.win_w, self.win_h)
        self.move((screen.width() - self.win_w) // 2, (screen.height() - self.win_h) // 2)
        
        self.apply_stylesheet()
        
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.trigger_detection_signal.connect(self.start_llm_detection)
        self.thread.error_signal.connect(self.on_video_error)
        self.llm_worker = None

        self.init_ui()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
            QFrame#ControlPanel, QFrame#LogPanel { background-color: #313244; border-radius: 12px; border: 1px solid #45475a; }
            QLabel#PanelTitle { color: #89b4fa; font-weight: bold; font-size: 16px; padding-bottom: 5px; border-bottom: 2px solid #45475a; }
            QLineEdit, QSpinBox { background-color: #181825; border: 1px solid #585b70; border-radius: 6px; padding: 8px; color: #cdd6f4; }
            QLineEdit:focus, QSpinBox:focus { border: 1px solid #89b4fa; background-color: #1e1e2e; }
            QPushButton { border-radius: 6px; padding: 10px; font-weight: bold; border: none; }
            QPushButton#BtnFile { background-color: #45475a; color: #cdd6f4; }
            QPushButton#BtnFile:hover { background-color: #585b70; }
            QPushButton#BtnStart { background-color: #a6e3a1; color: #1e1e2e; }
            QPushButton#BtnStart:hover { background-color: #94e2d5; }
            QPushButton#BtnStart:disabled { background-color: #45475a; color: #6c7086; }
            QPushButton#BtnStop { background-color: #f38ba8; color: #1e1e2e; }
            QPushButton#BtnStop:hover { background-color: #eba0ac; }
            QPushButton#BtnStop:disabled { background-color: #45475a; color: #6c7086; }
            
            /* 新增：图片检测按钮样式 */
            QPushButton#BtnImgDetect { background-color: #f9e2af; color: #1e1e2e; margin-top: 10px; }
            QPushButton#BtnImgDetect:hover { background-color: #fcebb6; }

            QTextEdit { background-color: #11111b; border: 1px solid #45475a; border-radius: 8px; padding: 10px; font-family: 'Consolas', monospace; font-size: 13px; }
            
            QCheckBox { spacing: 8px; font-weight: bold; color: #f9e2af; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #585b70; background: #181825; }
            QCheckBox::indicator:checked { background: #f9e2af; border: 1px solid #f9e2af; image: url(check_icon.png); } 
            QCheckBox::indicator:checked { background-color: #f9e2af; border-color: #f9e2af; }
        """)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === 左侧视频 ===
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel()
        self.image_label.setStyleSheet("background-color: #11111b; border: 2px solid #45475a; border-radius: 12px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("<html><head/><body><p align='center'><span style='font-size:18pt; color:#6c7086;'>NO SIGNAL</span></p></body></html>")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        video_layout.addWidget(self.image_label)
        main_layout.addWidget(video_container, 75)

        # === 右侧控制 ===
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(20)

        # 1. 控制面板
        control_panel = QFrame()
        control_panel.setObjectName("ControlPanel")
        cp_layout = QVBoxLayout(control_panel)
        cp_layout.setContentsMargins(15, 15, 15, 15)
        cp_layout.setSpacing(15)

        cp_layout.addWidget(QLabel("⚙️ 控制面板"))

        # 视频源
        cp_layout.addWidget(QLabel("视频源 (Source):"))
        h_source = QHBoxLayout()
        self.txt_source = QLineEdit("0")
        self.btn_file = QPushButton("📂")
        self.btn_file.setObjectName("BtnFile")
        self.btn_file.setFixedWidth(40)
        self.btn_file.clicked.connect(self.select_video_file)
        h_source.addWidget(self.txt_source)
        h_source.addWidget(self.btn_file)
        cp_layout.addLayout(h_source)

        # 检测间隔
        cp_layout.addWidget(QLabel("检测间隔 (s):"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(2, 3600)
        self.spin_interval.setValue(5)
        self.spin_interval.setSuffix(" s")
        cp_layout.addWidget(self.spin_interval)

        # 拼接模式
        self.chk_stitch = QCheckBox("启用 4帧时序拼接 (T1-T4)")
        self.chk_stitch.stateChanged.connect(self.on_mode_change)
        cp_layout.addWidget(self.chk_stitch)

        # Prompt
        cp_layout.addWidget(QLabel("提示词 (Prompt):"))
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setText("画面中是否有人或车辆？")
        cp_layout.addWidget(self.txt_prompt)

        # 视频启停按钮
        h_btns = QHBoxLayout()
        self.btn_start = QPushButton("▶ 启动监控")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.clicked.connect(self.start_video)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setObjectName("BtnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_video)
        h_btns.addWidget(self.btn_start, 2)
        h_btns.addWidget(self.btn_stop, 1)
        cp_layout.addLayout(h_btns)

        # --- 新增分割线 ---
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #45475a;")
        cp_layout.addWidget(line)

        # --- 新增：图片检测按钮 ---
        self.btn_img_detect = QPushButton("🖼️ 上传图片并检测")
        self.btn_img_detect.setObjectName("BtnImgDetect")
        self.btn_img_detect.clicked.connect(self.upload_and_detect_image)
        cp_layout.addWidget(self.btn_img_detect)

        # 2. 日志面板
        log_panel = QFrame()
        log_panel.setObjectName("LogPanel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(15, 15, 15, 15)
        log_layout.addWidget(QLabel("📝 实时日志"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        sidebar_layout.addWidget(control_panel)
        sidebar_layout.addWidget(log_panel, 1)

        main_layout.addLayout(sidebar_layout, 25)

    def select_video_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频', '.', "Video (*.mp4 *.avi)")
        if fname: self.txt_source.setText(fname)

    # --- 新增功能：上传图片检测 ---
    def upload_and_detect_image(self):
        # 1. 停止当前正在运行的视频流（防止冲突）
        if self.thread.isRunning():
            self.stop_video()
            self.append_log("<span style='color:#f9e2af;'>[System] 为进行图片检测，已暂停视频监控。</span>")

        # 2. 选择图片文件
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', "Images (*.jpg *.png *.jpeg *.bmp *.webp)")
        if not fname:
            return

        # 3. 读取并显示
        img = cv2.imread(fname)
        if img is None:
            self.append_log(f"<span style='color:#f38ba8;'>[Error] 无法读取图片: {fname}</span>")
            return
        
        self.update_image(img)

        # 4. 智能调整 Prompt (如果是单图，不需要时序Prompt)
        current_prompt = self.txt_prompt.text()
        if self.chk_stitch.isChecked() and "T1-T4" in current_prompt:
            self.append_log("<span style='color:#89b4fa;'>[Info] 检测到单张图片，忽略拼接设置。</span>")
            # 可以临时自动改一下 Prompt，或者直接用用户的
            # self.txt_prompt.setText("画面中有什么？") 
        
        # 5. 直接触发 LLM 分析
        self.append_log(f"<span style='color:#f9e2af;'>[Image] 已加载图片，正在请求分析...</span>")
        self.start_llm_detection(img)

    def on_mode_change(self):
        if self.chk_stitch.isChecked():
            self.txt_prompt.setText("图为T1-T4四个时刻。若车辆在4个画面完全静止，回答“停车报警”，否则回答“没有检测到事件”。")
            self.append_log("<span style='color:#f9e2af;'>[Mode] 切换模式: 4帧时序拼接</span>")
        else:
            self.txt_prompt.setText("画面中是否有人或车辆？")
            self.append_log("<span style='color:#89b4fa;'>[Mode] 切换模式: 单帧实时检测</span>")

    def update_image(self, cv_img):
        target_w, target_h = self.image_label.width(), self.image_label.height()
        if target_w <= 0: return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_video(self):
        source = self.txt_source.text().strip()
        interval = self.spin_interval.value()
        is_stitch = self.chk_stitch.isChecked()
        
        self.thread.set_config(source, interval, is_stitch)
        self.thread._run_flag = True
        self.thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.txt_source.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.chk_stitch.setEnabled(False) 
        self.btn_img_detect.setEnabled(True) # 视频运行时也允许点这个（会自动暂停）
        
        mode_str = "拼接模式" if is_stitch else "单帧模式"
        self.append_log(f"<span style='color:#a6e3a1;'>▶ 监控启动 ({mode_str})</span>")

    def stop_video(self):
        self.thread.stop()
        self.reset_ui_state()
        self.image_label.setText("<html><head/><body><p align='center'><span style='font-size:18pt; color:#f38ba8;'>STOPPED</span></p></body></html>")
        self.append_log("<span style='color:#f38ba8;'>⏹ 停止监控</span>")

    def on_video_error(self, msg):
        self.thread.stop()
        self.reset_ui_state()
        self.append_log(f"<span style='color:#f38ba8;'>[Error] {msg}</span>")

    def reset_ui_state(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.txt_source.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.chk_stitch.setEnabled(True)

    def start_llm_detection(self, frame):
        prompt = self.txt_prompt.text()
        self.append_log(f"<span style='color:#89b4fa;'>[Request] 正在发送 API...</span>")
        self.llm_worker = LLMWorker(frame, prompt)
        self.llm_worker.result_signal.connect(self.handle_llm_result)
        self.llm_worker.start()

    def handle_llm_result(self, res, cost):
        t = QDateTime.currentDateTime().toString("HH:mm:ss")
        
        # --- 控制台打印时长 ---
        print(f"[{t}] 分析完成 | 耗时: {cost:.2f}s | 结果: {res}")
        
        if cost > 0:
            if "停车报警" in res:
                border_color = "#ff5555" 
                display_text = f"🚨 <b style='color:#ff5555; font-size:14px;'>{res}</b>"
            elif "没有" in res or "不是" in res:
                border_color = "#a6e3a1"
                display_text = f"✅ <span style='color:#a6e3a1;'>{res}</span>"
            else:
                border_color = "#89b4fa"
                display_text = f"<span style='color:#cdd6f4;'>{res}</span>"

            self.append_log(
                f"<div style='border-left: 4px solid {border_color}; background-color: #1e1e2e; padding: 8px; margin: 5px 0; border-radius: 4px;'>"
                f"<div style='border-bottom: 1px dashed #45475a; padding-bottom: 4px; margin-bottom: 4px;'>"
                f"<span style='color:#bac2de; font-family:Consolas;'>[{t}]</span> "
                f"<span style='color:#f9e2af; font-weight:bold;'>⏱️ 耗时: {cost:.2f}s</span>"
                f"</div>"
                f"{display_text}"
                f"</div>"
            )
        else:
            self.append_log(f"<span style='color:#f38ba8;'>[API Error] {res}</span>")
        
        self.llm_worker = None

    def append_log(self, html):
        self.log_text.append(html)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    window = ModernWindow()
    window.show()
    sys.exit(app.exec_())