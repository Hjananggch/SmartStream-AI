# SmartStream-AI: 多模态视觉语义智能监控终端
**(SmartStream-AI: Multimodal Intelligent Vision Stream Monitor)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green) ![Model](https://img.shields.io/badge/Model-Qwen3--VL-purple) ![License](https://img.shields.io/badge/License-MIT-orange)

> **重塑视觉检测范式：从“像素感知”到“语义推理”。**
> SmartStream-AI 是一个基于 Qwen3-VL 大模型与时序拼图（Temporal Stitching）技术的通用视频分析平台。它允许用户通过自然语言定义监控任务，实时从视频流中提取复杂的行为语义。

---

## 📖 项目简介 (Introduction)

传统的视频监控系统通常受限于预定义的类别（如 YOLO 只能检测它学过的物体），面对复杂场景（如“车辆违停判断”、“工人违规行为”）时灵活性不足且误报率高。

**SmartStream-AI** 突破了这一限制。它构建了一个可视化的桌面控制台，将视频流实时切片，利用**多模态大语言模型（VLM）**的视觉理解能力，实现“即问即答”式的智能监控。

**核心解决痛点：**
1.  **逻辑判断难**：传统模型难以区分“车辆行驶”与“车辆违停”（视觉上都是车）。本项目通过**时序拼接**解决此问题。
2.  **泛化能力弱**：新需求通常需要重新训练模型。本项目仅需修改 **Prompt（提示词）** 即可上线新功能。

---

## 📸 核心功能 (Features)

### 1. 智能时序拼接模式 (Temporal Stitching Mode) 🌟
**SmartStream-AI 的核心创新点**。
针对需要“时间维度”判断的场景（如违停、物品遗留），系统自动维护一个帧缓冲区，以 1Hz 频率采集过去 4 秒的画面（$T_{-3}, T_{-2}, T_{-1}, T_{0}$），将其拼接为一张 2x2 的田字格图像发送给 VLM。
* **原理**：模型通过对比四个分格中物体的位置变化，精准推理出运动状态。
* **应用**：车辆违停报警、人员长时间滞留检测。

### 2. 实时流语义分析 (Real-time Stream Analysis)
对视频流进行周期性采样与推理，支持多种违规行为的并行检测。
* **应用**：抽烟检测、玩手机检测、未戴安全帽检测、火焰烟雾检测。
* **输出**：支持结构化日志输出（如：`🚨 停车报警` 或 `✅ 正常`）。

### 3. 静态影像诊断 (Static Image Diagnosis)
支持一键暂停视频流，上传本地高清图片进行深度分析。系统会自动识别单图模式，绕过时序逻辑，直接进行像素级分析。

---

## 🛠️ 技术栈 (Tech Stack)

* **核心语言**: `Python 3.9`
* **图形界面**: `PyQt5` (现代化 Dark Mode 设计，支持自适应布局)
* **视觉处理**: `OpenCV` (帧采集、缓冲区队列管理、图像拼接算法)
* **AI 推理**: `OpenAI SDK` (接入 Qwen3-VL-4B / 7B 等 VLM 模型 API)
* **并发架构**: `QThread` (GUI 渲染、视频流采集、LLM 推理三线异步处理，确保界面流畅)

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
确保本地环境已安装 Python 3.9 或以上版本。

```bash
# 1. 克隆项目到本地
git clone https://github.com/Hjananggch/SmartStream-AI.git
cd SmartStream-AI

# 2. 安装项目依赖
pip install -r requirements.txt
