<div align="center">
<!-- <img src="assets/logo.png" alt="Logo" width="200"/> -->

<h1>🦞 ABot-Claw</h1>

<p align="center">
  <b>AMAP CV Lab</b>
</p>
</div>


**ABot-Claw** is a next-generation embodied AI framework built on OpenClaw that unifies VLN, VLA, and WAM models through the **VLAC (Vision-Language-Action-Critic)** loop, enabling real-time task monitoring and self-adaptation. By sharing a **multimodal memory**—centered on vision, semantics, and geometry—across devices and leveraging centralized control, it breaks down the silos of single-task robots, delivering a robust, collaborative, and elastically scalable **multi-agent system**.

---

## ✨ Core Features

### 🧠 1. VLAC: Task Progress Feedback Mechanism  
Introduces the **VLAC (Vision-Language-Action-Critic) closed-loop feedback mechanism** for dynamic task control.  
*   **Real-time Assessment:** Agents assess task completion and execution progress.  
*   **Adaptive Adjustment:** VLAC provides feedback and triggers strategy updates when deviations occur, enhancing success rates and robustness in long-horizon tasks.

### 🔗 2. End-to-End Closed-Loop Interaction  
Supports **VLA (Vision-Language-Action)** and **WAM (World Action Model)** for full-cycle autonomy.  
*   **Seamless Integration:** Tightly aligns perception with action to accurately follow natural language instructions.  
*   **High Autonomy:** Executes complex multi-step tasks end-to-end without human intervention.

### 👥 3. Multi-Robot Collaboration & Elastic Architecture  
Enables decentralized collaboration via a shared "Brain."  
*   **Unified Decision Making:** All robots share one Agent Runtime for synchronized state and joint reasoning.  
*   **Hot-Swappable Support:** Robots can join or replace others anytime without disrupting task flow.

### 📸 4. Vision-Centric Memory Mechanism  
Features a unified **Memory System** for persistent knowledge storage.  
*   **Integrated Structure:** Combines geometric maps (for localization) and semantic maps / image-feature-GPS (for understanding).  
*   **Long-Context Understanding:** Retains and retrieves key visual history to overcome occlusion and delayed feedback.

---

## 🏗️ System Architecture

ABot-Claw employs a layered microservices architecture to ensure high cohesion and low coupling:

1.  **Infrastructure Layer:**
    *   **GPU Server:** Hosts high-performance computing models including Yolo, Depth, VLA, Grasp Anything, VLN, and WAM.
    *   **Robots & Cameras:** Physical terminals including Dog, G1, PiPer, and various camera devices.

2.  **Runtime Core (Agent Runtime):**
    *   **Gateway:** Handles message routing for CLI/Web UI and Channels (Telegram, DingTalk, Feishu).
    *   **Agent Loop:** The core intelligence cycle containing Context, Tools, and Skills.
    *   **Scheduler & Device:** Manages task scheduling (Heartbeat/Cron) and local device interaction (File System, Shell, Browser).

3.  **Memory & Knowledge:**
    *   **Vision-centric Memory:** A central repository at the top layer storing geometric maps, semantic maps, and image feature indices for global access.

---


## 🚀 Quick Start

```
coming soon...
```

<!-- ### Prerequisites

### Installation -->


---

## 🙏 Acknowledgement
This project builds upon the following open-source projects. We thank these teams for their contributions:
- [Tidybot-Universe](https://github.com/TidyBot-Services/Tidybot-Universe.git)

---
