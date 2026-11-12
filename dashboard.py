import sys
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QTextEdit, QFileDialog, QMessageBox, QTabWidget
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QIcon

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from xai_engine import CyberScopeXAI

DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# OPTIMIZED MATRIX DIMENSIONS (Must match train.py)
IMAGE_DIM = (6, 6)
TARGET_FEATURES = IMAGE_DIM[0] * IMAGE_DIM[1]

class MetricCard(QFrame):
    def __init__(self, title, initial_value, accent_color="#00E676"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1E293B;
                border-left: 4px solid {accent_color};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.title_label.setStyleSheet("color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;")

        self.value_label = QLabel(initial_value)
        self.value_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.value_label.setStyleSheet(f"color: {accent_color}; margin-top: 5px;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, val):
        self.value_label.setText(str(val))


class CYBERSCOPE_Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CYBERSCOPE AI — Advanced SOC Console")
        self.setGeometry(50, 50, 1400, 850)
        self.setStyleSheet("background-color: #0F172A; color: #F8FAFC; font-family: 'Segoe UI', sans-serif;")

        self.total_scanned = 0
        self.threats_detected = 0
        self.exported_records = []

        self.load_artifacts()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_flow)

    def load_artifacts(self):
        print("Loading AI Model and Preprocessors...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.df = pd.read_csv(DATA_PATH)
        drop_cols = [c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP"] if c in self.df.columns]
        self.features_df = self.df.drop(columns=drop_cols)
        self.feature_names = self.features_df.columns.tolist()

        self.scaled_features = self.scaler.transform(self.features_df.fillna(0)).astype(np.float32)

        if self.scaled_features.shape[1] < TARGET_FEATURES:
            pad = TARGET_FEATURES - self.scaled_features.shape[1]
            self.padded_features = np.pad(self.scaled_features, ((0, 0), (0, pad)), mode="constant")
        else:
            self.padded_features = self.scaled_features[:, :TARGET_FEATURES]

        self.images = self.padded_features.reshape(-1, IMAGE_DIM[0], IMAGE_DIM[1], 1)
        self.xai = CyberScopeXAI(MODEL_PATH, SCALER_PATH, ENCODER_PATH)

        self.indices = np.random.permutation(np.arange(10, len(self.scaled_features)))
        self.sample_pointer = 0

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # ---------------- TOP NAVIGATION BAR ----------------
        nav_layout = QHBoxLayout()
        
        brand_title = QLabel("CYBERSCOPE AI")
        brand_title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        brand_title.setStyleSheet("color: #38BDF8; letter-spacing: 2px;")
        
        nav_layout.addWidget(brand_title)
        nav_layout.addStretch()

        btn_style = """
            QPushButton {
                font-weight: bold; padding: 10px 20px; border-radius: 6px; border: none; font-size: 13px;
            }
        """
        
        self.btn_start = QPushButton("▶ START STREAM")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet(btn_style + "QPushButton { background-color: #10B981; color: #0F172A; } QPushButton:hover { background-color: #34D399; }")
        self.btn_start.clicked.connect(self.start_monitoring)

        self.btn_stop = QPushButton("⏸ PAUSE")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setStyleSheet(btn_style + "QPushButton { background-color: #334155; color: #F8FAFC; } QPushButton:hover { background-color: #475569; }")
        self.btn_stop.clicked.connect(self.stop_monitoring)

        nav_layout.addWidget(self.btn_start)
        nav_layout.addWidget(self.btn_stop)
        main_layout.addLayout(nav_layout)

        # ---------------- METRICS ROW ----------------
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)

        self.card_scanned = MetricCard("Flows Evaluated", "0", "#38BDF8")
        self.card_threats = MetricCard("Threats Intercepted", "0", "#F43F5E")
        self.card_threat_rate = MetricCard("Threat Ratio", "0.0%", "#FBBF24")
        self.card_model_status = MetricCard("AI Engine Status", "STANDBY", "#94A3B8")

        metrics_layout.addWidget(self.card_scanned)
        metrics_layout.addWidget(self.card_threats)
        metrics_layout.addWidget(self.card_threat_rate)
        metrics_layout.addWidget(self.card_model_status)
        main_layout.addLayout(metrics_layout)

        # ---------------- TRAY / TAB WIDGET ----------------
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #334155; background: #1E293B; border-radius: 8px; }
            QTabBar::tab { background: #0F172A; color: #94A3B8; padding: 12px 25px; margin-right: 2px; font-weight: bold; border-top-left-radius: 8px; border-top-right-radius: 8px; }
            QTabBar::tab:selected { background: #1E293B; color: #38BDF8; border-bottom: 3px solid #38BDF8; }
            QTabBar::tab:hover:!selected { background: #334155; }
        """)

        # TAB 1: Live Telemetry
        self.tab_telemetry = QWidget()
        self.setup_telemetry_tab()
        self.tabs.addTab(self.tab_telemetry, "📡 Live Telemetry")

        # TAB 2: Explainable AI (XAI)
        self.tab_xai = QWidget()
        self.setup_xai_tab()
        self.tabs.addTab(self.tab_xai, "🧠 Threat Diagnostics (XAI)")

        main_layout.addWidget(self.tabs)

    def setup_telemetry_tab(self):
        layout = QVBoxLayout(self.tab_telemetry)
        layout.setContentsMargins(15, 15, 15, 15)

        # Export Button inside Telemetry Tab
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.btn_export = QPushButton("💾 EXPORT CSV")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setStyleSheet("background-color: #38BDF8; color: #0F172A; font-weight: bold; padding: 8px 15px; border-radius: 4px;")
        self.btn_export.clicked.connect(self.export_results)
        export_layout.addWidget(self.btn_export)
        layout.addLayout(export_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["FLOW ID", "CLASSIFICATION", "CONFIDENCE", "SEVERITY"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #0F172A; border: none; font-size: 13px; }
            QHeaderView::section { background-color: #334155; color: #94A3B8; font-weight: bold; padding: 10px; border: none; }
            QTableWidget::item { padding: 10px; border-bottom: 1px solid #1E293B; }
        """)
        layout.addWidget(self.table)

    def setup_xai_tab(self):
        layout = QHBoxLayout(self.tab_xai)
        layout.setContentsMargins(15, 15, 15, 15)

        # Left: Visual Plots
        plots_layout = QVBoxLayout()
        
        self.fig_img = Figure(figsize=(3, 3), facecolor="#1E293B")
        self.canvas_img = FigureCanvas(self.fig_img)
        plots_layout.addWidget(self.canvas_img)

        self.fig_xai = Figure(figsize=(5, 3), facecolor="#1E293B")
        self.canvas_xai = FigureCanvas(self.fig_xai)
        plots_layout.addWidget(self.canvas_xai)

        layout.addLayout(plots_layout, 60)

        # Right: Text Diagnostic Console
        console_layout = QVBoxLayout()
        console_title = QLabel("REAL-TIME DIAGNOSTIC REPORT")
        console_title.setStyleSheet("color: #94A3B8; font-weight: bold; letter-spacing: 1px;")
        console_layout.addWidget(console_title)

        self.exp_text = QTextEdit()
        self.exp_text.setReadOnly(True)
        self.exp_text.setStyleSheet("""
            QTextEdit {
                background-color: #0F172A; color: #10B981; border: 1px solid #334155; border-radius: 6px;
                font-family: 'Consolas', monospace; font-size: 13px; padding: 12px;
            }
        """)
        self.exp_text.setText("Awaiting flow sequence...")
        console_layout.addWidget(self.exp_text)

        layout.addLayout(console_layout, 40)

    def start_monitoring(self):
        self.card_model_status.set_value("ACTIVE")
        self.card_model_status.setStyleSheet("QFrame { background-color: #1E293B; border-left: 4px solid #10B981; border-radius: 8px; padding: 15px; }")
        self.timer.start(1000)

    def stop_monitoring(self):
        self.card_model_status.set_value("PAUSED")
        self.card_model_status.setStyleSheet("QFrame { background-color: #1E293B; border-left: 4px solid #FBBF24; border-radius: 8px; padding: 15px; }")
        self.timer.stop()

    def process_next_flow(self):
        if self.sample_pointer >= len(self.indices):
            self.sample_pointer = 0

        idx = self.indices[self.sample_pointer]
        self.sample_pointer += 1

        seq = self.scaled_features[idx - 10 : idx].reshape(1, 10, -1)
        img = self.images[idx].reshape(1, IMAGE_DIM[0], IMAGE_DIM[1], 1)

        preds = self.model.predict([img, seq], verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = float(preds[class_idx])
        label_str = str(self.label_encoder.inverse_transform([class_idx])[0])

        # ==============================================================
        # FALSE POSITIVE KILL SWITCH (Open-Set Recognition Thresholding)
        # ==============================================================
        if confidence < 0.85 and label_str.lower() not in ["benign", "normal"]:
            label_str = "Anomaly (Unverified)"
            is_threat = True 
            severity = "SUSPICIOUS"
            sev_color = "#FBBF24" 
        else:
            is_threat = label_str.lower() not in ["benign", "normal"]
            if not is_threat:
                severity = "CLEAN"
                sev_color = "#10B981"
            elif any(k in label_str.lower() for k in ["portscan", "patator"]):
                severity = "MEDIUM"
                sev_color = "#F97316"
            elif any(k in label_str.lower() for k in ["webattack", "botnet"]):
                severity = "HIGH"
                sev_color = "#EF4444"
            else:
                severity = "CRITICAL"
                sev_color = "#E11D48"

        self.total_scanned += 1
        if is_threat:
            self.threats_detected += 1

        self.card_scanned.set_value(f"{self.total_scanned:,}")
        self.card_threats.set_value(f"{self.threats_detected:,}")
        self.card_threat_rate.set_value(f"{(self.threats_detected / self.total_scanned) * 100:.1f}%")

        flow_id = f"FLOW-{idx:05d}"
        conf_str = f"{confidence * 100:.2f}%"

        self.exported_records.append({"Flow_ID": flow_id, "Classification": label_str, "Confidence": conf_str, "Severity": severity})

        row = self.table.rowCount()
        self.table.insertRow(row)

        items = [
            QTableWidgetItem(flow_id), QTableWidgetItem(label_str),
            QTableWidgetItem(conf_str), QTableWidgetItem(severity)
        ]
        
        items[0].setForeground(QColor("#94A3B8"))
        items[1].setForeground(QColor(sev_color))
        items[1].setFont(QFont("Segoe UI", 10, QFont.Bold))
        items[2].setForeground(QColor("#F8FAFC"))
        items[3].setForeground(QColor(sev_color))
        items[3].setFont(QFont("Segoe UI", 10, QFont.Bold))

        for i, item in enumerate(items):
            self.table.setItem(row, i, item)
        self.table.scrollToBottom()

        shap_results = self.xai.compute_shap_attributions(img[0], seq[0], self.feature_names, class_idx)
        top_f1, v1 = shap_results[-1]
        top_f2, v2 = shap_results[-2]

        explanation_lines = [
            f"[{flow_id} — DIAGNOSTIC RUN]",
            f"Primary Class  : {label_str}",
            f"Net Confidence : {conf_str}",
            f"Assigned Risk  : {severity}\n",
            "SHAP Feature Vectors:"
        ]

        if not is_threat:
            explanation_lines.extend([" • Behavior matches known clean baseline.", f" • No anomalous spikes in '{top_f1}'."])
        else:
            explanation_lines.extend([
                f" • High variance triggered by '{top_f1}' (+{v1:.4f})",
                f" • Secondary vector: '{top_f2}' (+{v2:.4f})",
                f" • ACTION: Flag source IP for potential {label_str}."
            ])

        self.exp_text.setText("\n".join(explanation_lines))
        self.update_plots(img[0, :, :, 0], shap_results, is_threat, sev_color)

    def update_plots(self, img_data, shap_results, is_threat, accent_color):
        self.fig_img.clear()
        ax1 = self.fig_img.add_subplot(111)
        ax1.imshow(img_data, cmap="inferno" if is_threat else "viridis")
        ax1.axis("off")
        ax1.set_title("CNN Spatial Fingerprint (6x6)", color="#F8FAFC", fontsize=10, pad=10)
        self.fig_img.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
        self.canvas_img.draw()

        self.fig_xai.clear()
        ax2 = self.fig_xai.add_subplot(111)
        ax2.set_facecolor("#1E293B")

        fnames = [item[0][:12] for item in shap_results]
        fvals = [item[1] for item in shap_results]

        ax2.barh(fnames, fvals, color=accent_color, height=0.5)
        ax2.axvline(0, color='#334155', linewidth=1)
        ax2.tick_params(colors="#94A3B8", labelsize=9)
        ax2.set_title(r"Temporal Feature Drivers", color="#F8FAFC", fontsize=10, pad=10)
        
        for spine in ['top', 'right', 'bottom', 'left']:
            ax2.spines[spine].set_visible(False)

        self.fig_xai.subplots_adjust(left=0.35, right=0.95, top=0.85, bottom=0.15)
        self.canvas_xai.draw()

    def export_results(self):
        if not self.exported_records:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save SOC Logs", "cyberscope_soc_logs.csv", "CSV Files (*.csv)")
        if file_path:
            pd.DataFrame(self.exported_records).to_csv(file_path, index=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CYBERSCOPE_Dashboard()
    window.show()
    sys.exit(app.exec_())