"""
CyberScope AI — Next-Generation SOC Command Console (Fully Responsive)
======================================================================
Interactive, dynamically resizing cyber-SOC graphical console.

Key Architectural & Responsive Design Features:
- Fully dynamic layout: adapts seamlessly from compact windows (780x520) up to 4K displays
- Matplotlib canvases configure QSizePolicy.Expanding with tight-layout redraws on resizeEvent
- Interactive QSplitter panes allow smooth user-controlled pane sizing
- Responsive header bar with graceful wrapping and compact mode for smaller viewports
- Interactive real-time packet stream with live search, severity filtering, and row-click XAI inspection
- True Integrated Gradients XAI feature attributions and 1D tabular feature signatures
"""

import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QTextEdit, QFileDialog, QMessageBox, QTabWidget,
    QSlider, QComboBox, QLineEdit, QSplitter, QProgressBar, QStatusBar,
    QGridLayout, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt, QDateTime
from PyQt5.QtGui import QFont, QColor, QIcon
import matplotlib
import matplotlib.pyplot as plt
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__)))
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from xai_engine import CyberScopeXAI
from models.positional_encoding import PositionalEncoding, SinusoidalPositionalEncoding, LearnablePositionalEncoding

DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
CONFUSION_IMG = os.path.join("models", "final_test_confusion_matrix.png")
ROC_IMG = os.path.join("models", "final_test_roc_curves.png")

IMAGE_DIM = (6, 6)
TARGET_FEATURES = 36


class ModernMetricCard(QFrame):
    """Sleek neon-bordered metric card that dynamically scales font and padding."""
    def __init__(self, title, initial_value, subtitle="", accent_color="#06B6D4", icon_symbol="⚡"):
        super().__init__()
        self.accent_color = accent_color
        self.full_title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(120)
        self.setMinimumHeight(70)

        self.setStyleSheet(f"""
            QFrame {{
                background-color: #111827;
                border: 1px solid #1F2937;
                border-left: 4px solid {accent_color};
                border-radius: 8px;
                padding: 6px;
            }}
            QFrame:hover {{
                border-color: {accent_color};
                background-color: #131E32;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        top_layout = QHBoxLayout()
        self.icon_label = QLabel(icon_symbol)
        self.icon_label.setFont(QFont("Segoe UI Emoji", 10))
        self.icon_label.setStyleSheet("border: none; background: transparent;")
        
        self.title_label = QLabel(title.upper())
        self.title_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.title_label.setStyleSheet("color: #9CA3AF; letter-spacing: 1px; border: none; background: transparent;")
        
        top_layout.addWidget(self.icon_label)
        top_layout.addWidget(self.title_label)
        top_layout.addStretch()

        self.value_label = QLabel(initial_value)
        self.value_label.setFont(QFont("Consolas", 18, QFont.Bold))
        self.value_label.setStyleSheet(f"color: {accent_color}; border: none; background: transparent;")

        self.sub_label = QLabel(subtitle)
        self.sub_label.setFont(QFont("Segoe UI", 7))
        self.sub_label.setStyleSheet("color: #6B7280; border: none; background: transparent;")

        layout.addLayout(top_layout)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sub_label)

    def set_value(self, val, sub=None):
        self.value_label.setText(str(val))
        if sub is not None:
            self.sub_label.setText(str(sub))

    def set_compact_mode(self, is_compact):
        if is_compact:
            self.value_label.setFont(QFont("Consolas", 15, QFont.Bold))
            self.sub_label.setVisible(False)
        else:
            self.value_label.setFont(QFont("Consolas", 18, QFont.Bold))
            self.sub_label.setVisible(True)


class CYBERSCOPE_Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CYBERSCOPE AI // Autonomous SOC Defense Console")
        # Responsive default geometry and low minimum size
        self.setGeometry(40, 40, 1400, 850)
        self.setMinimumSize(780, 520)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0B0F19;
                color: #F9FAFB;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            }
            QWidget {
                color: #F9FAFB;
            }
            QToolTip {
                background-color: #1F2937;
                color: #38BDF8;
                border: 1px solid #38BDF8;
                padding: 4px;
                border-radius: 4px;
            }
        """)

        self.total_scanned = 0
        self.threats_detected = 0
        self.threat_counts_by_class = {}
        self.recent_latencies = []
        self.exported_records = []
        self.historic_flow_records = []

        self.stream_delay_ms = 800
        self.filter_text = ""
        self.filter_severity = "ALL"

        self.load_artifacts()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_flow)

        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def load_artifacts(self):
        print("Loading AI Model and Preprocessors...")
        custom_objects = {
            "PositionalEncoding": PositionalEncoding,
            "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
            "LearnablePositionalEncoding": LearnablePositionalEncoding
        }
        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.df = pd.read_csv(DATA_PATH)
        drop_cols = [c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP"] if c in self.df.columns]
        self.features_df = self.df.drop(columns=drop_cols)
        self.feature_names = self.features_df.columns.tolist()

        self.scaled_features = self.scaler.transform(self.features_df.fillna(0)).astype(np.float32)

        # Check if model has 2D or 1D spatial input
        self.is_2d = (len(self.model.inputs[0].shape) == 4)
        if self.is_2d:
            if self.scaled_features.shape[1] < TARGET_FEATURES:
                pad = TARGET_FEATURES - self.scaled_features.shape[1]
                self.padded_features = np.pad(self.scaled_features, ((0, 0), (0, pad)), mode="constant")
            else:
                self.padded_features = self.scaled_features[:, :TARGET_FEATURES]
            self.images = self.padded_features.reshape(-1, IMAGE_DIM[0], IMAGE_DIM[1], 1)
        else:
            self.images = self.scaled_features[:, :, np.newaxis]

        self.xai = CyberScopeXAI(MODEL_PATH, SCALER_PATH, ENCODER_PATH)
        self.indices = np.random.permutation(np.arange(10, len(self.scaled_features)))
        self.sample_pointer = 0

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)

        # ---------------- 1. DYNAMIC HEADER BAR ----------------
        self.header_frame = QFrame()
        self.header_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #111827, stop:1 #1E293B);
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 4px;
            }
        """)
        
        # Responsive 2-Row or 1-Row layout inside header
        header_vbox = QVBoxLayout(self.header_frame)
        header_vbox.setContentsMargins(8, 4, 8, 4)
        header_vbox.setSpacing(4)

        # Top line: Brand, Pulse, Badge, Clock
        row_top = QHBoxLayout()
        self.status_pulse = QLabel("●")
        self.status_pulse.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.status_pulse.setStyleSheet("color: #10B981; border: none; background: transparent;")

        self.brand_title = QLabel("CYBERSCOPE AI")
        self.brand_title.setFont(QFont("Consolas", 15, QFont.Bold))
        self.brand_title.setStyleSheet("color: #38BDF8; letter-spacing: 2px; border: none; background: transparent;")

        self.badge_sub = QLabel("1D-CNN + TRANSFORMER [POS-ENC]")
        self.badge_sub.setFont(QFont("Segoe UI", 7, QFont.Bold))
        self.badge_sub.setStyleSheet("""
            background-color: #0369A1;
            color: #E0F2FE;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #38BDF8;
        """)

        self.clock_label = QLabel()
        self.clock_label.setFont(QFont("Consolas", 9))
        self.clock_label.setStyleSheet("color: #94A3B8; border: none;")
        self.update_clock()

        row_top.addWidget(self.status_pulse)
        row_top.addWidget(self.brand_title)
        row_top.addWidget(self.badge_sub)
        row_top.addStretch()
        row_top.addWidget(self.clock_label)
        header_vbox.addLayout(row_top)

        # Bottom line: Controls & Speed Slider
        self.row_ctrl = QHBoxLayout()
        self.row_ctrl.setSpacing(8)

        lbl_speed = QLabel("SPEED:")
        lbl_speed.setFont(QFont("Segoe UI", 8, QFont.Bold))
        lbl_speed.setStyleSheet("color: #94A3B8; border: none;")
        
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(100, 2000)
        self.slider_speed.setValue(self.stream_delay_ms)
        self.slider_speed.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.slider_speed.setMinimumWidth(80)
        self.slider_speed.setMaximumWidth(140)
        self.slider_speed.setStyleSheet("""
            QSlider::groove:horizontal { height: 5px; background: #334155; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #38BDF8; border-radius: 2px; }
            QSlider::handle:horizontal { background: #F8FAFC; width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; }
        """)
        self.slider_speed.valueChanged.connect(self.on_speed_changed)

        self.lbl_speed_val = QLabel(f"{self.stream_delay_ms}ms")
        self.lbl_speed_val.setFont(QFont("Consolas", 8))
        self.lbl_speed_val.setStyleSheet("color: #38BDF8; border: none;")

        self.row_ctrl.addWidget(lbl_speed)
        self.row_ctrl.addWidget(self.slider_speed)
        self.row_ctrl.addWidget(self.lbl_speed_val)
        self.row_ctrl.addStretch()

        btn_base_style = """
            QPushButton {
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
                font-size: 10px;
                padding: 5px 12px;
                border-radius: 5px;
                border: none;
            }
        """

        self.btn_start = QPushButton("▶ START")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet(btn_base_style + """
            QPushButton { background-color: #10B981; color: #064E3B; }
            QPushButton:hover { background-color: #34D399; }
        """)
        self.btn_start.clicked.connect(self.start_monitoring)

        self.btn_stop = QPushButton("⏸ PAUSE")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setStyleSheet(btn_base_style + """
            QPushButton { background-color: #334155; color: #F8FAFC; }
            QPushButton:hover { background-color: #475569; }
        """)
        self.btn_stop.clicked.connect(self.stop_monitoring)

        self.btn_step = QPushButton("⏭ STEP")
        self.btn_step.setCursor(Qt.PointingHandCursor)
        self.btn_step.setStyleSheet(btn_base_style + """
            QPushButton { background-color: #0284C7; color: #F0F9FF; }
            QPushButton:hover { background-color: #38BDF8; }
        """)
        self.btn_step.clicked.connect(self.step_one_flow)

        self.btn_clear = QPushButton("🗑 CLEAR")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setStyleSheet(btn_base_style + """
            QPushButton { background-color: #4B5563; color: #F3F4F6; }
            QPushButton:hover { background-color: #6B7280; }
        """)
        self.btn_clear.clicked.connect(self.clear_telemetry)

        self.row_ctrl.addWidget(self.btn_start)
        self.row_ctrl.addWidget(self.btn_stop)
        self.row_ctrl.addWidget(self.btn_step)
        self.row_ctrl.addWidget(self.btn_clear)

        header_vbox.addLayout(self.row_ctrl)
        main_layout.addWidget(self.header_frame)

        # ---------------- 2. METRICS ROW (Responsive Grid) ----------------
        self.metrics_container = QWidget()
        self.metrics_layout = QHBoxLayout(self.metrics_container)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setSpacing(8)

        self.card_scanned = ModernMetricCard("Total Flows", "0", "Evaluated packets", "#38BDF8", "📊")
        self.card_threats = ModernMetricCard("Threats", "0", "0.0% Detection ratio", "#F43F5E", "🛡️")
        self.card_threat_rate = ModernMetricCard("Current State", "CLEAN", "Nominal state", "#10B981", "⚠️")
        self.card_model_status = ModernMetricCard("Engine State", "STANDBY", "1D-CNN + Transformer", "#A855F7", "⚡")

        self.metric_cards = [self.card_scanned, self.card_threats, self.card_threat_rate, self.card_model_status]
        for card in self.metric_cards:
            self.metrics_layout.addWidget(card)

        main_layout.addWidget(self.metrics_container)

        # ---------------- 3. NOTIFICATION BANNER ----------------
        self.alert_banner = QLabel("SYSTEM IDLE // PRESS 'START' TO COMMENCE REAL-TIME INTERCEPTION")
        self.alert_banner.setFont(QFont("Consolas", 9, QFont.Bold))
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setFixedHeight(28)
        self.alert_banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.alert_banner.setStyleSheet("""
            background-color: #1E293B; color: #94A3B8; border: 1px dashed #475569; border-radius: 5px;
        """)
        main_layout.addWidget(self.alert_banner)

        # ---------------- 4. MAIN TABS (Expanding with QSplitters) ----------------
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1E293B; background: #0F172A; border-radius: 8px;
            }
            QTabBar::tab {
                background: #111827; color: #9CA3AF; padding: 7px 16px; margin-right: 3px;
                font-weight: bold; font-size: 10px; border-top-left-radius: 6px; border-top-right-radius: 6px;
                border: 1px solid #1F2937;
            }
            QTabBar::tab:selected {
                background: #0F172A; color: #38BDF8; border-bottom: 2px solid #38BDF8; border-top: 1px solid #38BDF8;
            }
            QTabBar::tab:hover:!selected {
                background: #1F2937; color: #F3F4F6;
            }
        """)

        # TAB 1: Live Telemetry
        self.tab_telemetry = QWidget()
        self.setup_telemetry_tab()
        self.tabs.addTab(self.tab_telemetry, "📡 Live Telemetry")

        # TAB 2: Threat Diagnostics (XAI)
        self.tab_xai = QWidget()
        self.setup_xai_tab()
        self.tabs.addTab(self.tab_xai, "🧠 Threat Diagnostics (XAI)")

        # TAB 3: Flow Inspector
        self.tab_inspector = QWidget()
        self.setup_inspector_tab()
        self.tabs.addTab(self.tab_inspector, "🔍 Flow Inspector")

        # TAB 4: Model Architecture & Benchmarks
        self.tab_benchmarks = QWidget()
        self.setup_benchmarks_tab()
        self.tabs.addTab(self.tab_benchmarks, "📊 Model Specs")

        main_layout.addWidget(self.tabs, stretch=1)

        # ---------------- 5. STATUS BAR ----------------
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(True)
        self.status_bar.setStyleSheet("background-color: #0B0F19; color: #64748B; border-top: 1px solid #1E293B; font-size: 9px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Engine Ready | Model: cyberscope_model.keras (1D-CNN + Transformer) | Accuracy: 95.5% | Macro ROC-AUC: 0.896")

    def setup_telemetry_tab(self):
        layout = QVBoxLayout(self.tab_telemetry)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        lbl_filter = QLabel("FILTER:")
        lbl_filter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        lbl_filter.setStyleSheet("color: #94A3B8;")
        
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Search Flow ID or Attack type...")
        self.txt_search.setStyleSheet("""
            QLineEdit {
                background-color: #1E293B; color: #F8FAFC; border: 1px solid #334155;
                border-radius: 5px; padding: 4px 8px; font-size: 11px;
            }
        """)
        self.txt_search.textChanged.connect(self.on_filter_changed)

        self.combo_severity = QComboBox()
        self.combo_severity.addItems(["ALL", "CLEAN", "MEDIUM", "HIGH", "CRITICAL", "SUSPICIOUS"])
        self.combo_severity.setStyleSheet("""
            QComboBox {
                background-color: #1E293B; color: #F8FAFC; border: 1px solid #334155;
                border-radius: 5px; padding: 4px 8px; font-size: 11px;
            }
        """)
        self.combo_severity.currentTextChanged.connect(self.on_severity_filter_changed)

        self.btn_export = QPushButton("💾 EXPORT CSV")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #38BDF8; color: #0F172A; font-weight: bold;
                padding: 4px 10px; border-radius: 5px; font-size: 10px; border: none;
            }
            QPushButton:hover { background-color: #7DD3FC; }
        """)
        self.btn_export.clicked.connect(self.export_results)

        toolbar.addWidget(lbl_filter)
        toolbar.addWidget(self.txt_search, 1)
        toolbar.addWidget(self.combo_severity)
        toolbar.addWidget(self.btn_export)
        layout.addLayout(toolbar)

        # Dynamic Splitter (Scales cleanly on resize)
        self.telemetry_splitter = QSplitter(Qt.Horizontal)
        self.telemetry_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.telemetry_splitter.setStyleSheet("QSplitter::handle { background: #1E293B; width: 4px; }")

        # Left Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["FLOW ID", "ATTACK VECTOR", "CONFIDENCE", "SEVERITY", "TIMESTAMP"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #111827; border: 1px solid #1F2937; border-radius: 6px; font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1E293B; color: #94A3B8; font-weight: bold; padding: 6px; border: none;
                border-bottom: 2px solid #38BDF8;
            }
            QTableWidget::item { padding: 4px; border-bottom: 1px solid #1F2937; }
            QTableWidget::item:selected { background-color: #1E3A8A; color: #FFFFFF; }
        """)
        self.table.itemClicked.connect(self.on_table_row_clicked)
        self.telemetry_splitter.addWidget(self.table)

        # Right Charts
        charts_container = QWidget()
        charts_layout = QVBoxLayout(charts_container)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(4)

        # Small base size with Expanding QSizePolicy allows smooth shrinking & growing
        self.fig_telemetry = Figure(figsize=(3, 3), facecolor="#111827")
        self.canvas_telemetry = FigureCanvas(self.fig_telemetry)
        self.canvas_telemetry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_telemetry.setMinimumSize(100, 100)
        charts_layout.addWidget(self.canvas_telemetry)
        self.init_telemetry_plots()

        self.telemetry_splitter.addWidget(charts_container)
        self.telemetry_splitter.setStretchFactor(0, 3)
        self.telemetry_splitter.setStretchFactor(1, 2)
        self.telemetry_splitter.setCollapsible(0, False)
        self.telemetry_splitter.setCollapsible(1, False)
        layout.addWidget(self.telemetry_splitter, 1)

    def init_telemetry_plots(self):
        self.fig_telemetry.clear()
        self.ax_donut = self.fig_telemetry.add_subplot(211)
        self.ax_donut.set_facecolor("#111827")
        self.ax_donut.text(0.5, 0.5, "Awaiting Telemetry...", color="#64748B", ha="center", va="center", fontsize=9)
        self.ax_donut.axis("off")

        self.ax_trend = self.fig_telemetry.add_subplot(212)
        self.ax_trend.set_facecolor("#111827")
        self.ax_trend.set_title("Latency Trend (ms)", color="#94A3B8", fontsize=8, pad=4)
        self.ax_trend.tick_params(colors="#64748B", labelsize=7)
        for spine in self.ax_trend.spines.values():
            spine.set_color("#1F2937")
        self.fig_telemetry.tight_layout()
        self.canvas_telemetry.draw()

    def setup_xai_tab(self):
        layout = QVBoxLayout(self.tab_xai)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.xai_splitter = QSplitter(Qt.Horizontal)
        self.xai_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.xai_splitter.setStyleSheet("QSplitter::handle { background: #1E293B; width: 4px; }")

        # Left: Visual Plots
        plots_container = QWidget()
        plots_layout = QVBoxLayout(plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        self.fig_xai_plots = Figure(figsize=(4, 4), facecolor="#111827")
        self.canvas_xai_plots = FigureCanvas(self.fig_xai_plots)
        self.canvas_xai_plots.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_xai_plots.setMinimumSize(100, 100)
        plots_layout.addWidget(self.canvas_xai_plots)
        self.xai_splitter.addWidget(plots_container)

        # Right: Incident Report Console
        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(4)

        console_header = QHBoxLayout()
        console_title = QLabel("SOC INCIDENT INTELLIGENCE")
        console_title.setFont(QFont("Consolas", 10, QFont.Bold))
        console_title.setStyleSheet("color: #38BDF8;")
        console_header.addWidget(console_title)
        console_header.addStretch()

        self.btn_copy_report = QPushButton("📋 COPY")
        self.btn_copy_report.setStyleSheet("background: #1E293B; color: #94A3B8; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 9px;")
        self.btn_copy_report.clicked.connect(self.copy_diagnostic_report)
        console_header.addWidget(self.btn_copy_report)
        console_layout.addLayout(console_header)

        self.exp_text = QTextEdit()
        self.exp_text.setReadOnly(True)
        self.exp_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.exp_text.setStyleSheet("""
            QTextEdit {
                background-color: #0B0F19; color: #10B981; border: 1px solid #1F2937; border-radius: 6px;
                font-family: 'Consolas', monospace; font-size: 11px; padding: 8px; line-height: 1.3;
            }
        """)
        self.exp_text.setText("System standing by. Click 'START' or select a flow from Telemetry to inspect real-time Integrated Gradients.")
        console_layout.addWidget(self.exp_text)

        self.xai_splitter.addWidget(console_container)
        self.xai_splitter.setStretchFactor(0, 1)
        self.xai_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.xai_splitter, 1)

    def setup_inspector_tab(self):
        layout = QVBoxLayout(self.tab_inspector)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        info_bar = QHBoxLayout()
        self.lbl_inspector_flow = QLabel("ACTIVE FLOW: None selected (Click any row in Telemetry to inspect)")
        self.lbl_inspector_flow.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_inspector_flow.setStyleSheet("color: #F8FAFC;")
        info_bar.addWidget(self.lbl_inspector_flow)
        info_bar.addStretch()
        layout.addLayout(info_bar)

        self.table_inspector = QTableWidget()
        self.table_inspector.setColumnCount(4)
        self.table_inspector.setHorizontalHeaderLabels(["FEATURE NAME", "DOMAIN CATEGORY", "VALUE (RAW & SCALED)", "XAI ATTRIBUTION SCORE"])
        self.table_inspector.verticalHeader().setVisible(False)
        self.table_inspector.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_inspector.setShowGrid(False)
        self.table_inspector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_inspector.setStyleSheet("""
            QTableWidget {
                background-color: #111827; border: 1px solid #1F2937; border-radius: 6px; font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1E293B; color: #94A3B8; font-weight: bold; padding: 6px; border: none;
            }
            QTableWidget::item { padding: 4px; border-bottom: 1px solid #1F2937; }
        """)
        layout.addWidget(self.table_inspector, 1)

    def setup_benchmarks_tab(self):
        layout = QVBoxLayout(self.tab_benchmarks)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.bench_splitter = QSplitter(Qt.Horizontal)
        self.bench_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bench_splitter.setStyleSheet("QSplitter::handle { background: #1E293B; width: 4px; }")

        # Left: Model Spec Card
        specs_box = QFrame()
        specs_box.setStyleSheet("background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 10px;")
        s_layout = QVBoxLayout(specs_box)
        s_layout.setContentsMargins(6, 6, 6, 6)

        lbl_spec_title = QLabel("DEPLOYED MODEL ARCHITECTURE")
        lbl_spec_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_spec_title.setStyleSheet("color: #38BDF8; letter-spacing: 1px; border: none;")
        s_layout.addWidget(lbl_spec_title)

        specs_html = f"""
        <table style="color: #F8FAFC; font-size: 11px; line-height: 1.8; width: 100%;">
            <tr><td style="color: #94A3B8;">Architecture:</td><td><b>1D-CNN + Transformer</b></td></tr>
            <tr><td style="color: #94A3B8;">Positional Encoding:</td><td><b>Sinusoidal (Vaswani et al.)</b></td></tr>
            <tr><td style="color: #94A3B8;">Spatial Branch:</td><td><b>Domain 1D-CNN (30 feats)</b></td></tr>
            <tr><td style="color: #94A3B8;">Temporal Window:</td><td><b>10-Step Causal Window</b></td></tr>
            <tr><td style="color: #94A3B8;">Parameters:</td><td><b>100,424 params</b></td></tr>
            <tr><td style="color: #94A3B8;">Holdout Accuracy:</td><td><b style="color: #10B981;">95.55%</b></td></tr>
            <tr><td style="color: #94A3B8;">Weighted F1:</td><td><b style="color: #10B981;">94.89%</b></td></tr>
            <tr><td style="color: #94A3B8;">Macro ROC-AUC:</td><td><b style="color: #38BDF8;">0.8956</b></td></tr>
            <tr><td style="color: #94A3B8;">Weighted ROC-AUC:</td><td><b style="color: #38BDF8;">0.9936</b></td></tr>
            <tr><td style="color: #94A3B8;">False Alarms/1000:</td><td><b style="color: #10B981;">0.00 flows</b></td></tr>
            <tr><td style="color: #94A3B8;">Latency:</td><td><b>0.015 ms/flow (~64.9k fps)</b></td></tr>
        </table>
        """
        spec_text = QLabel(specs_html)
        spec_text.setStyleSheet("border: none; background: transparent;")
        spec_text.setWordWrap(True)
        s_layout.addWidget(spec_text)
        s_layout.addStretch()

        btn_open_folder = QPushButton("📂 OPEN MODELS DIRECTORY")
        btn_open_folder.setStyleSheet("""
            QPushButton {
                background: #1E293B; color: #38BDF8; border: 1px solid #38BDF8;
                padding: 6px; font-weight: bold; border-radius: 4px; font-size: 10px;
            }
            QPushButton:hover { background: #0284C7; color: #FFFFFF; }
        """)
        btn_open_folder.clicked.connect(lambda: os.startfile(os.path.abspath("models")))
        s_layout.addWidget(btn_open_folder)

        self.bench_splitter.addWidget(specs_box)

        # Right: Embed Confusion Matrix & ROC plots
        plots_box = QFrame()
        plots_box.setStyleSheet("background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 6px;")
        p_layout = QVBoxLayout(plots_box)
        p_layout.setContentsMargins(4, 4, 4, 4)

        self.fig_bench = Figure(figsize=(4, 3), facecolor="#111827")
        self.canvas_bench = FigureCanvas(self.fig_bench)
        self.canvas_bench.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_bench.setMinimumSize(100, 100)
        p_layout.addWidget(self.canvas_bench)

        self.render_benchmark_plots()
        self.bench_splitter.addWidget(plots_box)

        self.bench_splitter.setStretchFactor(0, 1)
        self.bench_splitter.setStretchFactor(1, 2)
        layout.addWidget(self.bench_splitter, 1)

    def render_benchmark_plots(self):
        self.fig_bench.clear()
        
        ax1 = self.fig_bench.add_subplot(121)
        ax1.set_facecolor("#111827")
        if os.path.exists(CONFUSION_IMG):
            img = matplotlib.image.imread(CONFUSION_IMG)
            ax1.imshow(img)
            ax1.axis("off")
            ax1.set_title("Holdout Confusion Matrix", color="#F8FAFC", fontsize=9, pad=4)
        else:
            ax1.text(0.5, 0.5, "Confusion matrix ready", color="#94A3B8", ha="center")
            ax1.axis("off")

        ax2 = self.fig_bench.add_subplot(122)
        ax2.set_facecolor("#111827")
        if os.path.exists(ROC_IMG):
            img_roc = matplotlib.image.imread(ROC_IMG)
            ax2.imshow(img_roc)
            ax2.axis("off")
            ax2.set_title("One-vs-Rest ROC Curves", color="#F8FAFC", fontsize=9, pad=4)
        else:
            ax2.text(0.5, 0.5, "ROC curves ready", color="#94A3B8", ha="center")
            ax2.axis("off")

        self.fig_bench.tight_layout()
        self.canvas_bench.draw()

    def resizeEvent(self, event):
        """Dynamic responsiveness hook that adjusts elements when window size changes."""
        super().resizeEvent(event)
        width = event.size().width()
        is_compact = (width < 1050)

        # Compact / Full title and metric card mode
        for card in self.metric_cards:
            card.set_compact_mode(is_compact)

        if width < 900:
            self.brand_title.setText("CYBERSCOPE")
            self.badge_sub.setText("1D-CNN+TF")
            self.btn_start.setText("▶")
            self.btn_stop.setText("⏸")
            self.btn_step.setText("⏭")
            self.btn_clear.setText("🗑")
        else:
            self.brand_title.setText("CYBERSCOPE AI")
            self.badge_sub.setText("1D-CNN + TRANSFORMER [POS-ENC]")
            self.btn_start.setText("▶ START")
            self.btn_stop.setText("⏸ PAUSE")
            self.btn_step.setText("⏭ STEP")
            self.btn_clear.setText("🗑 CLEAR")

        # Dynamically trigger redraw on Matplotlib canvases
        try:
            self.fig_telemetry.tight_layout()
            self.canvas_telemetry.draw_idle()
            self.fig_xai_plots.tight_layout()
            self.canvas_xai_plots.draw_idle()
            self.fig_bench.tight_layout()
            self.canvas_bench.draw_idle()
        except Exception:
            pass

    def update_clock(self):
        now = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.clock_label.setText(f"UTC {now}")

    def on_speed_changed(self, value):
        self.stream_delay_ms = value
        self.lbl_speed_val.setText(f"{value}ms")
        if self.timer.isActive():
            self.timer.start(self.stream_delay_ms)

    def start_monitoring(self):
        self.card_model_status.set_value("ACTIVE", "Streaming network flows")
        self.status_pulse.setStyleSheet("color: #10B981; border: none; background: transparent;")
        self.alert_banner.setText("SYSTEM ACTIVE // REAL-TIME PACKET INTERCEPTION IN PROGRESS")
        self.alert_banner.setStyleSheet("background-color: #064E3B; color: #A7F3D0; border: 1px solid #10B981; border-radius: 5px; font-weight: bold;")
        self.timer.start(self.stream_delay_ms)

    def stop_monitoring(self):
        self.card_model_status.set_value("PAUSED", "Telemetry halted")
        self.status_pulse.setStyleSheet("color: #F59E0B; border: none; background: transparent;")
        self.alert_banner.setText("STREAM PAUSED // SELECT ANY FLOW IN THE TABLE TO EXECUTE DEEP INSPECTION")
        self.alert_banner.setStyleSheet("background-color: #1E293B; color: #FBBF24; border: 1px solid #F59E0B; border-radius: 5px; font-weight: bold;")
        self.timer.stop()

    def step_one_flow(self):
        self.stop_monitoring()
        self.process_next_flow()

    def clear_telemetry(self):
        self.table.setRowCount(0)
        self.total_scanned = 0
        self.threats_detected = 0
        self.threat_counts_by_class.clear()
        self.recent_latencies.clear()
        self.exported_records.clear()
        self.historic_flow_records.clear()
        self.card_scanned.set_value("0", "Evaluated packets")
        self.card_threats.set_value("0", "0.0% Detection ratio")
        self.card_threat_rate.set_value("CLEAN", "Nominal state")
        self.init_telemetry_plots()

    def on_filter_changed(self, text):
        self.filter_text = text.lower()
        self.apply_table_filters()

    def on_severity_filter_changed(self, sev):
        self.filter_severity = sev
        self.apply_table_filters()

    def apply_table_filters(self):
        for row in range(self.table.rowCount()):
            f_id = self.table.item(row, 0).text().lower()
            f_cls = self.table.item(row, 1).text().lower()
            f_sev = self.table.item(row, 3).text()

            match_text = (self.filter_text in f_id or self.filter_text in f_cls) if self.filter_text else True
            match_sev = (self.filter_severity == "ALL" or self.filter_severity == f_sev)
            self.table.setRowHidden(row, not (match_text and match_sev))

    def process_next_flow(self):
        if self.sample_pointer >= len(self.indices):
            self.sample_pointer = 0

        idx = self.indices[self.sample_pointer]
        self.sample_pointer += 1

        seq = self.scaled_features[idx - 10 : idx].reshape(1, 10, -1)
        if self.is_2d:
            spatial = self.images[idx].reshape(1, IMAGE_DIM[0], IMAGE_DIM[1], 1)
        else:
            spatial = self.scaled_features[idx].reshape(1, -1, 1)

        t_start = time.perf_counter()
        preds = self.model.predict([spatial, seq], verbose=0)[0]
        inf_elapsed = (time.perf_counter() - t_start) * 1000.0
        self.recent_latencies.append(inf_elapsed)
        if len(self.recent_latencies) > 30:
            self.recent_latencies.pop(0)

        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        label_str = str(self.label_encoder.inverse_transform([class_idx])[0])

        # False Positive Kill-Switch
        if confidence < 0.85 and label_str.lower() not in ["benign", "normal"]:
            label_str = "Anomaly (Unverified)"
            is_threat = True
            severity = "SUSPICIOUS"
            sev_color = "#F59E0B"
        else:
            is_threat = (label_str.lower() not in ["benign", "normal"])
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
            self.threat_counts_by_class[label_str] = self.threat_counts_by_class.get(label_str, 0) + 1

        ratio = (self.threats_detected / self.total_scanned) * 100.0 if self.total_scanned > 0 else 0.0
        self.card_scanned.set_value(f"{self.total_scanned:,}", f"Flow #{idx:06d}")
        self.card_threats.set_value(f"{self.threats_detected:,}", f"{ratio:.1f}% ratio")
        self.card_threat_rate.set_value(severity, f"{label_str}")

        if severity in ["HIGH", "CRITICAL"]:
            self.alert_banner.setText(f"🚨 ALERT: {severity} [{label_str.upper()}] ON FLOW #{idx:06d} (CONF: {confidence*100:.1f}%)")
            self.alert_banner.setStyleSheet("background-color: #881337; color: #FECDD3; border: 1px solid #E11D48; border-radius: 5px; font-weight: bold;")

        flow_id = f"FLOW-{idx:06d}"
        conf_str = f"{confidence * 100:.1f}%"
        timestamp_str = QDateTime.currentDateTime().toString("hh:mm:ss")

        record_entry = {
            "flow_id": flow_id, "label_str": label_str, "confidence": conf_str,
            "severity": severity, "timestamp": timestamp_str, "spatial": spatial[0],
            "seq": seq[0], "class_idx": class_idx, "idx": idx, "sev_color": sev_color,
            "raw_features": self.features_df.iloc[idx].to_dict(),
            "scaled_features": self.scaled_features[idx]
        }
        self.historic_flow_records.append(record_entry)
        self.exported_records.append({"Flow_ID": flow_id, "Classification": label_str, "Confidence": conf_str, "Severity": severity, "Timestamp": timestamp_str})

        # Add to Table
        row = self.table.rowCount()
        self.table.insertRow(row)

        item_id = QTableWidgetItem(flow_id)
        item_id.setFont(QFont("Consolas", 9))
        item_id.setForeground(QColor("#94A3B8"))

        item_cls = QTableWidgetItem(label_str)
        item_cls.setFont(QFont("Segoe UI", 9, QFont.Bold))
        item_cls.setForeground(QColor(sev_color))

        item_conf = QTableWidgetItem(conf_str)
        item_conf.setFont(QFont("Consolas", 9))
        item_conf.setForeground(QColor("#F8FAFC"))

        item_sev = QTableWidgetItem(severity)
        item_sev.setFont(QFont("Segoe UI", 9, QFont.Bold))
        item_sev.setForeground(QColor(sev_color))

        item_time = QTableWidgetItem(timestamp_str)
        item_time.setFont(QFont("Consolas", 9))
        item_time.setForeground(QColor("#64748B"))

        self.table.setItem(row, 0, item_id)
        self.table.setItem(row, 1, item_cls)
        self.table.setItem(row, 2, item_conf)
        self.table.setItem(row, 3, item_sev)
        self.table.setItem(row, 4, item_time)

        if self.filter_text or self.filter_severity != "ALL":
            match_t = (self.filter_text in flow_id.lower() or self.filter_text in label_str.lower()) if self.filter_text else True
            match_s = (self.filter_severity == "ALL" or self.filter_severity == severity)
            self.table.setRowHidden(row, not (match_t and match_s))

        self.table.scrollToBottom()
        self.update_telemetry_charts()
        self.render_flow_diagnostics(record_entry)

    def on_table_row_clicked(self, item):
        row = item.row()
        if 0 <= row < len(self.historic_flow_records):
            record = self.historic_flow_records[row]
            self.render_flow_diagnostics(record)
            self.alert_banner.setText(f"INSPECTING HISTORIC FLOW: {record['flow_id']} [{record['label_str'].upper()}]")
            self.alert_banner.setStyleSheet("background-color: #1E293B; color: #38BDF8; border: 1px solid #38BDF8; border-radius: 5px;")

    def render_flow_diagnostics(self, record):
        spatial = record["spatial"]
        seq = record["seq"]
        class_idx = record["class_idx"]
        label_str = record["label_str"]
        conf_str = record["confidence"]
        severity = record["severity"]
        sev_color = record["sev_color"]
        flow_id = record["flow_id"]

        shap_results = self.xai.compute_shap_attributions(spatial, seq, self.feature_names, class_idx, top_k=7)
        top_f1, v1 = shap_results[-1]
        top_f2, v2 = shap_results[-2]

        if severity == "CLEAN":
            remediation = "✓ Clean flow adhering to normal baseline behavior."
        elif "portscan" in label_str.lower():
            remediation = "⚠️ RECONNAISSANCE ALERT: Implement rate-limiting on probe ports."
        elif "dos" in label_str.lower():
            remediation = "🚨 ACTIVE DOS: Throttle persistent connections; activate scrubbing center."
        elif "ddos" in label_str.lower():
            remediation = "🚨 DDOS FLOOD: Apply geo-rate limiting; drop non-whitelisted UDP/ICMP."
        elif "bruteforce" in label_str.lower():
            remediation = "⚠️ BRUTE-FORCE: Enforce exponential backoff on auth endpoint."
        else:
            remediation = "⚠️ UNVERIFIED ANOMALY: Isolate source in sandbox for deep payload audit."

        report_lines = [
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  CYBERSCOPE SOC INTELLIGENCE REPORT // {flow_id}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  VERDICT         : {label_str.upper()} ({conf_str})",
            f"  SEVERITY        : {severity}",
            f"  TIMESTAMP       : {record['timestamp']}\n",
            f"  TOP INTEGRATED GRADIENTS XAI DRIVERS:",
            f"   1. {top_f1:<24} : {'+' if v1>0 else ''}{v1:.5f} ({'Pushes Threat' if v1>0 else 'Suppresses'})",
            f"   2. {top_f2:<24} : {'+' if v2>0 else ''}{v2:.5f} ({'Pushes Threat' if v2>0 else 'Suppresses'})\n",
            f"  RECOMMENDED SOC REMEDIATION:",
            f"  {remediation}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ]
        self.exp_text.setText("\n".join(report_lines))
        self.render_xai_plots(spatial, shap_results, severity != "CLEAN", sev_color)
        self.render_inspector_table(record, shap_results)

    def render_xai_plots(self, spatial_data, shap_results, is_threat, accent_color):
        self.fig_xai_plots.clear()

        # Plot 1: 1D Tabular Feature Signature
        ax1 = self.fig_xai_plots.add_subplot(211)
        ax1.set_facecolor("#111827")
        
        flat_feats = spatial_data.flatten()[:30]
        x_idx = np.arange(len(flat_feats))
        bar_colors = ["#F43F5E" if abs(val) > 1.5 else "#38BDF8" for val in flat_feats]

        ax1.bar(x_idx, flat_feats, color=bar_colors, width=0.6)
        ax1.set_title("1D Tabular Feature Signature (Standardized Deviation)", color="#F8FAFC", fontsize=8, pad=4)
        ax1.set_xlabel("Feature Index (0-29)", color="#94A3B8", fontsize=7)
        ax1.tick_params(colors="#94A3B8", labelsize=7)
        ax1.axhline(0, color="#334155", linewidth=1)
        for spine in ax1.spines.values():
            spine.set_color("#1F2937")

        # Plot 2: Integrated Gradients Horizontal Bars
        ax2 = self.fig_xai_plots.add_subplot(212)
        ax2.set_facecolor("#111827")

        fnames = [item[0][:16] for item in shap_results]
        fvals = [item[1] for item in shap_results]
        bar_col = ["#F43F5E" if v > 0 else "#10B981" for v in fvals]

        y_pos = np.arange(len(fnames))
        ax2.barh(y_pos, fvals, color=bar_col, height=0.55)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(fnames, color="#94A3B8", fontsize=7)
        ax2.axvline(0, color="#475569", linewidth=1)
        ax2.tick_params(colors="#94A3B8", labelsize=7)
        ax2.set_title("Integrated Gradients Attributions (Red: Threat, Green: Benign)", color="#F8FAFC", fontsize=8, pad=4)
        for spine in ax2.spines.values():
            spine.set_color("#1F2937")

        self.fig_xai_plots.tight_layout()
        self.canvas_xai_plots.draw()

    def render_inspector_table(self, record, shap_results):
        self.lbl_inspector_flow.setText(f"INSPECTING FLOW: {record['flow_id']} | VERDICT: {record['label_str'].upper()} ({record['confidence']})")
        attr_dict = {item[0]: item[1] for item in shap_results}
        self.table_inspector.setRowCount(len(self.feature_names))

        categories = {
            "Destination Port": "Endpoints & Ports",
            "min_seg_size_forward": "Endpoints & Ports",
            "Init_Win_bytes_forward": "TCP Window Control",
            "Init_Win_bytes_backward": "TCP Window Control",
            "Flow Duration": "Timing & Duration",
            "Flow IAT Mean": "Timing & Duration",
            "Flow IAT Max": "Timing & Duration",
            "Flow IAT Min": "Timing & Duration",
            "Active Mean": "Timing & Duration",
            "Idle Mean": "Timing & Duration",
            "Flow Bytes/s": "Traffic Volume & Rate",
            "Flow Packets/s": "Traffic Volume & Rate",
            "Total Fwd Packets": "Traffic Volume & Rate",
            "Total Backward Packets": "Traffic Volume & Rate",
            "Total Length of Fwd Packets": "Traffic Volume & Rate",
            "Total Length of Bwd Packets": "Traffic Volume & Rate",
            "Down/Up Ratio": "Traffic Volume & Rate",
            "FIN Flag Count": "Protocol Flags",
            "SYN Flag Count": "Protocol Flags",
            "RST Flag Count": "Protocol Flags",
            "PSH Flag Count": "Protocol Flags",
            "ACK Flag Count": "Protocol Flags"
        }

        for i, fname in enumerate(self.feature_names):
            cat = categories.get(fname, "Packet Size Distribution")
            raw_v = record["raw_features"].get(fname, 0.0)
            scaled_v = record["scaled_features"][i]
            attr_v = attr_dict.get(fname, 0.0)

            item_f = QTableWidgetItem(fname)
            item_f.setFont(QFont("Consolas", 9))
            item_f.setForeground(QColor("#F8FAFC"))

            item_cat = QTableWidgetItem(cat)
            item_cat.setFont(QFont("Segoe UI", 8))
            item_cat.setForeground(QColor("#38BDF8"))

            item_val = QTableWidgetItem(f"{raw_v:,.2f}  (std: {scaled_v:+.2f})")
            item_val.setFont(QFont("Consolas", 8))
            item_val.setForeground(QColor("#94A3B8"))

            attr_str = f"{attr_v:+.5f}" if fname in attr_dict else "0.00000"
            item_attr = QTableWidgetItem(attr_str)
            item_attr.setFont(QFont("Consolas", 9, QFont.Bold))
            item_attr.setForeground(QColor("#F43F5E" if attr_v > 0 else "#10B981" if attr_v < 0 else "#64748B"))

            self.table_inspector.setItem(i, 0, item_f)
            self.table_inspector.setItem(i, 1, item_cat)
            self.table_inspector.setItem(i, 2, item_val)
            self.table_inspector.setItem(i, 3, item_attr)

    def update_telemetry_charts(self):
        self.ax_donut.clear()
        self.ax_donut.set_facecolor("#111827")

        if self.threat_counts_by_class:
            labels = list(self.threat_counts_by_class.keys())
            sizes = list(self.threat_counts_by_class.values())
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = self.ax_donut.pie(
                sizes, labels=labels, autopct="%1.0f%%", startangle=140,
                colors=colors, pctdistance=0.75,
                textprops=dict(color="#94A3B8", fontsize=7)
            )
            for at in autotexts:
                at.set_color("#FFFFFF")
                at.set_fontsize(7)
                at.set_weight("bold")
            centre_circle = plt.Circle((0, 0), 0.55, fc="#111827")
            self.ax_donut.add_artist(centre_circle)
            self.ax_donut.set_title("Attack Distribution", color="#F8FAFC", fontsize=8, pad=4)
        else:
            self.ax_donut.text(0.5, 0.5, "Nominal Traffic", color="#10B981", ha="center", va="center", fontsize=8, weight="bold")
            self.ax_donut.axis("off")

        self.ax_trend.clear()
        self.ax_trend.set_facecolor("#111827")
        if len(self.recent_latencies) > 1:
            self.ax_trend.plot(self.recent_latencies, color="#38BDF8", linewidth=1.5, marker="o", markersize=2)
            self.ax_trend.fill_between(range(len(self.recent_latencies)), self.recent_latencies, color="#38BDF8", alpha=0.15)
            self.ax_trend.set_title(f"Inference Latency: {self.recent_latencies[-1]:.2f} ms", color="#F8FAFC", fontsize=8, pad=4)
            self.ax_trend.tick_params(colors="#64748B", labelsize=7)
            self.ax_trend.grid(axis="y", linestyle=":", color="#1F2937")
            for spine in self.ax_trend.spines.values():
                spine.set_color("#1F2937")

        self.fig_telemetry.tight_layout()
        self.canvas_telemetry.draw()

    def copy_diagnostic_report(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.exp_text.toPlainText())
        QMessageBox.information(self, "Copied", "SOC Diagnostic Report copied to clipboard.")

    def export_results(self):
        if not self.exported_records:
            QMessageBox.warning(self, "No Data", "No telemetry records to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save SOC Logs", "cyberscope_soc_logs.csv", "CSV Files (*.csv)")
        if file_path:
            pd.DataFrame(self.exported_records).to_csv(file_path, index=False)
            QMessageBox.information(self, "Exported", f"Successfully exported {len(self.exported_records):,} records to:\n{file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CYBERSCOPE_Dashboard()
    window.show()
    sys.exit(app.exec_())
