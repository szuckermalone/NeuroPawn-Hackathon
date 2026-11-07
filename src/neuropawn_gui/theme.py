"""OpenBCI-GUI-inspired look: navy top bar, light-gray workspace, white widget panels."""

NAVY = "#1F456E"
NAVY_DARK = "#173552"
WORKSPACE = "#D5D8DC"
PANEL = "#FFFFFF"
PANEL_TITLE = "#EEF0F2"
BORDER = "#B9BEC4"
TEXT = "#1B1F23"
MUTED = "#5F6B76"
GREEN = "#2E8B57"
GREEN_HOVER = "#277A4C"
RED = "#C8382F"
RED_HOVER = "#AE2F27"
AMBER = "#C98A00"

# OpenBCI GUI channel colors (ch1..ch8).
CHANNEL_COLORS = [
    "#818181", "#7C4B8D", "#36579E", "#317159",
    "#DDB20D", "#FD5E34", "#E0382D", "#A25231",
]
BAND_COLORS = ["#36579E", "#317159", "#DDB20D", "#FD5E34", "#E0382D"]
IMU_COLORS = ["#E0382D", "#317159", "#36579E"]

STYLESHEET = f"""
QMainWindow, QWidget#Workspace {{ background: {WORKSPACE}; }}
QWidget {{ color: {TEXT}; font-size: 12px; }}

QFrame#TopBar {{ background: {NAVY}; border-bottom: 2px solid {NAVY_DARK}; }}
QFrame#TopBar QLabel {{ color: #E8EEF5; }}
QLabel#AppTitle {{ color: white; font-size: 17px; font-weight: 700; letter-spacing: 1px; }}
QLabel#AppSubtitle {{ color: #A9C1DA; font-size: 11px; }}
QLabel#StateLabel {{ font-weight: 600; padding: 0 6px; }}

QFrame#TopBar QComboBox {{
    background: white; border: 1px solid {NAVY_DARK}; border-radius: 3px; padding: 3px 6px; min-height: 20px;
}}
QPushButton {{
    background: #F4F5F7; border: 1px solid {BORDER}; border-radius: 3px; padding: 5px 12px;
}}
QPushButton:hover {{ background: #E7EAEE; }}
QPushButton:disabled {{ color: #9AA3AD; background: #E9EBEE; }}
QFrame#TopBar QPushButton {{ border: 1px solid {NAVY_DARK}; }}
QFrame#TopBar QToolButton {{ background: #F4F5F7; border: 1px solid {NAVY_DARK}; border-radius: 3px; padding: 4px 7px; }}
QPushButton#StartButton {{ background: {GREEN}; color: white; font-weight: 700; border: none; }}
QPushButton#StartButton:hover {{ background: {GREEN_HOVER}; }}
QPushButton#StopButton {{ background: {RED}; color: white; font-weight: 700; border: none; }}
QPushButton#StopButton:hover {{ background: {RED_HOVER}; }}
QPushButton#StartButton:disabled, QPushButton#StopButton:disabled {{ background: #8A99A8; color: #E3E7EB; }}
QPushButton#RecordButton {{ font-weight: 700; }}
QPushButton#RecordButton[recording="true"] {{ background: {RED}; color: white; border: none; }}

QFrame#Panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 2px; }}
QFrame#PanelTitleBar {{ background: {PANEL_TITLE}; border-bottom: 1px solid {BORDER}; }}
QLabel#PanelTitle {{ font-weight: 700; font-size: 13px; }}
QFrame#PanelTitleBar QComboBox {{ background: white; border: 1px solid {BORDER}; border-radius: 3px; padding: 2px 6px; }}

QPushButton#ChannelButton {{
    border-radius: 13px; min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px;
    padding: 0; font-weight: 700; color: white; border: none;
}}
QLabel#Readout {{ font-family: Menlo, Consolas, monospace; font-size: 11px; color: {MUTED}; }}

QTabWidget::pane {{ border: none; background: {PANEL}; }}
QTabBar::tab {{ background: {PANEL_TITLE}; border: 1px solid {BORDER}; padding: 5px 12px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: white; border-bottom-color: white; font-weight: 600; }}

QTableWidget {{ gridline-color: #E3E6EA; font-family: Menlo, Consolas, monospace; font-size: 12px; }}
QHeaderView::section {{ background: {PANEL_TITLE}; border: none; border-bottom: 1px solid {BORDER}; padding: 4px; font-weight: 600; }}

QStatusBar {{ background: {NAVY_DARK}; color: #DCE6F0; }}
QStatusBar QLabel {{ color: #DCE6F0; padding: 0 8px; }}
QSplitter::handle {{ background: {WORKSPACE}; }}
"""
