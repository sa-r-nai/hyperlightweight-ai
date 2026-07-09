@echo off
chcp 65001 >nul
cd /d "%~dp0"
python terminal_chat.py --checkpoint .\checkpoints_5m_ko_chat\best.pt --preset 5m
