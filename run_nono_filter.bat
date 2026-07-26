@echo off
REM Launch directly from the source tree; no package installation is required.
set "PYTHONPATH=%~dp0src"
py -3 -m nono_filter.main gui
if errorlevel 1 pause
