@ECHO OFF
REM setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )
chcp 65001
set PYTHONIOENCODING=utf-8
md logs
call conda activate qlbt
python runall.py
