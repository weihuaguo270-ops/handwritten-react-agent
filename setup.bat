@echo off
REM hagent-setup.bat — 将 hagent 添加到系统 PATH
REM 以管理员身份运行一次即可

set "TARGET_DIR=%~dp0"
set "CURRENT_PATH=%PATH%"

echo "%CURRENT_PATH%" | find /i "%TARGET_DIR%" >nul
if %errorlevel% equ 0 (
    echo hagent 已在 PATH 中
) else (
    echo 正在将 %TARGET_DIR% 添加到用户 PATH...
    setx PATH "%TARGET_DIR%;%PATH%"
    echo 完成！请重新打开终端后即可直接使用 hagent
)
echo.
echo 使用方法:
echo   hagent                   启动交互模式
echo   hagent run "你的问题"    单次执行
echo   hagent config            查看配置
