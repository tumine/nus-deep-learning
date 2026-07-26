详细启动流程
1. 树莓派端执行 `export ROBOT_AUDIO_SERVER_URL="http://100.107.163.5:8000/api/audio"`
2. 树莓派端执行 `python ws_car_control_<latest version>.py`
3. 树莓派端执行 `python webstream.py`
4. 笔记本电脑端执行 `python main_speech.py`
5. 树莓派端在 `ws_car_control.py` 命令行上输入 `start`

