from audio.speech_to_text import convert_speech_to_text
import keyboard
import time


def check_key_press(event):
    if event.scan_code == 74:
        text_result = convert_speech_to_text()


if __name__ == "__main__":
    keyboard.on_press_key('subtract', check_key_press)
    keyboard.wait('esc')
    keyboard.unhook_all()
