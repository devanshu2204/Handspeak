import cv2
import numpy as np
from recognition import recognize_hand_sign
import time
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
import re  # Importing re for handling the regex logic in calculations

# Text-to-speech engine setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)

# Constants
DELETE_HOLD_TIME = 5
FRAME_RESIZE = (320, 240)
WORD_COOLDOWN = 1.5
CURSOR_BLINK_TIME = 0.5

# States
formed_text = []
last_detected = ""
delete_start_time = None
last_word_time = 0
cursor_visible = True
last_cursor_toggle = time.time()

# Calculator mode state
calculator_mode = False
calc_expression = ""

# Allowed characters in calculator mode
allowed_symbols = set("0123456789+-*/().=")

# Camera
camera = cv2.VideoCapture(0)

# GUI Setup
root = tk.Tk()
root.title("HandSpeak - Press 'c' to toggle Calculator Mode")
root.configure(bg="#f0f2f6")

# Center window
window_width = 640
window_height = 480 + 100
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cord = int((screen_width / 2) - (window_width / 2))
y_cord = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")

# Video display
video_label = tk.Label(root)
video_label.pack(pady=10)

# Text display
text_frame = tk.Frame(root, bg="#f0f2f6")
text_frame.pack(fill="x", pady=(10, 20))

text_display = tk.Label(
    text_frame,
    text="📝 Formed Sentence:",
    font=("Courier", 14),
    anchor="w",
    justify="left",
    bg="#f0f2f6"
)
text_display.pack(side="top", anchor="w", padx=20)

sentence_box = tk.Label(
    text_frame,
    text="",
    font=("Courier", 18),
    bg="#ffffff",
    fg="#000000",
    height=2,
    anchor="w",
    relief="sunken",
    padx=10
)
sentence_box.pack(fill="x", padx=20)

# Add label for live calculation result
calc_result_label = tk.Label(
    text_frame,
    text="",
    font=("Courier", 16),
    bg="#f0f2f6",
    fg="#333333",
    anchor="w",
    padx=20
)
calc_result_label.pack(fill="x", padx=20, pady=(5, 0))

def update_gui():
    global last_detected, delete_start_time, last_word_time, cursor_visible
    global last_cursor_toggle, formed_text, calculator_mode, calc_expression

    ret, frame = camera.read()
    if not ret:
        sentence_box.config(text="Error: Could not access webcam.")
        return

    frame = cv2.resize(frame, FRAME_RESIZE)
    detected_sign, processed_frame = recognize_hand_sign(frame)
    current_time = time.time()

    if calculator_mode:
        def is_operator(c): return c in "+-*/"
        def is_number(c): return c.isdigit() or c == "."

        live_result = ""

        if detected_sign and all(char in allowed_symbols for char in detected_sign):
            if detected_sign == "[DEL]":
                calc_expression = calc_expression[:-1]
            elif detected_sign != last_detected:
                if is_operator(detected_sign):
                    if calc_expression and not is_operator(calc_expression[-1]):
                        calc_expression += detected_sign
                elif detected_sign == ".":
                    last_number = re.split(r'[+\-*/]', calc_expression)[-1]
                    if "." not in last_number:
                        calc_expression += "."
                elif detected_sign.isdigit():
                    calc_expression += detected_sign

        # Try evaluating the expression live
        try:
            if calc_expression and not is_operator(calc_expression[-1]):
                live_result = str(eval(calc_expression))
        except:
            live_result = "Error"

        # Update result label with live calculation
        calc_result_label.config(text=f"= {live_result}" if live_result else "")

        display_text = f"🧮 Calculator Mode:\n{calc_expression}"

    else:
        # Sentence Mode
        if detected_sign == "[DEL]":
            if delete_start_time is None:
                delete_start_time = current_time

            elapsed_time = current_time - delete_start_time

            if elapsed_time >= DELETE_HOLD_TIME:
                formed_text = []
            elif detected_sign != last_detected and formed_text:
                formed_text.pop()
        else:
            delete_start_time = None

        # Only accept gestures that are NOT calculator symbols
        if (
            detected_sign
            and detected_sign != "[DEL]"
            and detected_sign not in allowed_symbols
            and detected_sign != last_detected
            and (current_time - last_word_time) >= WORD_COOLDOWN
        ):
            formed_text.append(detected_sign)
            last_word_time = current_time

            tts_engine.say(detected_sign)
            tts_engine.runAndWait()

        if current_time - last_cursor_toggle >= CURSOR_BLINK_TIME:
            cursor_visible = not cursor_visible
            last_cursor_toggle = current_time

        cursor = "|" if cursor_visible else " "
        display_text = "📝 Formed Sentence:\n" + " ".join(formed_text) + " " + cursor

        # Clear calculation result when in sentence mode
        calc_result_label.config(text="")

    last_detected = detected_sign

    sentence_box.config(text=display_text)

    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(processed_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(30, update_gui)

def toggle_mode(event):
    global calculator_mode, calc_expression, formed_text
    calculator_mode = not calculator_mode
    calc_expression = ""
    formed_text = []
    mode = "Calculator Mode" if calculator_mode else "Sentence Mode"
    print(f"Switched to {mode}")

def on_closing():
    camera.release()
    root.destroy()

root.bind("<c>", toggle_mode)  # Press 'c' to toggle mode
root.protocol("WM_DELETE_WINDOW", on_closing)

update_gui()
root.mainloop()
