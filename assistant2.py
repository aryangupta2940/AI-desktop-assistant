import tkinter as tk
import speech_recognition as sr
import webbrowser
import datetime
import urllib.parse
import os
import pywhatkit
import re

# App paths (edit if needed)
apps = {
    "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "notepad": "C:\\Windows\\System32\\notepad.exe",
    "vscode": "C:\\Users\\aryan\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
}

# Intent detection
def process_command(command):
    command = command.lower()

    if any(word in command for word in ["search", "find", "look up"]):
        return "search"

    elif any(word in command for word in ["open", "go to", "launch"]):
        return "open"

    elif "time" in command:
        return "time"

    elif "message" in command or "send" in command:
        return "message"

    else:
        return "unknown"

# Voice input
def listen():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            output_text.insert(tk.END, "\nListening...\n")
            root.update()
            audio = recognizer.listen(source)

            command = recognizer.recognize_google(audio)
            output_text.insert(tk.END, f"You: {command}\n")
            return command.lower()

    except Exception as e:
        output_text.insert(tk.END, f"Voice error: {str(e)}\n")
        return ""

# Execute command
def execute(command):
    try:
        intent = process_command(command)

        # SEARCH
        if intent == "search":
            query = command.replace("search", "").strip()

            if query == "":
                output_text.insert(tk.END, "Enter something to search\n")
                return

            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            webbrowser.open(url)
            output_text.insert(tk.END, f"Searching: {query}\n")

        # OPEN
        elif intent == "open":
            site = command.replace("open", "").replace("go to", "").strip()

            if site == "":
                output_text.insert(tk.END, "Specify what to open\n")
                return

            if site in apps:
                os.startfile(apps[site])
                output_text.insert(tk.END, f"Opening {site}\n")
            else:
                url = f"https://www.{site}.com"
                webbrowser.open(url)
                output_text.insert(tk.END, f"Opening {site}\n")

        # TIME
        elif intent == "time":
            current_time = datetime.datetime.now().strftime("%H:%M")
            output_text.insert(tk.END, f"Time: {current_time}\n")

        # WHATSAPP MESSAGE
        elif intent == "message":
            try:
                match = re.search(r"send message to (\d+) (.+)", command)

                if match:
                    number = match.group(1)
                    message = match.group(2)

                    output_text.insert(tk.END, f"Sending message to {number}\n")

                    pywhatkit.sendwhatmsg_instantly(
                        phone_no="+91" + number,
                        message=message,
                        wait_time=10,
                        tab_close=True
                    )

                else:
                    output_text.insert(tk.END, "Format: send message to <number> <text>\n")

            except Exception as e:
                output_text.insert(tk.END, f"Error: {str(e)}\n")

        else:
            output_text.insert(tk.END, "Command not understood\n")

    except Exception as e:
        output_text.insert(tk.END, f"Error: {str(e)}\n")

# Text command
def text_command():
    command = entry.get().strip()

    if command == "":
        return

    output_text.insert(tk.END, f"You: {command}\n")
    execute(command)
    entry.delete(0, tk.END)

# Voice command
def voice_command():
    command = listen()
    if command:
        execute(command)

# GUI Setup
root = tk.Tk()
root.title("AI Assistant")
root.geometry("600x550")
root.configure(bg="#1e1e1e")

# Title
title = tk.Label(root, text="AI Assistant", font=("Arial", 18, "bold"),
                 bg="#1e1e1e", fg="white")
title.pack(pady=10)

# Output box
output_text = tk.Text(root, height=18, width=70,
                      bg="#2b2b2b", fg="white",
                      insertbackground="white")
output_text.pack(pady=10)

# Input field
entry = tk.Entry(root, width=40, bg="#333333", fg="white",
                 insertbackground="white")
entry.pack(pady=5)

# Buttons frame
frame = tk.Frame(root, bg="#1e1e1e")
frame.pack(pady=10)

btn_text = tk.Button(frame, text="Run Command", command=text_command,
                     bg="#4CAF50", fg="white", width=15)
btn_text.grid(row=0, column=0, padx=10)

btn_voice = tk.Button(frame, text="Voice Input", command=voice_command,
                      bg="#2196F3", fg="white", width=15)
btn_voice.grid(row=0, column=1, padx=10)

# Start GUI
root.mainloop()