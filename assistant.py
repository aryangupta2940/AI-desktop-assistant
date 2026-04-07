import tkinter as tk
import speech_recognition as sr
import webbrowser
import datetime
import urllib.parse
import os

# App paths (edit if needed)
apps = {
    "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "notepad": "C:\\Windows\\System32\\notepad.exe",
    "vscode": "C:\\Users\\aryan\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
}

# 🧠 Intent detection
def process_command(command):
    command = command.lower()

    if any(word in command for word in ["search", "find", "look up"]):
        return "search"

    elif any(word in command for word in ["open", "go to", "launch"]):
        return "open"

    elif "time" in command:
        return "time"

    else:
        return "unknown"

# 🎤 Voice input
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

#  Command execution (SAFE)
def execute(command):
    try:
        intent = process_command(command)

        #  SEARCH
        if intent == "search":
            query = command.replace("search", "").strip()

            if query == "":
                output_text.insert(tk.END, "Please type something to search\n")
                return

            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            webbrowser.open(url)
            output_text.insert(tk.END, f"Searching: {query}\n")

        #  OPEN
        elif intent == "open":
            site = command.replace("open", "").replace("go to", "").replace("launch", "").strip()

            if site == "":
                output_text.insert(tk.END, "Please specify what to open\n")
                return

            if site in apps:
                os.startfile(apps[site])
                output_text.insert(tk.END, f"Opening {site}\n")
            else:
                url = f"https://www.{site}.com"
                webbrowser.open(url)
                output_text.insert(tk.END, f"Opening {site} in browser\n")

        #  TIME
        elif intent == "time":
            current_time = datetime.datetime.now().strftime("%H:%M")
            output_text.insert(tk.END, f"Time: {current_time}\n")

        else:
            output_text.insert(tk.END, "I don't understand\n")

    except Exception as e:
        output_text.insert(tk.END, f"Error: {str(e)}\n")

#  Text input handler (SAFE)
def text_command():
    try:
        command = entry.get().strip()

        if command == "":
            return

        output_text.insert(tk.END, f"You: {command}\n")
        execute(command)

        entry.delete(0, tk.END)

    except Exception as e:
        output_text.insert(tk.END, f"Error: {str(e)}\n")

# p Voice button handler
def voice_command():
    command = listen()
    if command:
        execute(command)

#  GUI Setup
root = tk.Tk()
root.title("AI Assistant")
root.geometry("500x500")

# Output box
output_text = tk.Text(root, height=20, width=60)
output_text.pack(pady=10)

# Input field
entry = tk.Entry(root, width=40)
entry.pack(pady=5)

# Buttons
btn_text = tk.Button(root, text="Run Text Command", command=text_command)
btn_text.pack(pady=5)

btn_voice = tk.Button(root, text="Use Voice", command=voice_command)
btn_voice.pack(pady=5)

# Start GUI
root.mainloop() 