import streamlit as st
import speech_recognition as sr
import pyttsx3

def speak(text):
    """Converts text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listens to a voice command and returns the recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Reduces noise impact
        try:
            audio = recognizer.listen(source, timeout=5)  # Listens for 5 seconds
            text = recognizer.recognize_google(audio)  # Uses Google Speech Recognition
            print(f"You said: {text}")
            return text.lower()
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand. Please repeat.")
            return None
        except sr.RequestError:
            speak("Speech service is unavailable right now.")
            return None

def voice_assistant():
    """Listens to a voice command and responds accordingly."""
    command = listen()
    if command:
        if "hello" in command:
            response = "Hello! How can I assist you today?"
        elif "your name" in command:
            response = "I am your voice assistant!"
        elif "goodbye" in command or "exit" in command:
            response = "Goodbye! Have a great day."
        else:
            response = "I didn't understand that. Can you repeat?"
        
        speak(response)
        return response

if __name__ == "__main__":
    voice_assistant()


