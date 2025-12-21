import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import os
from utils.career_info import career_descriptions, skill_suggestions

model_path = 'model/career_model.pkl'
if not os.path.exists(model_path):
    messagebox.showerror("Error", "Model not found! Run train_model.py first.")
    exit()

with open(model_path, 'rb') as f:
    model, le = pickle.load(f)

careers = le.classes_

def predict_career():
    try:
        inputs = [
            float(entry_db.get()),
            float(entry_cloud.get()),
            float(entry_prog.get()),
            float(entry_hack.get()),
            float(entry_mgmt.get()),
            float(entry_game.get()),
            float(entry_ai.get()),
            float(entry_math.get()),
            float(entry_science.get()),
            float(entry_person.get())
        ]
        probas = model.predict_proba([inputs])[0]
        top3_idx = probas.argsort()[-3:][::-1]
        top3_careers = [careers[i] for i in top3_idx]
        top3_probs = [probas[i] * 100 for i in top3_idx]
        result_text = "Top Recommended Careers:\n\n"
        for i, (career, prob) in enumerate(zip(top3_careers, top3_probs), 1):
            result_text += f"{i}. {career} ({prob:.1f}% match)\n"
            result_text += f"   Description: {career_descriptions.get(career, 'N/A')}\n"
            result_text += f"   Suggestion: {skill_suggestions.get(career, 'Keep learning!')}\n\n"
        result_label.config(text=result_text)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers (0-10 for skills, 0-100 for scores).")

root = tk.Tk()
root.title("AI-Enhanced Career Guidance System")
root.geometry("800x900")
root.configure(bg="#f0f0f0")

title = tk.Label(root, text="AI Career Guidance System", font=("Helvetica", 20, "bold"), bg="#f0f0f0")
title.pack(pady=20)

frame = ttk.Frame(root)
frame.pack(pady=10)

labels = [
    "Database Fundamentals (0-10)", "Cloud Computing (0-10)", "Programming Skills (0-10)",
    "Hacking Skills (0-10)", "Management Skills (0-10)", "Game Development (0-10)",
    "Interest in AI (0-10)", "Math Score (0-100)", "Science Score (0-100)", "Personality Score (0-10)"
]

entries = []
for i, label_text in enumerate(labels):
    ttk.Label(frame, text=label_text).grid(row=i, column=0, pady=5, sticky="w")
    entry = ttk.Entry(frame)
    entry.grid(row=i, column=1, pady=5)
    entries.append(entry)

entry_db, entry_cloud, entry_prog, entry_hack, entry_mgmt, entry_game, entry_ai, entry_math, entry_science, entry_person = entries

predict_btn = ttk.Button(root, text="Predict Career", command=predict_career)
predict_btn.pack(pady=20)

result_label = tk.Label(root, text="Enter values and click Predict", justify="left", font=("Helvetica", 12), bg="#f0f0f0", wraplength=700)
result_label.pack(pady=20)

root.mainloop()