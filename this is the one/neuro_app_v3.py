import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class NeuroQoreXApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("NeuroQoreX | Intelligent Digital Neuro-Twin")
        self.geometry("1300x850")

        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- DATA STORAGE ---
        self.patient_data = None
        self.diagnosis_status = "Unknown"

        # --- SIDEBAR ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="NEUROQOREX\nAI ENGINE", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="System: Waiting for Data...", text_color="gray", wraplength=180)
        self.status_label.grid(row=1, column=0, padx=20, pady=20, sticky="n")

        # --- MAIN AREA ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.header = ctk.CTkLabel(self.main_frame, text="🧠 Intelligent Patient Diagnosis", font=ctk.CTkFont(size=26, weight="bold"))
        self.header.pack(anchor="w")
        
        self.subheader = ctk.CTkLabel(self.main_frame, text="Upload clinical CSV data to auto-detect neurological health status.", font=ctk.CTkFont(size=14, slant="italic"))
        self.subheader.pack(anchor="w", pady=(0, 20))

        # INPUT BUTTON
        self.upload_btn = ctk.CTkButton(self.main_frame, text="📂 Upload Patient CSV & Analyze", command=self.load_data, height=45, font=ctk.CTkFont(size=14, weight="bold"))
        self.upload_btn.pack(fill="x", pady=10)

        # PROGRESS BAR
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, height=15)
        self.progress_bar.set(0)
        self.console_log = ctk.CTkLabel(self.main_frame, text="")

        # DASHBOARD CONTAINER
        self.dashboard_frame = ctk.CTkScrollableFrame(self.main_frame, label_text="Live Analysis Output")
        self.dashboard_frame.pack(fill="both", expand=True, pady=10)

    def load_data(self):
        """Opens file dialog and loads CSV"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        
        if file_path:
            try:
                # Read the CSV
                self.patient_data = pd.read_csv(file_path)
                
                # Check for required columns
                required_cols = ['Timepoint', 'Survival_Prob', 'ATP', 'BDNF', 'ROS', 'Inflammation']
                if not all(col in self.patient_data.columns for col in required_cols):
                    messagebox.showerror("Error", f"CSV must contain columns: {', '.join(required_cols)}")
                    return

                # Start Simulation
                self.upload_btn.configure(state="disabled", text="Processing Biomarkers...")
                self.progress_bar.pack(fill="x", pady=10)
                self.console_log.pack(pady=5)
                threading.Thread(target=self.run_analysis).start()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def run_analysis(self):
        """Analyzes data to determine health status"""
        steps = ["Reading Molecular Profile...", "Calculating ATP/ROS Ratio...", "Predicting Neural Survival..."]
        
        for i, step in enumerate(steps):
            self.console_log.configure(text=f"⚙️ {step}")
            time.sleep(0.8) 
            self.progress_bar.set((i + 1) / len(steps))

        # --- DIAGNOSTIC LOGIC ---
        # If Average ATP > 70 AND Average ROS (Stress) < 40 = HEALTHY
        avg_atp = self.patient_data['ATP'].mean()
        avg_ros = self.patient_data['ROS'].mean()

        if avg_atp > 70 and avg_ros < 40:
            self.diagnosis_status = "Healthy"
        else:
            self.diagnosis_status = "Diseased"

        self.after(0, self.finish_analysis)

    def finish_analysis(self):
        self.console_log.configure(text="✅ Diagnosis Complete.")
        self.upload_btn.configure(state="normal", text="📂 Upload Another Patient CSV")
        self.progress_bar.pack_forget()
        self.console_log.pack_forget()
        self.show_dashboard()

    def show_dashboard(self):
        """Renders charts using the ACTUAL uploaded data"""
        
        for widget in self.dashboard_frame.winfo_children():
            widget.destroy()

        # 1. VISUALIZATION ROW
        chart_frame = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
        chart_frame.pack(fill="x", expand=True)

        # -- Chart 1: Trajectory (Actual Data) --
        fig1 = self.create_trajectory_chart()
        canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5)

        # -- Chart 2: Biomarkers (Actual Data) --
        fig2 = self.create_biomarker_chart()
        canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="right", fill="both", expand=True, padx=5)

        # 2. DIAGNOSTIC REPORT
        self.create_report_card()

    def create_trajectory_chart(self):
        """Plots Survival Probability from CSV"""
        timepoints = self.patient_data['Timepoint']
        survival = self.patient_data['Survival_Prob']
        
        # Color logic based on diagnosis
        if self.diagnosis_status == "Healthy":
            line_color = '#00FF00' # Green
            fill_color = 'rgba(0, 255, 0, 0.1)'
        else:
            line_color = '#FF4444' # Red
            fill_color = 'rgba(255, 0, 0, 0.1)'

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')

        ax.plot(timepoints, survival, marker='o', color=line_color, linewidth=2.5, label='Patient Survival')
        
        # Add Healthy Baseline for comparison (Static)
        healthy_baseline = [0.95] * len(timepoints)
        ax.plot(timepoints, healthy_baseline, color='gray', linestyle='--', alpha=0.5, label='Healthy Baseline')

        ax.set_title("Neuron Survival Probability (Actual Data)", color="white", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(facecolor='#2b2b2b', edgecolor='white')
        ax.grid(True, linestyle=':', alpha=0.3)
        return fig

    def create_biomarker_chart(self):
        """Bar chart for average biomarker levels from CSV"""
        # Calculate means from the uploaded data
        avg_atp = self.patient_data['ATP'].mean()
        avg_bdnf = self.patient_data['BDNF'].mean()
        avg_ros = self.patient_data['ROS'].mean()
        avg_inf = self.patient_data['Inflammation'].mean()

        markers = ['ATP', 'BDNF', 'ROS', 'Inflammation']
        values = [avg_atp, avg_bdnf, avg_ros, avg_inf]
        
        # Color coding: High ATP/BDNF is good (Green), High ROS/Inf is bad (Red)
        colors = []
        # ATP Color
        colors.append('#00FF00' if avg_atp > 60 else '#FF4444') 
        # BDNF Color
        colors.append('#00FF00' if avg_bdnf > 60 else '#FF4444')
        # ROS Color
        colors.append('#FF4444' if avg_ros > 50 else 'gray')
        # Inflammation Color
        colors.append('#FF4444' if avg_inf > 50 else 'gray')

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        
        ax.bar(markers, values, color=colors, alpha=0.8)
        ax.set_title("Average Biomarker Levels", color="white", fontsize=10)
        ax.set_ylim(0, 100)
        
        return fig

    def create_report_card(self):
        rec_frame = ctk.CTkFrame(self.dashboard_frame, border_width=2)
        rec_frame.pack(fill="x", pady=20, padx=10)
        
        if self.diagnosis_status == "Diseased":
            rec_frame.configure(border_color="#FF4444", fg_color="#330000")
            title = "⚠ DIAGNOSIS: HIGH RISK DETECTED"
            sub = f"ATP Levels Critical (Avg: {self.patient_data['ATP'].mean():.1f}). High Oxidative Stress."
            therapy = "Recommended: Initiate BDNF-GABA Peptide Protocol immediately."
            color = "#FF5555"
        else:
            rec_frame.configure(border_color="#00FF00", fg_color="#003300")
            title = "✅ DIAGNOSIS: HEALTHY PATIENT"
            sub = f"Bioenergetics Optimal (Avg ATP: {self.patient_data['ATP'].mean():.1f}). Low Inflammation."
            therapy = "Recommended: No intervention required."
            color = "#55FF55"

        ctk.CTkLabel(rec_frame, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=color).pack(pady=(15, 5))
        ctk.CTkLabel(rec_frame, text=sub, font=ctk.CTkFont(size=14), text_color="white").pack(pady=2)
        ctk.CTkLabel(rec_frame, text=therapy, font=ctk.CTkFont(size=14, weight="bold"), text_color="white").pack(pady=(10, 15))

if __name__ == "__main__":
    app = NeuroQoreXApp()
    app.mainloop()