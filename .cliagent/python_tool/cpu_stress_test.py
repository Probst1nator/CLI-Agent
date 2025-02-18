import tkinter as tk
from tkinter import ttk
import multiprocessing
import threading
import time
from typing import List, Optional
import psutil

class CPUStressTest(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        
        self.title("CPU Stress Test")
        self.geometry("400x300")
        
        self.stress_threads: List[threading.Thread] = []
        self.running: bool = False
        
        # CPU Count Label
        self.cpu_count: int = multiprocessing.cpu_count()
        tk.Label(self, text=f"Available CPU Cores: {self.cpu_count}").pack(pady=5)
        
        # Thread Count Frame
        thread_frame = ttk.LabelFrame(self, text="Thread Count")
        thread_frame.pack(pady=10, padx=10, fill="x")
        
        self.thread_count = tk.IntVar(value=self.cpu_count)
        self.thread_slider = ttk.Scale(
            thread_frame,
            from_=1,
            to=self.cpu_count * 2,
            variable=self.thread_count,
            orient="horizontal"
        )
        self.thread_slider.pack(pady=5, padx=5, fill="x")
        
        # Load Percentage Frame
        load_frame = ttk.LabelFrame(self, text="Load Percentage")
        load_frame.pack(pady=10, padx=10, fill="x")
        
        self.load_percent = tk.IntVar(value=100)
        self.load_slider = ttk.Scale(
            load_frame,
            from_=1,
            to=100,
            variable=self.load_percent,
            orient="horizontal"
        )
        self.load_slider.pack(pady=5, padx=5, fill="x")
        
        # Control Buttons
        self.start_button = ttk.Button(self, text="Start Test", command=self.start_test)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(self, text="Stop Test", command=self.stop_test, state="disabled")
        self.stop_button.pack(pady=5)
        
        # CPU Usage Display
        self.usage_label = tk.Label(self, text="CPU Usage: 0%")
        self.usage_label.pack(pady=10)
        
        # Start monitoring CPU usage
        self.update_cpu_usage()

    def stress_function(self) -> None:
        while self.running:
            load_time = self.load_percent.get() / 100.0
            if load_time < 1.0:
                end_time = time.time() + load_time
                while time.time() < end_time:
                    pass
                time.sleep(1 - load_time)
            else:
                pass  # Full load, no sleep

    def start_test(self) -> None:
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.thread_slider.config(state="disabled")
        self.load_slider.config(state="disabled")
        
        # Create and start stress threads
        for _ in range(self.thread_count.get()):
            thread = threading.Thread(target=self.stress_function)
            thread.daemon = True
            thread.start()
            self.stress_threads.append(thread)

    def stop_test(self) -> None:
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.thread_slider.config(state="normal")
        self.load_slider.config(state="normal")
        
        # Wait for threads to finish
        for thread in self.stress_threads:
            thread.join(timeout=1.0)
        self.stress_threads.clear()

    def update_cpu_usage(self) -> None:
        cpu_percent = psutil.cpu_percent(interval=1)
        self.usage_label.config(text=f"CPU Usage: {cpu_percent:.1f}%")
        self.after(1000, self.update_cpu_usage)

if __name__ == "__main__":
    app = CPUStressTest()
    app.mainloop()
