import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = "#0f0f0f"
SURFACE     = "#1a1a1a"
BORDER      = "#2a2a2a"
ACCENT_WALL = "#6c63ff"   # purple  – wall
ACCENT_ROAD = "#00c896"   # teal    – road
TEXT        = "#f0f0f0"
TEXT_DIM    = "#888888"
WHITE       = "#ffffff"


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crack Detection Launcher")
        self.configure(bg=BG)
        self.resizable(False, False)

        # State
        self.selected_mode = tk.StringVar(value="")   # "wall" | "road"
        self.image_path    = tk.StringVar(value="")

        self._build_ui()
        self._center_window(560, 500)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=30, pady=(30, 8))
        tk.Label(hdr, text="🔍  Crack Detection Pipeline",
                 font=("Segoe UI", 17, "bold"),
                 bg=BG, fg=WHITE).pack(anchor="w")
        tk.Label(hdr, text="Choose a surface type then select an image",
                 font=("Segoe UI", 10),
                 bg=BG, fg=TEXT_DIM).pack(anchor="w")

        # Divider
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=30, pady=(0, 20))

        # ── Step 1: mode cards ────────────────────────────────────────────
        tk.Label(self, text="Step 1 — Select surface type",
                 font=("Segoe UI", 10, "bold"),
                 bg=BG, fg=TEXT_DIM).pack(anchor="w", padx=30)

        cards = tk.Frame(self, bg=BG)
        cards.pack(fill="x", padx=30, pady=(8, 20))
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        self.wall_card = self._make_card(
            cards, col=0,
            icon="🧱", title="Wall Crack", subtitle="Concrete surface analysis",
            accent=ACCENT_WALL, value="wall"
        )
        self.road_card = self._make_card(
            cards, col=1,
            icon="🛣️", title="Road Crack", subtitle="Asphalt surface analysis",
            accent=ACCENT_ROAD, value="road"
        )

        # ── Step 2: image picker ──────────────────────────────────────────
        tk.Label(self, text="Step 2 — Pick an image",
                 font=("Segoe UI", 10, "bold"),
                 bg=BG, fg=TEXT_DIM).pack(anchor="w", padx=30)

        picker_row = tk.Frame(self, bg=BG)
        picker_row.pack(fill="x", padx=30, pady=(8, 24))

        self.path_label = tk.Label(
            picker_row,
            textvariable=self.image_path,
            text="No image selected",
            font=("Segoe UI", 9),
            bg=SURFACE, fg=TEXT_DIM,
            anchor="w", padx=10,
            width=44, relief="flat"
        )
        self.path_label.pack(side="left", ipady=8, fill="x", expand=True)

        browse_btn = tk.Button(
            picker_row, text="Browse…",
            font=("Segoe UI", 9, "bold"),
            bg=BORDER, fg=TEXT,
            activebackground="#333", activeforeground=WHITE,
            relief="flat", cursor="hand2", padx=12,
            command=self._browse
        )
        browse_btn.pack(side="left", padx=(8, 0), ipady=8)

        # ── Run button ────────────────────────────────────────────────────
        self.run_btn = tk.Button(
            self, text="▶  Run Pipeline",
            font=("Segoe UI", 12, "bold"),
            bg=ACCENT_WALL, fg=WHITE,
            activebackground="#857eff", activeforeground=WHITE,
            relief="flat", cursor="hand2",
            command=self._run
        )
        self.run_btn.pack(fill="x", padx=30, ipady=12)

        # Status bar
        self.status = tk.Label(self, text="",
                               font=("Segoe UI", 8),
                               bg=BG, fg=TEXT_DIM)
        self.status.pack(pady=(8, 20))

    def _make_card(self, parent, col, icon, title, subtitle, accent, value):
        """Creates a clickable mode-selection card."""
        frame = tk.Frame(parent, bg=SURFACE, cursor="hand2",
                         highlightbackground=BORDER,
                         highlightthickness=1)
        frame.grid(row=0, column=col,
                   sticky="nsew",
                   padx=(0, 8) if col == 0 else (8, 0),
                   ipady=14)

        tk.Label(frame, text=icon,
                 font=("Segoe UI Emoji", 22),
                 bg=SURFACE, fg=WHITE).pack(pady=(12, 4))
        tk.Label(frame, text=title,
                 font=("Segoe UI", 11, "bold"),
                 bg=SURFACE, fg=WHITE).pack()
        tk.Label(frame, text=subtitle,
                 font=("Segoe UI", 8),
                 bg=SURFACE, fg=TEXT_DIM).pack(pady=(2, 12))

        # Bind click on every child widget
        for widget in [frame] + frame.winfo_children():
            widget.bind("<Button-1>", lambda e, v=value, a=accent, f=frame: self._select_mode(v, a, f))

        return frame

    # ── Interactions ──────────────────────────────────────────────────────────
    def _select_mode(self, value, accent, frame):
        self.selected_mode.set(value)

        # Reset both cards
        for card in (self.wall_card, self.road_card):
            card.configure(highlightbackground=BORDER, highlightthickness=1)
            for w in card.winfo_children():
                w.configure(bg=SURFACE)
            card.configure(bg=SURFACE)

        # Highlight chosen card
        frame.configure(highlightbackground=accent, highlightthickness=2, bg=SURFACE)
        for w in frame.winfo_children():
            w.configure(bg=SURFACE)

        # Update run-button colour to match accent
        self.run_btn.configure(bg=accent,
                               activebackground=accent)
        self._set_status(f"{'Wall' if value == 'wall' else 'Road'} Crack Detector selected.")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files",   "*.*")
            ]
        )
        if path:
            self.image_path.set(path)
            self.path_label.configure(fg=TEXT)
            self._set_status(f"Image: {os.path.basename(path)}")

    def _run(self):
        mode = self.selected_mode.get()
        path = self.image_path.get()

        if not mode:
            messagebox.showwarning("Missing selection", "Please choose Wall or Road first.")
            return
        if not path:
            messagebox.showwarning("No image", "Please select an image file first.")
            return

        if mode == "wall":
            main_path = os.path.join(BASE_DIR, "Wall", "main.py")
        else:
            main_path = os.path.join(BASE_DIR, "Road", "main.py")

        self._set_status("Launching pipeline…")
        self.update()

        subprocess.run([sys.executable, main_path, path])
        self._set_status("Pipeline finished. Ready for another run.")

    def _set_status(self, msg):
        self.status.configure(text=msg)

    def _center_window(self, w, h):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - w) // 2
        y  = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
