import os
import sys
import time
import json
import logging
import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import List, Dict, Any

from src.core.provider_manager import ProviderManager, ModelInfo
from src.core.model_registry import discover_models
from src.core.session_manager import SessionManager, Session, Message
from src.tools.permissions import get_tool_permission, set_tool_permission, PermissionLevel
from src.tools.tool_registry import global_registry
from src.automation.scheduler import list_jobs, schedule_job, remove_job
from src.research.deep_research_local import run_local_research
from src.research.deep_research_web import run_deep_research
from src.rag.ingest import ingest_file, get_embeddings
from src.rag.vector_store import VectorStore

logger = logging.getLogger("dbert.ui.main_window")

# Try to enable crisp Windows DPI awareness
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        logger.warning(f"Failed to set high-DPI scaling: {e}")

class DBERTApp:
    def __init__(self, root: tk.Tk, config_manager: Any, provider_manager: ProviderManager):
        self.root = root
        self.config_manager = config_manager
        self.provider_manager = provider_manager
        
        # Load directories
        self.session_manager = SessionManager(self.config_manager.app_dir / "history" / "chat.db")
        self.active_session: Session = None
        self.active_model: ModelInfo = None
        self.available_models: List[ModelInfo] = []
        self.planning_mode = tk.BooleanVar(master=self.root, value=True)
        self.voice_mode = tk.BooleanVar(master=self.root, value=False)
        self.mcp_clients = []
        
        self.root.title("DBERT Agent — Local AI Assistant")
        self.root.geometry("1100x700")
        
        # Apply dark theme styling
        self._setup_styles()
        
        # Initialize UI layout elements
        self._build_layout()
        
        # Load initial models and session files
        self._load_models()
        self._load_sessions()
        
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Dark color palette (Catppuccin style)
        self.bg = "#1e1e2e"
        self.sidebar_bg = "#181825"
        self.card_bg = "#313244"
        self.accent = "#89b4fa"
        self.text_fg = "#cdd6f4"
        self.text_dim = "#a6adc8"
        self.border_color = "#45475a"
        
        self.root.configure(bg=self.bg)
        
        # Configure frames and elements
        style.configure("TFrame", background=self.bg)
        style.configure("Sidebar.TFrame", background=self.sidebar_bg)
        style.configure("Card.TFrame", background=self.card_bg)
        
        # Labels
        style.configure("TLabel", background=self.bg, foreground=self.text_fg, font=("Segoe UI", 10))
        style.configure("Sidebar.TLabel", background=self.sidebar_bg, foreground=self.text_fg, font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", background=self.bg, foreground=self.text_fg, font=("Segoe UI", 14, "bold"))
        style.configure("Accent.TLabel", background=self.bg, foreground=self.accent, font=("Segoe UI", 10, "bold"))
        
        # Buttons
        style.configure("TButton", background=self.card_bg, foreground=self.text_fg, bordercolor=self.border_color, font=("Segoe UI", 9))
        style.map("TButton", background=[("active", self.accent), ("pressed", self.accent)], foreground=[("active", "#11111b")])
        style.configure("Accent.TButton", background=self.accent, foreground="#11111b", bordercolor=self.border_color, font=("Segoe UI", 9, "bold"))
        
        # Notebooks / Tabs
        style.configure("TNotebook", background=self.bg, bordercolor=self.border_color)
        style.configure("TNotebook.Tab", background=self.sidebar_bg, foreground=self.text_dim, padding=(10, 4), bordercolor=self.border_color)
        style.map("TNotebook.Tab", background=[("selected", self.bg)], foreground=[("selected", self.text_fg)])
        
        # Entry
        style.configure("TEntry", fieldbackground=self.card_bg, foreground=self.text_fg, bordercolor=self.border_color)

    def _build_layout(self):
        # 1. Top bar controls
        top_bar = ttk.Frame(self.root, height=45)
        top_bar.pack(side="top", fill="x", padx=10, pady=5)
        
        logo = ttk.Label(top_bar, text="DBERT Agent", style="Title.TLabel")
        logo.pack(side="left", padx=5)
        
        # Active Model display
        self.model_label = ttk.Label(top_bar, text="Model: Loading...", style="Accent.TLabel")
        self.model_label.pack(side="left", padx=20)
        
        # Model switcher dropdown
        self.model_combo = ttk.Combobox(top_bar, width=30, state="readonly")
        self.model_combo.pack(side="left", padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        
        # Refresh models button
        refresh_btn = ttk.Button(top_bar, text="↻ Refresh", command=self._load_models, width=10)
        refresh_btn.pack(side="left", padx=5)
        

        # 2. Main Paned layout split
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        
        # Left Pane: Agent Manager
        self.sidebar = ttk.Frame(paned, style="Sidebar.TFrame", width=250)
        paned.add(self.sidebar, weight=1)
        
        # Right Pane: Workspace Notebook (Tabs)
        self.workspace = ttk.Notebook(paned)
        paned.add(self.workspace, weight=4)
        
        # Build Sidebar Panels
        self._build_sidebar()
        
        # Build Workspace Tabs
        self._build_chat_tab()
        self._build_files_tab()
        self._build_research_tab()
        self._build_permissions_tab()
        
    def _build_sidebar(self):
        # Session manager section
        lbl = ttk.Label(self.sidebar, text="Resumable Sessions", style="Sidebar.TLabel")
        lbl.pack(anchor="w", padx=10, pady=10)
        
        # Sessions listbox
        self.session_listbox = tk.Listbox(self.sidebar, bg=self.sidebar_bg, fg=self.text_fg, selectbackground=self.accent, selectforeground="#11111b", bd=0, highlightthickness=0)
        self.session_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.session_listbox.bind("<<ListboxSelect>>", self._on_session_selected)
        
        new_sess_btn = ttk.Button(self.sidebar, text="+ New Session", command=self._create_new_session)
        new_sess_btn.pack(fill="x", padx=10, pady=5)
        
        # Active background monitor tasks panel
        mon_lbl = ttk.Label(self.sidebar, text="Background Monitors", style="Sidebar.TLabel")
        mon_lbl.pack(anchor="w", padx=10, pady=10)
        
        self.monitor_listbox = tk.Listbox(self.sidebar, height=6, bg=self.sidebar_bg, fg=self.text_dim, bd=0, highlightthickness=0)
        self.monitor_listbox.pack(fill="x", padx=5, pady=5)
        self._refresh_monitors()
        
        # Quick Settings
        set_lbl = ttk.Label(self.sidebar, text="Quick Settings", style="Sidebar.TLabel")
        set_lbl.pack(anchor="w", padx=10, pady=(15, 5))
        
        voice_btn = ttk.Checkbutton(self.sidebar, text="Voice HUD", variable=self.voice_mode, command=self._toggle_voice_hud, style="TCheckbutton")
        voice_btn.pack(anchor="w", padx=15, pady=2)
        
        plan_btn = ttk.Checkbutton(self.sidebar, text="Planning Mode", variable=self.planning_mode, style="TCheckbutton")
        plan_btn.pack(anchor="w", padx=15, pady=2)

    def _build_chat_tab(self):
        chat_frame = ttk.Frame(self.workspace)
        self.workspace.add(chat_frame, text="Chat Session")
        
        # Chat log display
        scroll_frame = ttk.Frame(chat_frame)
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.chat_display = tk.Text(scroll_frame, bg=self.bg, fg=self.text_fg, state="disabled", wrap="word", relief="flat", font=("Segoe UI", 10))
        chat_scroll = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.chat_display.yview)
        self.chat_display.configure(yscrollcommand=chat_scroll.set)
        
        chat_scroll.pack(side="right", fill="y")
        self.chat_display.pack(side="left", fill="both", expand=True)
        
        # Prompt Entry Row
        entry_row = ttk.Frame(chat_frame)
        entry_row.pack(side="bottom", fill="x", padx=5, pady=5)
        
        self.prompt_entry = tk.Text(entry_row, height=3, bg=self.card_bg, fg=self.text_fg, insertbackground=self.text_fg, relief="flat", font=("Segoe UI", 10))
        self.prompt_entry.pack(side="left", fill="x", expand=True, pady=4, padx=(0, 5))
        
        def on_enter(e):
            self._send_user_message()
            return 'break'
            
        def on_shift_enter(e):
            self.prompt_entry.insert(tk.INSERT, '\n')
            return 'break'
            
        self.prompt_entry.bind("<Return>", on_enter)
        self.prompt_entry.bind("<Shift-Return>", on_shift_enter)
        
        send_btn = ttk.Button(entry_row, text="Send", style="Accent.TButton", command=self._send_user_message)
        send_btn.pack(side="right")

    def _build_files_tab(self):
        files_frame = ttk.Frame(self.workspace)
        self.workspace.add(files_frame, text="File Ingestion (RAG)")
        
        ctrl_row = ttk.Frame(files_frame)
        ctrl_row.pack(fill="x", padx=10, pady=10)
        
        select_btn = ttk.Button(ctrl_row, text="Select File to Ingest", command=self._ingest_file_dialog)
        select_btn.pack(side="left", padx=5)
        
        self.ingest_status = ttk.Label(ctrl_row, text="Idle", style="Accent.TLabel")
        self.ingest_status.pack(side="left", padx=10)
        
        self.ingest_progress = ttk.Progressbar(ctrl_row, mode="indeterminate", length=150)
        self.ingest_progress.pack(side="left", padx=10)
        
        # Workspace Files List
        list_lbl = ttk.Label(files_frame, text="Ingested Workspace Files:")
        list_lbl.pack(anchor="w", padx=10, pady=5)
        
        self.files_listbox = tk.Listbox(files_frame, bg=self.card_bg, fg=self.text_fg, bd=0, highlightthickness=0)
        self.files_listbox.pack(fill="both", expand=True, padx=10, pady=10)
        self._refresh_files()

    def _build_research_tab(self):
        res_frame = ttk.Frame(self.workspace)
        self.workspace.add(res_frame, text="Deep Research")
        
        lbl = ttk.Label(res_frame, text="Topic to Research:")
        lbl.pack(anchor="w", padx=10, pady=5)
        
        self.research_entry = tk.Entry(res_frame, bg=self.card_bg, fg=self.text_fg, insertbackground=self.text_fg, relief="flat", font=("Segoe UI", 10))
        self.research_entry.pack(fill="x", padx=10, pady=5)
        
        opt_row = ttk.Frame(res_frame)
        opt_row.pack(fill="x", padx=10, pady=5)
        
        self.research_scope = tk.StringVar(master=self.root, value="web")
        r_web = ttk.Radiobutton(opt_row, text="Web (DuckDuckGo)", variable=self.research_scope, value="web")
        r_web.pack(side="left", padx=5)
        r_loc = ttk.Radiobutton(opt_row, text="Local (Vector index)", variable=self.research_scope, value="local")
        r_loc.pack(side="left", padx=10)
        
        run_btn = ttk.Button(res_frame, text="Run Deep Research", command=self._run_deep_research_async)
        run_btn.pack(anchor="w", padx=10, pady=10)
        
        self.research_status = ttk.Label(res_frame, text="", style="Accent.TLabel")
        self.research_status.pack(anchor="w", padx=10, pady=5)
        
        self.research_progress = ttk.Progressbar(res_frame, mode="indeterminate", length=200)
        self.research_progress.pack(anchor="w", padx=10, pady=5)
        
        self.research_report_view = tk.Text(res_frame, bg=self.bg, fg=self.text_fg, state="disabled", wrap="word", font=("Segoe UI", 9))
        self.research_report_view.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_permissions_tab(self):
        perm_frame = ttk.Frame(self.workspace)
        self.workspace.add(perm_frame, text="Tool Permissions")
        
        lbl = ttk.Label(perm_frame, text="Configure Workspace Action Permission Levels:")
        lbl.pack(anchor="w", padx=10, pady=10)
        
        # Tool Permission Matrix table list
        self.perm_tree = ttk.Treeview(perm_frame, columns=("Tool", "Permission Level"), show="headings")
        self.perm_tree.heading("Tool", text="Tool/Capability")
        self.perm_tree.heading("Permission Level", text="Access Level")
        self.perm_tree.pack(fill="both", expand=True, padx=10, pady=5)
        
        btn_row = ttk.Frame(perm_frame)
        btn_row.pack(fill="x", padx=10, pady=10)
        
        allow_btn = ttk.Button(btn_row, text="Allow Always", command=lambda: self._set_selected_perm(PermissionLevel.ALWAYS_ALLOW))
        allow_btn.pack(side="left", padx=5)
        
        ask_btn = ttk.Button(btn_row, text="Ask Every Time", command=lambda: self._set_selected_perm(PermissionLevel.ASK_EVERY_TIME))
        ask_btn.pack(side="left", padx=5)
        
        deny_btn = ttk.Button(btn_row, text="Deny / Off", command=lambda: self._set_selected_perm(PermissionLevel.OFF))
        deny_btn.pack(side="left", padx=5)
        
        self._refresh_permissions()

    def _setup_voice_hud_window(self):
        self.hud_win = tk.Toplevel(self.root)
        self.hud_win.title("Voice HUD")
        self.hud_win.geometry("250x150")
        self.hud_win.resizable(False, False)
        self.hud_win.configure(bg="#11111b")
        
        lbl = ttk.Label(self.hud_win, text="DBERT Voice Active", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=10)
        
        self.waveform_lbl = ttk.Label(self.hud_win, text="[ Listening ... ]", foreground=self.accent)
        self.waveform_lbl.pack(pady=10)
        
        self.voice_text_lbl = ttk.Label(self.hud_win, text="Speak naturally.", wraplength=200, foreground=self.text_fg)
        self.voice_text_lbl.pack(pady=5)
        
        # Override close button behavior
        self.hud_win.protocol("WM_DELETE_WINDOW", self._close_voice_hud)

    # ----------------- Actions & Logic -----------------

    def _load_models(self):
        """Discovers available models and populates picker."""
        try:
            self.available_models = discover_models(self.provider_manager)
        except Exception as e:
            logger.error(f"Error loading models for GUI: {e}")
            self.available_models = []
            
        model_names = [m.id for m in self.available_models]
        if not model_names:
            model_names = ["No models detected"]
            
        self.model_combo["values"] = model_names
        
        # Default preference selection
        pref = self.config_manager.config.get("model_preference")
        matched = [m for m in self.available_models if m.id == pref]
        if matched:
            self.active_model = matched[0]
        elif self.available_models:
            self.active_model = self.available_models[0]
        else:
            self.active_model = ModelInfo(id="offline-setup", provider="setup", is_local=True)
            
        self.model_label.configure(text=f"Active: {self.active_model.id}")
        self.model_combo.set(self.active_model.id)

    def _on_model_selected(self, event=None):
        m_id = self.model_combo.get()
        matched = [m for m in self.available_models if m.id == m_id]
        if matched:
            self.active_model = matched[0]
            self.config_manager.config["model_preference"] = self.active_model.id
            self.config_manager.save_config()
            self.model_label.configure(text=f"Active: {self.active_model.id}")

    def _load_sessions(self):
        self.session_listbox.delete(0, tk.END)
        self.sessions = self.session_manager.list_sessions()
        for s in self.sessions:
            self.session_listbox.insert(tk.END, s.name)
            
        if self.sessions:
            self.session_listbox.selection_set(0)
            self._select_session(self.sessions[0])
        else:
            self._create_new_session()

    def _on_session_selected(self, event=None):
        sel = self.session_listbox.curselection()
        if sel:
            idx = sel[0]
            self._select_session(self.sessions[idx])

    def _select_session(self, session_summary):
        self.active_session = self.session_manager.resume_session(session_summary.id)
        self._refresh_chat_display()
        self._refresh_files()
        self._refresh_permissions()

    def _create_new_session(self):
        new_sess = self.session_manager.create_session("default_workspace")
        self._load_sessions()

    def _refresh_chat_display(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        
        # Configure markdown tags
        self.chat_display.tag_configure("user_msg", background=self.card_bg, foreground=self.text_fg, lmargin1=40, lmargin2=40, rmargin=10, spacing1=10, spacing3=10)
        self.chat_display.tag_configure("ai_msg", background=self.bg, foreground=self.text_fg, lmargin1=10, lmargin2=10, rmargin=40, spacing1=10, spacing3=10)
        self.chat_display.tag_configure("bold", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_configure("code", background="#11111b", foreground="#a6adc8", font=("Consolas", 10))
        
        if self.active_session and self.active_session.messages:
            import re
            for msg in self.active_session.messages:
                tag = "user_msg" if msg.role == "user" else "ai_msg"
                prefix = "You:\n" if msg.role == "user" else "DBERT:\n"
                
                self.chat_display.insert(tk.END, prefix, (tag, "bold"))
                
                parts = re.split(r'(\*\*.*?\*\*|`.*?`)', msg.content)
                for p in parts:
                    if p.startswith('**') and p.endswith('**'):
                        self.chat_display.insert(tk.END, p[2:-2], (tag, "bold"))
                    elif p.startswith('`') and p.endswith('`'):
                        self.chat_display.insert(tk.END, p[1:-1], (tag, "code"))
                    else:
                        self.chat_display.insert(tk.END, p, tag)
                self.chat_display.insert(tk.END, "\n", tag)
                
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

    def _send_user_message(self):
        if getattr(self, '_is_inferencing', False):
            return
            
        user_text = self.prompt_entry.get("1.0", tk.END).strip()
        if not user_text:
            return
            
        self.prompt_entry.delete("1.0", tk.END)
        self.prompt_entry.configure(state="disabled")
        self._is_inferencing = True
        
        # Save message in session DB
        self.session_manager.append_message(self.active_session.id, "user", user_text)
        
        # Update local UI state
        from src.core.session_manager import Message
        self.active_session.messages.append(Message(role="user", content=user_text))
        
        # Add thinking placeholder
        self.active_session.messages.append(Message(role="assistant", content="*Thinking...*"))
        
        self._refresh_chat_display()
        
        # Call LLM generation asynchronously in a background thread to prevent UI freezing
        threading.Thread(target=self._run_llm_inference, args=(user_text,), daemon=True).start()

    def _run_llm_inference(self, user_text: str):
        from src.main import execute_completion_with_fallback
        
        # Ground context retrieval block (PDF + History)
        grounded_input = user_text
        try:
            # Check for similarities over RAG document chunks
            vs = VectorStore(self.active_session.workspace_id, app_dir=self.config_manager.app_dir)
            emb = get_embeddings([user_text], self.active_model, self.provider_manager)[0]
            matches = vs.query(emb, top_k=2)
            valid_matches = [m for m in matches if m[3] > 0.35]
            
            if valid_matches:
                grounded_input = "[CONTEXT FROM LOCAL DOCUMENTS]\n"
                for doc_path, chunk_text, meta, score in valid_matches:
                    grounded_input += f"Source: {Path(doc_path).name}\nContent: {chunk_text}\n---\n"
                grounded_input += f"\nUser Query: {user_text}"
        except Exception as e:
            err_msg = str(e)
            logger.error(f"RAG Context retrieval failed: {err_msg}")
            self.root.after(0, lambda m=err_msg: messagebox.showwarning("RAG Error", f"Failed to retrieve local documents context:\n{m}"))
            
        litellm_messages = [
            {"role": "system", "content": "You are DBERT, a local-first privacy-first AI desktop assistant."},
            {"role": "user", "content": grounded_input}
        ]
        
        # Override standard confirm prompt behavior inside gui context
        # We hook a custom permissions modal callback!
        def gui_permissions_check(tool_name: str, arguments: Dict[str, Any]) -> str:
            # We must run this callback in the main thread since GUI widgets are not thread-safe!
            result_container = []
            
            def show_matrix():
                choice = self._prompt_permission_matrix_dialog(tool_name, arguments)
                result_container.append(choice)
                
            self.root.after(0, show_matrix)
            
            # Block/wait until user registers their selection in container
            while not result_container:
                time.sleep(0.1)
                
            return result_container[0]
            
        try:
            # Update GUI state to show thinking
            self.root.after(0, lambda: self.root.configure(cursor="watch"))
            
            reply, _ = execute_completion_with_fallback(
                self.active_model,
                litellm_messages,
                self.provider_manager,
                self.config_manager,
                workspace_id=self.active_session.workspace_id,
                permission_callback=gui_permissions_check
            )
            
            # Remove thinking placeholder and update UI safely on main thread
            def _apply_reply(r):
                if self.active_session.messages and self.active_session.messages[-1].content == "*Thinking...*":
                    self.active_session.messages.pop()
                from src.core.session_manager import Message
                self.active_session.messages.append(Message(role="assistant", content=r))
                self._refresh_chat_display()
                
            # Save assistant reply to the database in the background thread
            self.session_manager.append_message(self.active_session.id, "assistant", reply)
            self.root.after(0, lambda r=reply: _apply_reply(r))
            
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Inference run failed: {err_msg}")
            
            def _handle_err(m):
                if self.active_session.messages and self.active_session.messages[-1].content == "*Thinking...*":
                    self.active_session.messages.pop()
                self._refresh_chat_display()
                messagebox.showerror("Inference Error", f"Failed to complete prompt turn: {m}")
                
            self.root.after(0, lambda m=err_msg: _handle_err(m))
        finally:
            def _restore_ui():
                self.root.configure(cursor="")
                self.prompt_entry.configure(state="normal")
                self._is_inferencing = False
            self.root.after(0, _restore_ui)

    def _prompt_permission_matrix_dialog(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Modal dialog prompt asking to allow/deny tool executions."""
        win = tk.Toplevel(self.root)
        win.title("Permission Request")
        win.geometry("400x200")
        win.transient(self.root)
        win.grab_set()
        
        lbl = ttk.Label(win, text=f"Tool call requested: {tool_name}", font=("Segoe UI", 10, "bold"))
        lbl.pack(pady=10)
        
        args_lbl = ttk.Label(win, text=f"Arguments: {arguments}", wraplength=350)
        args_lbl.pack(pady=5)
        
        choice = tk.StringVar(master=win, value="y")
        
        def save_and_close(val):
            choice.set(val)
            win.destroy()
            
        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=15)
        
        ttk.Button(btn_frame, text="Allow once", command=lambda: save_and_close("y")).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Always Allow", command=lambda: save_and_close("always")).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Deny once", command=lambda: save_and_close("n")).pack(side="left", padx=5)
        
        win.wait_window()
        return choice.get()

    def _ingest_file_dialog(self):
        f_path = filedialog.askopenfilename(filetypes=[("PDF/Text Files", "*.pdf *.txt")])
        if f_path:
            self.ingest_status.configure(text="Ingesting...")
            self.ingest_progress.start()
            threading.Thread(target=self._run_ingestion, args=(f_path,), daemon=True).start()

    def _run_ingestion(self, f_path: str):
        try:
            res = ingest_file(
                f_path,
                self.active_session.workspace_id,
                self.provider_manager,
                self.active_model,
                app_dir=self.config_manager.app_dir
            )
            if res["success"]:
                self.root.after(0, lambda: messagebox.showinfo("Ingestion Successful", f"Indexed {res['chunks_count']} chunks!"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Ingestion Failed", res.get("error", "Unknown error.")))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda m=err_msg: messagebox.showerror("Ingestion Error", f"Error: {m}"))
        finally:
            self.root.after(0, self.ingest_progress.stop)
            self.root.after(0, lambda: self.ingest_status.configure(text="Idle"))
            self.root.after(0, self._refresh_files)

    def _refresh_files(self):
        self.files_listbox.delete(0, tk.END)
        if not self.active_session:
            return
        # Query VectorStore for ingested files
        try:
            vs = VectorStore(self.active_session.workspace_id, app_dir=self.config_manager.app_dir)
            files = vs.get_ingested_files()
            for f in files:
                self.files_listbox.insert(tk.END, Path(f).name)
        except Exception as e:
            logger.error(f"Failed to refresh ingested files: {e}")

    def _run_deep_research_async(self):
        q = self.research_entry.get().strip()
        if not q:
            return
            
        self.research_status.configure(text="Deep Research running (this may take up to 2 minutes)...")
        self.research_progress.start()
        threading.Thread(target=self._run_research_logic, args=(q,), daemon=True).start()

    def _run_research_logic(self, q: str):
        scope = self.research_scope.get()
        try:
            if scope == "web":
                report = run_deep_research(
                    q,
                    self.active_session.workspace_id,
                    self.active_model,
                    self.provider_manager,
                    self.config_manager
                )
            else:
                report = run_local_research(
                    q,
                    self.active_session.workspace_id,
                    self.active_model,
                    self.provider_manager,
                    self.config_manager
                )
                
            def render_report(rep):
                self.research_report_view.configure(state="normal")
                self.research_report_view.delete("1.0", tk.END)
                self.research_report_view.insert(tk.END, rep)
                self.research_report_view.configure(state="disabled")
                self.research_progress.stop()
                self.research_status.configure(text="Completed.")
                
            self.root.after(0, lambda: render_report(report))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, self.research_progress.stop)
            self.root.after(0, lambda m=err_msg: self.research_status.configure(text=f"Failed: {m}"))

    def _refresh_permissions(self):
        # Clear table
        for item in self.perm_tree.get_children():
            self.perm_tree.delete(item)
            
        if not self.active_session:
            return
            
        # Insert permissions for pre-registered tools
        tools = ["web_search", "read_file", "write_file", "run_shell"]
        for t in tools:
            level = get_tool_permission(t, self.active_session.workspace_id, self.config_manager)
            self.perm_tree.insert("", "end", values=(t, level.value))

    def _set_selected_perm(self, level: PermissionLevel):
        sel = self.perm_tree.selection()
        if sel:
            item = self.perm_tree.item(sel[0])
            tool_name = item["values"][0]
            set_tool_permission(tool_name, level, self.active_session.workspace_id, self.config_manager)
            self._refresh_permissions()

    def _refresh_monitors(self):
        self.monitor_listbox.delete(0, tk.END)
        jobs = list_jobs(self.config_manager.app_dir)
        for j in jobs:
            self.monitor_listbox.insert(tk.END, f"{j['name']} ({j['type']}) — {j['status']}")

    def _toggle_voice_hud(self):
        if self.voice_mode.get():
            self._setup_voice_hud_window()
        else:
            self._close_voice_hud()

    def _close_voice_hud(self):
        if hasattr(self, "hud_win") and self.hud_win.winfo_exists():
            self.hud_win.destroy()
        self.voice_mode.set(False)

def start_gui(config_manager, provider_manager):
    root = tk.Tk()
    app = DBERTApp(root, config_manager, provider_manager)
    root.mainloop()
