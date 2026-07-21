import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure src directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_manager import ConfigManager
from src.core.hardware_profiler import HardwareProfiler
from src.core.provider_manager import ProviderManager, ModelInfo
from src.core.model_registry import discover_models, resolve_model_choice, NoModelsAvailableError
from src.core.session_manager import SessionManager, SessionFilter, Message

# Setup rich console
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.live import Live
except ImportError:
    # Basic print fallback if rich is not fully initialized yet (though requirements.txt installs it)
    class FallbackText:
        def __init__(self, t, *args, **kwargs): self.t = t
        def __str__(self): return self.t
    class FallbackPanel:
        def __init__(self, text, *args, **kwargs): self.text = text
        def __str__(self): return f"=== {self.text} ==="
    class FallbackConsole:
        def print(self, msg, *args, **kwargs): print(msg)
    Console = FallbackConsole
    Panel = FallbackPanel
    Text = FallbackText

console = Console()
logger = logging.getLogger("dbert.main")

def setup_logging(app_dir: Path) -> None:
    log_dir = app_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "dbert.log", encoding="utf-8")
        ]
    )

def scheduler_daemon_loop(config_manager, provider_manager, active_model_ref):
    import time
    from src.automation.scheduler import run_pending_jobs
    while True:
        try:
            current_active_model = active_model_ref[0]
            if current_active_model.id != "offline-setup":
                run_pending_jobs(config_manager, provider_manager, current_active_model)
        except Exception as e:
            logger.error(f"Error in scheduler daemon loop: {e}")
        time.sleep(10)

def print_help() -> None:
    console.print("\n[bold cyan]Available Slash Commands:[/]")
    console.print("  [bold yellow]/model[/]             - Switch active model")
    console.print("  [bold yellow]/permissions[/]       - View current tool permissions level")
    console.print("  [bold yellow]/provider[/]          - List registered API providers and status")
    console.print("  [bold yellow]/provider add[/]      - Add a cloud provider (e.g. `/provider add openai sk-...`)")
    console.print("  [bold yellow]/provider remove[/]   - Remove a provider (e.g. `/provider remove openai`)")
    console.print("  [bold yellow]/fallback[/]          - View global fallback chain order")
    console.print("  [bold yellow]/fallback set[/]      - Set fallback order (e.g. `/fallback set lmstudio-local,openai`)")
    console.print("  [bold yellow]/ingest[/]            - Ingest a local PDF or text document (e.g. `/ingest doc.pdf`)")
    console.print("  [bold yellow]/rag[/]               - Direct query local vector store chunks")
    console.print("  [bold yellow]/voice[/]             - Enter hands-free voice dialogue loop")
    console.print("  [bold yellow]/research[/]          - Deep research query (e.g. `/research web trading strategies`)")
    console.print("  [bold yellow]/mcp add[/]           - Connect a local stdio MCP server (e.g. `/mcp add npx ...`)")
    console.print("  [bold yellow]/schedule list[/]     - List all background scheduled jobs")
    console.print("  [bold yellow]/schedule add[/]      - Add a job (e.g. `/schedule add url name url interval`)")
    console.print("  [bold yellow]/schedule remove[/]   - Remove a job (e.g. `/schedule remove id`)")
    console.print("  [bold yellow]/schedule run[/]      - Run all pending jobs immediately")
    console.print("  [bold yellow]/help[/]              - Show this help menu")
    console.print("  [bold yellow]/exit[/]              - Terminate the session and exit")
    console.print("")

def _call_litellm_completion(
    model_info: ModelInfo,
    messages: List[dict],
    provider_manager: ProviderManager,
    tools: List[dict] = None
) -> Any:
    p_info = provider_manager.active_providers.get(model_info.provider, {})
    
    kwargs = {
        "messages": messages,
        "timeout": 15.0
    }
    if tools:
        kwargs["tools"] = tools
        
    if model_info.is_local:
        kwargs["model"] = f"openai/{model_info.id}"
        kwargs["api_base"] = p_info.get("base_url")
        kwargs["api_key"] = p_info.get("api_key", "lm-studio")
    else:
        kwargs["model"] = model_info.id
        kwargs["api_key"] = p_info.get("api_key")
        
    kwargs["timeout"] = 180
        
    import litellm
    return litellm.completion(**kwargs)

def execute_completion_with_fallback(
    active_model: ModelInfo,
    messages: List[dict],
    provider_manager: ProviderManager,
    config_manager: ConfigManager,
    workspace_id: str = "default",
    permission_callback = None
) -> tuple[str, ModelInfo]:
    """
    Executes a completion request using the active model first, supporting tool calls,
    user permissions dialogue, and fallback providers.
    """
    from src.tools.tool_registry import global_registry
    from src.tools.permissions import get_tool_permission, set_tool_permission, PermissionLevel
    
    tools = global_registry.get_available_tools()
    current_model = active_model
    tool_turns = 0
    
    def get_completion_with_fallback(msg_payload, model_choice):
        try:
            return _call_litellm_completion(model_choice, msg_payload, provider_manager, tools), model_choice
        except Exception as e:
            logger.warning(f"Model {model_choice.id} failed: {e}. Cascading fallback...")
            console.print(f"[bold yellow][!] Model failed:[/] [red]{e}[/]")
            
            fallback_order = config_manager.config.get("default_provider_order", ["lmstudio-local", "openai", "anthropic", "gemini"])
            try:
                available_models = discover_models(provider_manager)
            except Exception:
                available_models = []
                
            for provider in fallback_order:
                if provider == model_choice.provider:
                    continue
                if provider not in provider_manager.active_providers:
                    continue
                provider_models = [m for m in available_models if m.provider == provider]
                if not provider_models:
                    continue
                fallback_model = provider_models[0]
                console.print(f"[bold yellow][!] Attempting fallback to provider {provider} via model {fallback_model.id}...[/]")
                try:
                    return _call_litellm_completion(fallback_model, msg_payload, provider_manager, tools), fallback_model
                except Exception as ex:
                    console.print(f"[bold red]Fallback to {fallback_model.id} failed: {ex}[/]")
            raise Exception("All models in the fallback chain failed.")

    while tool_turns < 5:
        response, answered_model = get_completion_with_fallback(messages, current_model)
        message = response.choices[0].message
        
        # Check if the model requested any tool calls
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return message.content or "", answered_model
            
        # Append assistant tool calls request to messages
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ]
        })
        
        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc.function.name
            
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except Exception as e:
                logger.error(f"Failed to parse tool call arguments: {e}")
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": tc.id,
                    "content": f"Error: Failed to parse arguments as JSON: {e}"
                })
                continue
                
            # Permissions validation check
            permission = get_tool_permission(tool_name, workspace_id, config_manager)
            
            allowed = False
            if permission == PermissionLevel.ALWAYS_ALLOW:
                allowed = True
            elif permission == PermissionLevel.OFF:
                allowed = False
            else:
                if permission_callback:
                    choice = permission_callback(tool_name, args)
                else:
                    # Dynamic User CLI Confirmation
                    console.print(Panel(
                        f"Tool: [cyan]{tool_name}[/]\nArguments: [cyan]{args}[/]",
                        title="Tool Permission Request",
                        border_style="yellow"
                    ))
                    choice = Prompt.ask(
                        "Allow tool execution?",
                        choices=["y", "n", "always", "deny"],
                        default="y"
                    )
                    
                if choice == "y":
                    allowed = True
                elif choice == "always":
                    allowed = True
                    set_tool_permission(tool_name, PermissionLevel.ALWAYS_ALLOW, workspace_id, config_manager)
                elif choice == "deny":
                    allowed = False
                    set_tool_permission(tool_name, PermissionLevel.OFF, workspace_id, config_manager)
                else:
                    allowed = False
                    
            if allowed:
                console.print(f"[dim cyan]Executing tool {tool_name}...[/]")
                result = global_registry.execute_tool(tool_name, args)
            else:
                console.print(f"[yellow]Execution of tool {tool_name} was denied.[/]")
                result = f"Error: Execution of tool {tool_name} was denied by the user."
                
            messages.append({
                "role": "tool",
                "name": tool_name,
                "tool_call_id": tc.id,
                "content": result
            })
            
        tool_turns += 1
        
    return "Error: Maximum agentic tool execution turns reached.", answered_model

def main():
    # 1. Initialize Directories & Config
    config_manager = ConfigManager()
    setup_logging(config_manager.app_dir)
    logger.info("Starting DBERT Agent...")
    active_mcp_clients = []

    # 2. Hardware profiling
    console.print("[cyan]Profiling host hardware capabilities...[/]")
    profiler = HardwareProfiler(config_manager.app_dir)
    hw, display, defaults = profiler.run_profiling()
    
    logger.info(f"Hardware profile: {hw}")
    logger.info(f"Display profile: {display}")
    logger.info(f"Adaptive defaults: {defaults}")

    # Show welcome banner
    console.print(Panel(
        f"[bold green]DBERT Agent — Startup Successful[/]\n"
        f"OS: {hw.os_version} | CPU: {hw.cpu_cores} Cores | RAM: {hw.ram_gb} GB\n"
        f"GPU: {hw.gpu_vendor} ({hw.gpu_vram_gb} GB VRAM) | DPI: {display.dpi_scale}x\n"
        f"Logs: {config_manager.app_dir}/logs/dbert.log",
        title="[bold white]Google Antigravity & Codex-style AI Assistant[/]",
        border_style="green"
    ))

    # 3. Provider & Model Setup
    provider_manager = ProviderManager(config_manager)
    provider_manager.register_local_provider()
    provider_manager.register_cloud_providers()

    try:
        available_models = discover_models(provider_manager)
    except Exception as e:
        logger.error(f"Error discovering models: {e}")
        available_models = []

    # Get saved preference from config
    saved_model_pref = config_manager.config.get("model_preference")
    
    # Custom console picker callback for multiple models
    def console_picker(models_list: List[ModelInfo]) -> ModelInfo:
        # Group models
        local_m = [m for m in models_list if m.is_local]
        cloud_m = [m for m in models_list if not m.is_local]
        sorted_m = local_m + cloud_m
        
        table = Table(title="Available LLM Models", border_style="cyan")
        table.add_column("No.", style="cyan", no_wrap=True)
        table.add_column("Model ID", style="green")
        table.add_column("Provider", style="yellow")
        table.add_column("Type", style="magenta")
        
        for idx, m in enumerate(sorted_m, 1):
            type_str = "Local (On-device)" if m.is_local else "Cloud API"
            table.add_row(str(idx), m.id, m.provider, type_str)
            
        console.print(table)
        
        while True:
            choice = Prompt.ask(f"Select a model to activate (1-{len(sorted_m)})")
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(sorted_m):
                    selected = sorted_m[choice_idx]
                    return selected
            except ValueError:
                pass
            console.print("[red]Invalid selection. Try again.[/]")

    active_model = None
    try:
        active_model = resolve_model_choice(
            available_models,
            saved_pref=saved_model_pref,
            prompt_callback=console_picker
        )
    except NoModelsAvailableError as e:
        console.print(Panel(
            f"[bold red]Error:[/] {e}\n\n"
            f"[bold yellow]LM Studio Guide:[/]\n"
            f"1. Open LM Studio.\n"
            f"2. Load/Run any instruction model (e.g. Qwen2.5-0.5B-Instruct).\n"
            f"3. Start the local server at port 1234.\n"
            f"4. Re-launch DBERT.\n\n"
            f"[bold cyan]Cloud Option:[/] Use `/provider add <name> <api_key>` in DBERT to register cloud models.",
            title="Model Registry Resolution Failure",
            border_style="red"
        ))
        logger.error("Startup halted: No models available.")
        
        # If no models, we will start a dummy fallback command loop for cloud credentials management
        console.print("[yellow]Entering offline setup console...[/]")
        active_model = ModelInfo(id="offline-setup", provider="setup", is_local=True)

    if active_model.id != "offline-setup":
        # Save model choice back to config
        config_manager.config["model_preference"] = active_model.id
        config_manager.save_config()
        console.print(f"[bold green]Active Model:[/] [yellow]{active_model.id}[/] ([cyan]{active_model.provider}[/])\n")

    # Start scheduler background daemon thread
    active_model_ref = [active_model]
    import threading
    threading.Thread(
        target=scheduler_daemon_loop,
        args=(config_manager, provider_manager, active_model_ref),
        daemon=True
    ).start()

    # Launch desktop GUI if running interactively and --cli is not requested
    if "--cli" not in sys.argv and sys.stdin.isatty():
        console.print("[green]Launching DBERT Desktop GUI window...[/]")
        from src.ui.main_window import start_gui
        start_gui(config_manager, provider_manager)
        return

    # 4. Session Resolution
    session_manager = SessionManager(config_manager.app_dir / "history" / "chat.db")
    sessions = session_manager.list_sessions()
    active_session = None

    if sessions and active_model.id != "offline-setup":
        table = Table(title="Resumable Sessions", border_style="blue")
        table.add_column("No.", style="cyan")
        table.add_column("Session Name", style="white")
        table.add_column("Created At", style="dim white")
        table.add_column("Last Message Preview", style="dim green", max_width=40)
        
        for idx, s in enumerate(sessions[:5], 1):
            last_msg = s.last_message or "<No messages>"
            table.add_row(str(idx), s.name, s.created_at[:16].replace("T", " "), last_msg)
            
        console.print(table)
        console.print("  [bold cyan][N][/] Start a new session")
        
        while True:
            choice = Prompt.ask("Choose a session to resume, or [N] for new", default="N")
            if choice.upper() == "N":
                break
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(sessions[:5]):
                    resumed_summary = sessions[choice_idx]
                    active_session = session_manager.resume_session(resumed_summary.id)
                    console.print(f"[green]Resumed session: [bold]{active_session.name}[/][/]")
                    break
            except ValueError:
                pass
            console.print("[red]Invalid choice. Select a session index or 'N'.[/]")

    if not active_session:
        # Start new workspace session
        workspace_id = "default_workspace"
        active_session = session_manager.create_session(workspace_id)
        console.print(f"[green]Started new workspace session: [bold]{active_session.name}[/][/]")

    # Print session messages if resumed
    if active_session.messages:
        console.print("\n[bold dim white]--- Session History Recall ---[/]")
        for msg in active_session.messages:
            role_style = "bold green" if msg.role == "user" else "bold blue"
            role_name = "You" if msg.role == "user" else "DBERT"
            console.print(f"[{role_style}]{role_name}:[/] {msg.content}")
        console.print("[bold dim white]-----------------------------[/]\n")

    # 5. Interactive Chat Loop
    print_help()
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/]").strip()
        except KeyboardInterrupt:
            console.print("\n[yellow]Session terminated by KeyboardInterrupt. Goodbye![/]")
            break
            
        if not user_input:
            continue

        # Check for Slash Commands
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=2)
            cmd = cmd_parts[0].lower()
            
            if cmd == "/exit" or cmd == "/quit":
                for mcp_client in active_mcp_clients:
                    mcp_client.close()
                console.print("[cyan]Exiting DBERT. Your session is saved. Goodbye![/]")
                break
                
            elif cmd == "/help":
                print_help()
                continue
                
            elif cmd == "/permissions":
                default_level = config_manager.config.get("permissions", {}).get("default_level", "ask_every_time")
                console.print(f"[bold cyan]Permissions Configuration:[/]")
                console.print(f"  Default level: [yellow]{default_level}[/]")
                console.print(f"  Workspace ID: [yellow]{active_session.workspace_id}[/]")
                console.print("  Sandbox root: [yellow]Restricted to project workspace[/]")
                continue
                
            elif cmd == "/model":
                if active_model.id == "offline-setup":
                    # Refresh discovery
                    try:
                        available_models = discover_models(provider_manager)
                    except Exception:
                        available_models = []
                        
                if not available_models:
                    console.print("[red]No models available. Start LM Studio or add cloud provider key.[/]")
                    continue
                    
                if len(cmd_parts) > 1:
                    target_model = cmd_parts[1].strip()
                    matched = [m for m in available_models if m.id == target_model]
                    if matched:
                        active_model = matched[0]
                        active_model_ref[0] = active_model
                        config_manager.config["model_preference"] = active_model.id
                        config_manager.save_config()
                        console.print(f"[green]Active model switched to: [bold]{active_model.id}[/][/]")
                    else:
                        console.print(f"[red]Model '{target_model}' not found in available models list.[/]")
                else:
                    # Interactive Picker
                    active_model = console_picker(available_models)
                    active_model_ref[0] = active_model
                    config_manager.config["model_preference"] = active_model.id
                    config_manager.save_config()
                    console.print(f"[green]Active model switched to: [bold]{active_model.id}[/][/]")
                continue
                
            elif cmd == "/provider":
                tokens = user_input.split()
                if len(tokens) == 1:
                    providers_info = provider_manager.list_active_providers()
                    table = Table(title="Registered API Providers", border_style="cyan")
                    table.add_column("Provider Name", style="yellow")
                    table.add_column("Type", style="magenta")
                    table.add_column("Key Status", style="green")
                    table.add_column("Masked Key", style="white")
                    table.add_column("Connection Check", style="blue")
                    
                    for name, details in providers_info.items():
                        conn_str = "[green]Pass[/]" if details["connection_ok"] else "[red]Fail/Offline[/]"
                        table.add_row(
                            name,
                            details["type"],
                            details["key_status"],
                            details["masked_key"],
                            conn_str
                        )
                    console.print(table)
                    continue
                    
                elif len(tokens) >= 2:
                    subcmd = tokens[1].lower()
                    if subcmd == "add":
                        if len(tokens) >= 4:
                            p_name = tokens[2].lower()
                            p_key = tokens[3]
                            if p_name not in ["openai", "anthropic", "gemini"]:
                                console.print("[red]Unsupported provider. Choose 'openai', 'anthropic', or 'gemini'.[/]")
                                continue
                            console.print(f"[cyan]Adding provider {p_name} and verifying connection...[/]")
                            try:
                                provider_manager.register_cloud_provider(p_name, p_key)
                                provider_manager.register_cloud_providers()
                                
                                if provider_manager.test_provider_connection(p_name):
                                    console.print(f"[bold green]Provider {p_name} registered successfully![/]")
                                    available_models = discover_models(provider_manager)
                                else:
                                    console.print(f"[red]Provider {p_name} key verification failed. Check key and connection.[/]")
                            except Exception as e:
                                console.print(f"[red]Error registering provider: {e}[/]")
                        else:
                            console.print("[red]Syntax: /provider add <openai|anthropic|gemini> <key>[/]")
                            
                    elif subcmd == "remove":
                        if len(tokens) >= 3:
                            p_name = tokens[2].lower()
                            if p_name not in provider_manager.active_providers:
                                console.print(f"[red]Provider '{p_name}' is not currently active.[/]")
                                continue
                            provider_manager.remove_provider(p_name)
                            console.print(f"[green]Removed provider {p_name} successfully.[/]")
                            available_models = discover_models(provider_manager)
                        else:
                            console.print("[red]Syntax: /provider remove <name>[/]")
                    else:
                        console.print(f"[red]Unknown provider sub-command: {subcmd}. Use 'add' or 'remove'.[/]")
                continue

            elif cmd == "/fallback":
                tokens = user_input.split()
                if len(tokens) == 1:
                    fallback_order = config_manager.config.get("default_provider_order", [])
                    console.print(f"[bold cyan]Current Global Fallback Chain Order:[/]")
                    chain_str = " -> ".join([f"[yellow]{p}[/]" for p in fallback_order])
                    console.print(f"  {chain_str}")
                    continue
                elif len(tokens) >= 2:
                    subcmd = tokens[1].lower()
                    if subcmd == "set":
                        if len(tokens) >= 3:
                            order_str = "".join(tokens[2:])
                            new_order = [p.strip() for p in order_str.split(",") if p.strip()]
                            valid_providers = ["lmstudio-local", "openai", "anthropic", "gemini", "ollama"]
                            invalid = [p for p in new_order if p not in valid_providers]
                            if invalid:
                                console.print(f"[red]Invalid providers in order: {invalid}. Supported: {valid_providers}[/]")
                                continue
                            config_manager.config["default_provider_order"] = new_order
                            config_manager.save_config()
                            console.print(f"[green]Fallback chain order updated to: {new_order}[/]")
                        else:
                            console.print("[red]Syntax: /fallback set <provider1,provider2,...>[/]")
                    else:
                        console.print(f"[red]Unknown fallback sub-command: {subcmd}. Use 'set'.[/]")
                continue

            elif cmd == "/ingest":
                if len(cmd_parts) > 1:
                    f_path = cmd_parts[1].strip()
                    console.print(f"[cyan]Ingesting document {f_path}...[/]")
                    try:
                        from src.rag.ingest import ingest_file
                        from src.rag.summarizer import summarize_document
                        res = ingest_file(
                            f_path,
                            active_session.workspace_id,
                            provider_manager,
                            active_model,
                            app_dir=config_manager.app_dir,
                            batch_size=defaults.embedding_batch_size
                        )
                        if res["success"]:
                            console.print(f"[bold green]Ingestion successful![/] Chunks indexed: {res['chunks_count']}")
                            if Prompt.ask("Would you like to generate a document summary?", choices=["y", "n"], default="y") == "y":
                                console.print("[dim cyan]Generating summary...[/]")
                                summary = summarize_document(
                                    active_session.workspace_id,
                                    f_path,
                                    active_model,
                                    provider_manager,
                                    app_dir=config_manager.app_dir
                                )
                                console.print(Panel(summary, title=f"Summary: {Path(f_path).name}", border_style="cyan"))
                        else:
                            console.print(f"[red]Ingestion failed: {res.get('error', 'Unknown error')}[/]")
                    except Exception as e:
                        console.print(f"[red]Error during ingestion: {e}[/]")
                else:
                    console.print("[red]Syntax: /ingest <file_path>[/]")
                continue

            elif cmd == "/rag":
                if len(cmd_parts) > 1:
                    query_text = cmd_parts[1].strip()
                    try:
                        console.print(f"[cyan]Retrieving matches for query: '{query_text}'...[/]")
                        from src.rag.ingest import get_embeddings
                        from src.rag.vector_store import VectorStore
                        query_emb = get_embeddings([query_text], active_model, provider_manager)[0]
                        vs = VectorStore(active_session.workspace_id, config_manager.app_dir)
                        matches = vs.query(query_emb, top_k=3)
                        
                        table = Table(title=f"RAG Matches for '{query_text}'", border_style="cyan")
                        table.add_column("Document", style="yellow")
                        table.add_column("Similarity Score", style="green")
                        table.add_column("Text Preview", style="white")
                        
                        for doc_path, chunk_text, meta, score in matches:
                            table.add_row(
                                Path(doc_path).name,
                                f"{score:.4f}",
                                chunk_text[:80].replace("\n", " ") + "..."
                            )
                        console.print(table)
                    except Exception as e:
                        console.print(f"[red]Error querying vector store: {e}[/]")
                else:
                    console.print("[red]Syntax: /rag <query>[/]")
                continue

            elif cmd == "/voice":
                if active_model.id == "offline-setup":
                    console.print("[red]Voice mode requires an active model. Please select a model or load providers first.[/]")
                    continue
                from src.voice.voice_controller import start_voice_loop
                start_voice_loop(
                    active_session=active_session,
                    active_model=active_model,
                    provider_manager=provider_manager,
                    config_manager=config_manager,
                    session_manager=session_manager
                )
                continue

            elif cmd == "/research":
                if len(cmd_parts) >= 3:
                    scope = cmd_parts[1].strip().lower()
                    q = cmd_parts[2].strip()
                    if scope == "web":
                        console.print(f"[cyan]Initiating Web Deep Research for: '{q}'...[/]")
                        from src.research.deep_research_web import run_deep_research
                        try:
                            report = run_deep_research(q, active_session.workspace_id, active_model, provider_manager, config_manager)
                            console.print(Panel(report, title="Web Deep Research Report", border_style="green"))
                        except Exception as e:
                            console.print(f"[red]Web Deep Research failed: {e}[/]")
                    elif scope == "local":
                        console.print(f"[cyan]Initiating Local Deep Research for: '{q}'...[/]")
                        from src.research.deep_research_local import run_local_research
                        try:
                            report = run_local_research(q, active_session.workspace_id, active_model, provider_manager, config_manager)
                            console.print(Panel(report, title="Local Deep Research Report", border_style="green"))
                        except Exception as e:
                            console.print(f"[red]Local Deep Research failed: {e}[/]")
                    else:
                        console.print("[red]Syntax: /research <web|local> <query>[/]")
                else:
                    console.print("[red]Syntax: /research <web|local> <query>[/]")
                continue

            elif cmd == "/mcp":
                tokens = user_input.split()
                if len(tokens) >= 4 and tokens[1].lower() == "add":
                    cmd_str = " ".join(tokens[2:])
                    console.print(f"[cyan]Connecting to stdio MCP server: '{cmd_str}'...[/]")
                    from src.tools.mcp_client import MCPClient
                    try:
                        mcp_client = MCPClient(cmd_str)
                        if mcp_client.connect():
                            tools_list = mcp_client.list_tools()
                            for t in tools_list:
                                tool_name = t["name"]
                                description = t["description"]
                                params = t["inputSchema"]
                                
                                def make_mcp_handler(client, name):
                                    return lambda **kwargs: client.call_tool(name, kwargs)
                                    
                                global_registry.register_tool(
                                    name=tool_name,
                                    handler=make_mcp_handler(mcp_client, tool_name),
                                    description=description,
                                    parameters=params
                                )
                            console.print(f"[bold green]MCP server initialized successfully![/] Registered {len(tools_list)} tools.")
                            active_mcp_clients.append(mcp_client)
                        else:
                            console.print("[red]Failed to connect to MCP server.[/]")
                    except Exception as e:
                        console.print(f"[red]Error adding MCP server: {e}[/]")
                else:
                    console.print("[red]Syntax: /mcp add <command>[/]")
                continue

            elif cmd == "/schedule":
                tokens = user_input.split()
                import datetime
                import time
                from src.automation.scheduler import list_jobs, schedule_job, remove_job, run_pending_jobs
                if len(tokens) >= 2:
                    subcmd = tokens[1].lower()
                    if subcmd == "list":
                        jobs = list_jobs(config_manager.app_dir)
                        if not jobs:
                            console.print("[yellow]No scheduled jobs active.[/]")
                        else:
                            table = Table(title="Scheduled Automation Jobs", border_style="cyan")
                            table.add_column("ID", style="cyan")
                            table.add_column("Name", style="green")
                            table.add_column("Type", style="yellow")
                            table.add_column("Interval", style="magenta")
                            table.add_column("Next Run", style="blue")
                            table.add_column("Status", style="red")
                            table.add_column("Last Result", style="white")
                            
                            for job in jobs:
                                next_run_str = datetime.datetime.fromtimestamp(job["next_run"]).strftime("%H:%M:%S") if job.get("next_run") else "-"
                                table.add_row(
                                    job["id"],
                                    job["name"],
                                    job["type"],
                                    f"{job['interval_seconds']}s",
                                    next_run_str,
                                    job["status"],
                                    str(job.get("last_result", ""))[:40]
                                )
                            console.print(table)
                            
                    elif subcmd == "add" and len(tokens) >= 6:
                        job_type = tokens[2].lower()
                        name = tokens[3]
                        if job_type == "url":
                            url = tokens[4]
                            try:
                                interval = int(tokens[5])
                            except ValueError:
                                interval = 60
                                
                            job_def = {
                                "name": name,
                                "type": "url_monitor",
                                "interval_seconds": interval,
                                "payload": {"url": url}
                            }
                            job_id = schedule_job(job_def, config_manager.app_dir)
                            console.print(f"[bold green]URL Monitor scheduled successfully! ID: {job_id}[/]")
                            
                        elif job_type == "task":
                            q_parts = user_input.split('"')
                            if len(q_parts) >= 3:
                                prompt_str = q_parts[1]
                                rest = q_parts[2].strip().split()
                                try:
                                    interval = int(rest[0])
                                except Exception:
                                    interval = 60
                            else:
                                prompt_str = tokens[4]
                                try:
                                    interval = int(tokens[-1])
                                except Exception:
                                    interval = 60
                                    
                            job_def = {
                                "name": name,
                                "type": "agent_task",
                                "interval_seconds": interval,
                                "payload": {"prompt": prompt_str}
                            }
                            job_id = schedule_job(job_def, config_manager.app_dir)
                            console.print(f"[bold green]AI Task scheduled successfully! ID: {job_id}[/]")
                        else:
                            console.print("[red]Syntax: /schedule add <url|task> <name> <url|prompt> <interval>[/]")
                            
                    elif subcmd == "remove" and len(tokens) >= 3:
                        job_id = tokens[2]
                        if remove_job(job_id, config_manager.app_dir):
                            console.print(f"[green]Job {job_id} removed successfully.[/]")
                        else:
                            console.print(f"[red]Job {job_id} not found.[/]")
                            
                    elif subcmd == "run":
                        console.print("[cyan]Triggering execution of pending jobs immediately...[/]")
                        jobs = list_jobs(config_manager.app_dir)
                        for j in jobs:
                            j["next_run"] = time.time() - 10
                            schedule_job(j, config_manager.app_dir)
                        run_pending_jobs(config_manager, provider_manager, active_model)
                        console.print("[green]Execution run triggered.[/]")
                    else:
                        console.print("[red]Syntax: /schedule <list|add|remove|run>[/]")
                else:
                    console.print("[red]Syntax: /schedule <list|add|remove|run>[/]")
                continue
                
            else:
                console.print(f"[red]Unknown slash command: {cmd}. Type /help for assistance.[/]")
                continue

        # If we are offline and don't have a valid model, we block chat queries
        if active_model.id == "offline-setup":
            console.print("[red]No model active. Run LM Studio local server or add cloud provider key using `/provider add`.[/]")
            continue

        # 1. Generate query embedding (if provider is active)
        query_emb = None
        try:
            from src.rag.ingest import get_embeddings
            query_emb = get_embeddings([user_input], active_model, provider_manager)[0]
        except Exception as e:
            logger.warning(f"Could not generate query embedding: {e}")

        # Save user message to session DB (include embedding)
        session_manager.append_message(active_session.id, "user", user_input, embedding=query_emb)
        active_session.messages.append(Message(role="user", content=user_input))

        # 2. Retrieve document context
        context_docs = []
        if query_emb:
            try:
                from src.rag.vector_store import VectorStore
                vs = VectorStore(active_session.workspace_id, config_manager.app_dir)
                matches = vs.query(query_emb, top_k=3)
                for doc_path, chunk_text, meta, score in matches:
                    if score > 0.35:  # Relevance threshold
                        context_docs.append(f"Source: {Path(doc_path).name} (Score: {score:.2f})\n{chunk_text}")
            except Exception as e:
                logger.error(f"Error querying document context: {e}")

        # 3. Retrieve semantic chat history context
        context_history = []
        if query_emb:
            try:
                from src.memory.history_search import semantic_search_history
                history_matches = semantic_search_history(
                    session_manager.db_path,
                    query_emb,
                    active_session.workspace_id,
                    top_k=3
                )
                for match in history_matches:
                    if match.similarity > 0.4 and match.content.strip().lower() != user_input.strip().lower():
                        context_history.append(f"Past Turn ({match.role}): {match.content}")
            except Exception as e:
                logger.error(f"Error querying history context: {e}")

        # 4. Form grounded user input
        grounded_input = ""
        if context_docs:
            grounded_input += "\n[CONTEXT FROM LOCAL DOCUMENTS]\n" + "\n---\n".join(context_docs) + "\n"
        if context_history:
            grounded_input += "\n[CONTEXT FROM PAST CONVERSATIONS]\n" + "\n---\n".join(context_history) + "\n"
            
        if grounded_input:
            grounded_input += f"\nUser Query: {user_input}\n"
        else:
            grounded_input = user_input

        # 5. Form message payload for LiteLLM
        litellm_messages = []
        litellm_messages.append({"role": "system", "content": "You are DBERT, a local-first privacy-first AI desktop assistant."})
        
        # Add rolling session history (exclude the user message we just appended)
        for msg in active_session.messages[-15:-1]:
            litellm_messages.append({"role": msg.role, "content": msg.content})
            
        # Add grounded current turn user input
        litellm_messages.append({"role": "user", "content": grounded_input})

        console.print("[dim cyan]DBERT thinking...[/]")
        try:
            assistant_reply, answered_model = execute_completion_with_fallback(
                active_model,
                litellm_messages,
                provider_manager,
                config_manager,
                workspace_id=active_session.workspace_id
            )
            
            if answered_model.id != active_model.id:
                console.print(f"[bold yellow][!] Primary model offline — fell back to {answered_model.id}[/]")
                
            console.print(f"[bold blue]DBERT:[/] {assistant_reply}")
            
            # Embed DBERT response
            reply_emb = None
            try:
                from src.rag.ingest import get_embeddings
                reply_emb = get_embeddings([assistant_reply], active_model, provider_manager)[0]
            except Exception as e:
                logger.warning(f"Could not embed assistant response: {e}")
                
            session_manager.append_message(active_session.id, "assistant", assistant_reply, embedding=reply_emb)
            active_session.messages.append(Message(role="assistant", content=assistant_reply))
            
        except Exception as e:
            console.print(f"[bold red]Inference Error:[/] {e}")
            logger.error(f"Error during completion: {e}")

if __name__ == "__main__":
    main()
