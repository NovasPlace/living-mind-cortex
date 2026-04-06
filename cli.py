#!./.venv/bin/python
import asyncio
import json
import websockets
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, RichLog
from textual.reactive import reactive
from datetime import datetime

SPINNERS = ["(⌐■_■)", "(⊙_⊙)", "( ⚆_⚆)", "(¬_¬)", "(ᵔᴥᵔ)", "(✿◠‿◠)"]

class TelemetryCLI(App):
    TITLE = "Living Mind"
    CSS = """
    Screen { background: #0d1117; }

    /* Top identity bar */
    #header-bar {
        height: 3;
        dock: top;
        background: #0d1117;
        color: #58a6ff;
        content-align: center middle;
        text-style: bold;
        border-bottom: solid #21262d;
        padding: 0 2;
    }

    /* Main body */
    #main-layout {
        height: 1fr;
        layout: horizontal;
    }

    /* Chat pane - takes most of the space */
    #chat-pane {
        width: 75%;
        height: 1fr;
        border-right: solid #21262d;
    }

    #chat-log {
        height: 1fr;
        background: #0d1117;
        padding: 0 2;
        scrollbar-color: #21262d #0d1117;
    }

    /* Side pane */
    #side-pane {
        width: 28%;
        height: 1fr;
        layout: vertical;
    }

    /* Inner monologue - runtime thinking */
    #thought-stream {
        height: 40%;
        background: #0d1117;
        color: #e3b341;
        padding: 1 2;
        border-bottom: solid #21262d;
    }

    /* Trace log - system traces */
    #trace-log {
        height: 60%;
        background: #0d1117;
        padding: 0 1;
        color: #3fb950;
        scrollbar-color: #21262d #0d1117;
    }

    /* Bottom stats footer */
    #footer-bar {
        height: 1;
        dock: bottom;
        background: #161b22;
        color: #8b949e;
        content-align: left middle;
        padding: 0 2;
    }

    /* Input */
    #input-box {
        dock: bottom;
        height: 3;
        background: #0d1117;
        border-top: solid #21262d;
        border-bottom: none;
        border-left: none;
        border-right: none;
        padding: 0 2;
        color: #e6edf3;
    }
    Input > .input--placeholder { color: #484f58; }
    """

    pulse_count = reactive(0)
    organ_count = reactive(0)
    memory_count = reactive(0)
    awakening_count = reactive(0)
    gate_count = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static(
            "🧬  LIVING MIND  ·  Sovereign Neural Interface  ·  v1.0",
            id="header-bar"
        )

        with Horizontal(id="main-layout"):
            with Vertical(id="chat-pane"):
                yield RichLog(id="chat-log", markup=True, highlight=False, wrap=True)

            with Vertical(id="side-pane"):
                yield Static("", id="thought-stream")
                yield RichLog(id="trace-log", markup=True, highlight=False, wrap=True)

        yield Input(placeholder="  ❯  speak to the runtime, or /hesed /gevurah /approve /reject ...", id="input-box")
        yield Static("", id="footer-bar")

    async def on_mount(self):
        self.last_thought_cache = ""
        self.spinner_idx = 0
        self.thinking = False

        log = self.query_one("#chat-log")
        log.write("[dim]  neural link active. runtime online.[/dim]")
        log.write("")

        self.query_one("#thought-stream").update("[dim]  ...[/dim]")
        self._update_footer()
        asyncio.create_task(self.connect_pulse())

    def _update_footer(self):
        stats = (
            f"  ψ pulse:{self.pulse_count}  "
            f"⬡ organs:{self.organ_count}  "
            f"⌖ gates:{self.gate_count}  "
            f"◈ mem:{self.memory_count}  "
            f"✦ awakenings:{self.awakening_count}"
        )
        try:
            self.query_one("#footer-bar").update(stats)
        except:
            pass

    def _clean(self, text: str) -> str:
        import re
        # Strip simulation tags first
        text = re.sub(r'\[Simulation:.*?\]', '', text, flags=re.DOTALL).strip()
        cleaned = text.replace("```json", "").replace("```", "").strip("`").strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                for key in ("response", "thought", "prediction", "outcome"):
                    if key in data:
                        return re.sub(r'\[Simulation:.*?\]', '', str(data[key]), flags=re.DOTALL).strip()
                return " · ".join(str(v) for v in data.values() if v)
            return str(data)
        except:
            return cleaned

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    async def connect_pulse(self):
        while True:
            try:
                async with websockets.connect("ws://localhost:8008/ws/pulse") as ws:
                    self.query_one("#trace-log").write(
                        f"[dim green]  ✓ tethered to hypervisor[/dim green]"
                    )
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        if data.get("type") == "pulse":
                            vitals = data.get("data", {})
                            self.pulse_count = vitals.get("event_loops", 0)
                            self.organ_count = len(vitals.get("immune", {}).get("census", []))
                            self.memory_count = vitals.get("memory", {}).get("total", 0)
                            self.awakening_count = vitals.get("awakening", {}).get("total_meditations", 0)
                            self._update_footer()

                            t = self._clean(vitals.get("brain", {}).get("last_thought", ""))
                            if t and t != self.last_thought_cache:
                                self.last_thought_cache = t
                                self.query_one("#thought-stream").update(
                                    f"[bold yellow]  ❯[/bold yellow] [yellow]{t}[/yellow]"
                                )

                        elif data.get("type") == "event" and data.get("event") == "topology":
                            top = json.loads(data.get("message", "{}"))
                            self.gate_count = sum(len(n.get("functions", [])) for n in top.get("nodes", []))
                            self._update_footer()

                        elif data.get("type") == "event" and data.get("event") == "function_fire":
                            fire = json.loads(data.get("message", "{}"))
                            org = fire.get("organ", "?")
                            func = fire.get("function", "?")
                            self.query_one("#trace-log").write(
                                f"[dim]  [green]✓[/green] {org}[/dim][dim green].{func}()[/dim green]"
                            )

                        elif data.get("type") == "event" and data.get("event") == "chat_reply":
                            msg = self._clean(data.get("message", ""))
                            ts = self._ts()
                            self.query_one("#chat-log").write(
                                f"[dim]  {ts}[/dim]  [bold white]MIND[/bold white]  {msg}"
                            )

            except Exception as e:
                self.query_one("#trace-log").write(f"[dim red]  ✗ link lost — retrying...[/dim red]")
                await asyncio.sleep(3)

    async def on_input_submitted(self, message: Input.Submitted):
        text = message.value.strip()
        if not text:
            return
        message.input.value = ""

        log = self.query_one("#chat-log")
        ts = self._ts()
        log.write(f"[dim]  {ts}[/dim]  [bold cyan]YOU[/bold cyan]   {text}")

        asyncio.create_task(self.send_stimulus(text, log, ts))

    async def send_stimulus(self, text: str, log: RichLog, ts: str):
        try:
            async with websockets.connect("ws://localhost:8008/ws/stimulus") as ws:
                payload = {"node": "malkhut", "text": text}
                sys_msg = None

                if text == "/hesed":
                    payload = {"node": "hesed"}
                    sys_msg = "dopamine expansion injected (Hesed)"
                elif text == "/gevurah":
                    payload = {"node": "gevurah"}
                    sys_msg = "pruning strike initiated (Gevurah)"
                elif text == "/approve":
                    payload = {"node": "approve"}
                    sys_msg = "Motor Cortex action approved ✓"
                elif text == "/reject":
                    payload = {"node": "reject"}
                    sys_msg = "Motor Cortex action rejected ✗"

                await ws.send(json.dumps(payload))

                if sys_msg:
                    log.write(f"[dim]  {ts}[/dim]  [dim yellow]SYS[/dim yellow]   [dim]{sys_msg}[/dim]")

        except Exception as e:
            log.write(f"[dim]  {ts}[/dim]  [dim red]ERR[/dim red]   {e}")


if __name__ == "__main__":
    app = TelemetryCLI()
    app.run()
