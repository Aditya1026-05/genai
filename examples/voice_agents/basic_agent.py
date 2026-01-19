import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# --------------------------------------------------
# Setup
# --------------------------------------------------

logger = logging.getLogger("basic-agent")
logging.basicConfig(level=logging.INFO)

load_dotenv()

# ----- Interruption handling (REQUIRED BY TASK) -----
agent_is_speaking = False

IGNORE_WORDS = {
    "yeah",
    "ok",
    "okay",
    "hmm",
    "uh huh",
    "uh-huh",
    "right",
}

INTERRUPT_WORDS = {
    "stop",
    "wait",
    "no",
    "hold on",
}

# --------------------------------------------------
# Agent definition
# --------------------------------------------------

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact with users via voice. "
                "Keep responses concise and to the point. "
                "Do not use emojis, markdown, or special characters. "
                "You are curious, friendly, and speak English."
            )
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."

# --------------------------------------------------
# Server setup
# --------------------------------------------------

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

# --------------------------------------------------
# Server session
# --------------------------------------------------

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    global agent_is_speaking

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # ----- Track agent speaking state -----
    @session.on("tts_start")
    def _():
        global agent_is_speaking
        agent_is_speaking = True
        logger.info("Agent started speaking")

    @session.on("tts_end")
    def _():
        global agent_is_speaking
        agent_is_speaking = False
        logger.info("Agent finished speaking")

    # ----- CORE TASK LOGIC -----
    @session.on("transcript_final")
    def on_user_transcript(text: str):
        global agent_is_speaking

        clean = text.lower().strip()
        logger.info(f"User said: {clean}")

        if agent_is_speaking:
            # Ignore passive acknowledgements
            if clean in IGNORE_WORDS:
                logger.info("Ignoring passive acknowledgement")
                return

            # Interrupt commands
            if any(word in clean for word in INTERRUPT_WORDS):
                logger.info("Interrupt command detected")
                session.interrupt()
                return

            # Ignore everything else while speaking
            logger.info("Ignoring input while agent is speaking")
            return

        # Agent is silent → normal behavior
        session.generate_reply(instructions=clean)

    # ----- Metrics (unchanged) -----
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":
    cli.run_app(server)

