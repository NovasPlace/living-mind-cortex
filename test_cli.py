import asyncio
from cli import TelemetryCLI
async def test():
    app = TelemetryCLI()
    async with app.run_test() as pilot:
        await pilot.click("#input-box")
        await pilot.press(*"Hello")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log")
        print("CHAT LINES:", log.lines)
asyncio.run(test())
