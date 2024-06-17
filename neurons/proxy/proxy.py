import asyncio
from multiprocessing import Process
from time import sleep, time
import threading
from fastapi import APIRouter, Request
import os
import contextlib
import bittensor as bt
from dotenv import load_dotenv
from bittensor.axon import FastAPI, uvicorn
from sse_starlette.sse import EventSourceResponse

from targon import protocol
from targon.verifier.inference import select_highest_n_peers


import time
from typing import Optional


def safeEnv(key: str) -> str:
    var = os.getenv(key)
    if var == None:
        bt.logging.error(f"Missing env variable {key}")
        exit()
    return var


class MetagraphNotSyncedException(Exception):
    pass


class MetagraphController:
    def __init__(
        self, netuid: int, rest_seconds: int = 60, extra_fail_rest_seconds: int = 60
    ):
        self.last_sync_success = None
        self.netuid = netuid
        self.is_syncing = False
        self.rest_seconds = rest_seconds
        self.extra_fail_rest_seconds = extra_fail_rest_seconds
        self.metagraph: Optional[bt.metagraph] = None

    def sync(self):
        subtensor = bt.subtensor()
        self.is_syncing = True
        try:
            sync_start = time.time()
            metagraph: bt.metagraph = subtensor.metagraph(netuid=self.netuid)
            self.metagraph = metagraph
            self.last_sync_success = time.time()
            bt.logging.info(
                f"Synced metagraph for netuid {self.netuid} (took {self.last_sync_success - sync_start:.2f} seconds)",
            )
            return metagraph
        except Exception as e:
            bt.logging.warning("Could not sync metagraph: ", e)
            raise e  # Reraise the exception to be handled by the caller
        finally:
            self.is_syncing = False
            return None

    def start_sync_thread(self):
        self.last_sync_success = time.time()

        # Run in a separate thread
        def loop():
            while True:
                try:
                    self.sync()
                except Exception:
                    time.sleep(self.extra_fail_rest_seconds)
                time.sleep(self.rest_seconds)

        import threading

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        bt.logging.info("Started metagraph sync thread")
        return thread


load_dotenv()
TOKEN = os.getenv("HUB_SECRET_TOKEN")

router = APIRouter()


@router.post("/api/chat/completions")
async def safeParseAndCall(req: Request):
    bt.logging.info("New Parse Request")
    data = await req.json()
    if data.get("api_key") != TOKEN and TOKEN is not None:
        bt.logging.warning("Unverified request")
        return "", 401

    bt.logging.info("Received organic request")
    messages = data.get("messages")
    if not isinstance(messages, list):
        return "", 403
    prompt = "\n".join([p["role"] + ": " + p["content"] for p in messages])

    async def api_chat_completions(
        prompt: str,
        sampling_params: protocol.InferenceSamplingParams,
    ):
        try:
            synapse = protocol.Inference(
                sources=[],
                query=prompt,
                sampling_params=sampling_params,
            )

            start_time = time.time()
            token_count = 0
            uid = select_highest_n_peers(1, metagraph_controller.metagraph)[0]
            res = ""
            yield {"event": "new_token", "data": "FIRST TOKEN"}
            async for token in await dendrite(
                metagraph_controller.metagraph.axons[uid],
                synapse,
                deserialize=False,
                streaming=True,
            ):
                if isinstance(token, list):
                    res += token[0]
                    bt.logging.info(f"token: {token[0]}")
                    yield {"event": "new_token", "data": token[0]}
                elif isinstance(token, str):
                    res += token
                    bt.logging.info(f"token: {token}")
                    yield {"event": "new_token", "data": token}
                token_count += 1

            end_time = time.time()
            elapsed_time = end_time - start_time
            tokens_per_second = token_count / elapsed_time
            bt.logging.info(f"Token generation rate: {tokens_per_second} tokens/second")
            bt.logging.info(f"{res} | {token_count}")
        except Exception as e:
            bt.logging.error(e)

    return EventSourceResponse(
        api_chat_completions(
            prompt,
            protocol.InferenceSamplingParams(
                max_new_tokens=data.get("max_tokens", 1024)
            ),
        ),
        media_type="text/event-stream",
    )


async def testDendrite():
    uid = select_highest_n_peers(1, metagraph_controller.metagraph)[0]
    res = ""
    synapse = protocol.Inference(
        sources=[],
        query="what is the x y problem",
        sampling_params=protocol.InferenceSamplingParams(
            max_new_tokens=1024,
        ),
    )
    bt.logging.info(synapse.dict())
    async for token in await dendrite(
        metagraph_controller.metagraph.axons[uid],
        synapse,
        deserialize=False,
        streaming=True,
    ):
        if isinstance(token, list):
            res += token[0]
            yield token[0]
        elif isinstance(token, str):
            res += token
            yield token


async def testWrapper():
    async for token in testDendrite():
        bt.logging.info(token)

class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

if __name__ == "__main__":
    bt.logging.on()
    bt.logging.set_debug(True)
    bt.logging.set_trace(True)
    bt.turn_console_on()

    wallet_name = safeEnv("PROXY_WALLET")
    wallet = bt.wallet(wallet_name)
    dendrite = bt.dendrite(wallet=wallet)
    metagraph_controller = MetagraphController(netuid=4)
    metagraph_controller.start_sync_thread()
    while metagraph_controller.metagraph is None:
        sleep(1)
    app = FastAPI()
    app.include_router(router)
    bt.logging.info("Starting Proxy")
    config = uvicorn.Config(app,loop='asyncio', host="0.0.0.0", port=int(os.getenv('PROXY_PORT', 8081)))
    server = Server(config=config)
    with server.run_in_thread():
        pass
    bt.logging.info('shutting down')
