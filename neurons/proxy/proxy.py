from time import time
from sse_starlette.sse import EventSourceResponse
from fastapi import APIRouter, Request
import os
import bittensor as bt
import typing
from dotenv import load_dotenv
from bittensor.axon import FastAPI, uvicorn

from targon import protocol
from targon.verifier.inference import select_highest_n_peers


import time
from typing import Optional, Union

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


async def api_chat_completions(
    prompt: str,
    sampling_params: protocol.InferenceSamplingParams,
) -> typing.Tuple[bool, protocol.Inference]:
    """
    Handles a inference sent to a prover and verifies the response.

    Returns:
    - Tuple[bool, protocol.Inference]: A tuple containing the verification result and the inference.
    """
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
        async for token in await dendrite(
            metagraph_controller.metagraph.axons[uid],
            synapse,
            deserialize=False,
            run_async=False,
            streaming=True,
        ):
            if isinstance(token, list):
                res += token[0]
                yield token[0]
            elif isinstance(token, str):
                res += token
                yield token
            token_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = token_count / elapsed_time
        bt.logging.info(f"Token generation rate: {tokens_per_second} tokens/second")
        bt.logging.info(f"{res} | {token_count}")
    except Exception as e:
        bt.logging.error(e)


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

    try:
        return EventSourceResponse(
            api_chat_completions(
                prompt,
                protocol.InferenceSamplingParams(
                    max_new_tokens=data.get("max_tokens", 1024)
                ),
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        bt.logging.error(f"Failed due to: {e}")
        return "", 500


if __name__ == "__main__":
    bt.logging.on()
    bt.logging.set_debug(True)
    bt.logging.set_trace(True)
    bt.turn_console_on()

    wallet_name = safeEnv('PROXY_WALLET')
    wallet = bt.wallet(wallet_name)
    dendrite = bt.dendrite(wallet=wallet)

    metagraph_controller = MetagraphController(netuid=4)
    metagraph_controller.start_sync_thread()

    app = FastAPI()
    app.include_router(router)
    bt.logging.info("Starting Prxy")
    uvicorn.run(
        app, host="0.0.0.0", loop="asyncio", port=int(os.getenv("PROXY_PORT", 8081))
    )
