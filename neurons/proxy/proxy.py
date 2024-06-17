from time import time
from sse_starlette.sse import EventSourceResponse
from fastapi import Request
import os
from bittensor import typing, logging, 
from bittensor import subtensor as Subtensor, dendrite as Dendrite
from dotenv import load_dotenv
from bittensor.axon import FastAPI, uvicorn

from targon import protocol
from targon.verifier.inference import select_highest_n_peers


import logging
import time
from typing import Optional


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
        self.metagraph: Optional[Subtensor.metagraph] = None

    def sync(self):
        subtensor = Subtensor()
        self.is_syncing = True
        try:
            sync_start = time.time()
            metagraph: metagraph = subtensor.metagraph(netuid=self.netuid)
            self.metagraph = metagraph
            self.last_sync_success = time.time()
            logging.info(
                f"Synced metagraph for netuid {self.netuid} (took {self.last_sync_success - sync_start:.2f} seconds)",
            )
            return metagraph
        except Exception as e:
            logging.warning("Could not sync metagraph: ", e)
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
                except Exception as e:
                    time.sleep(self.extra_fail_rest_seconds)
                time.sleep(self.rest_seconds)

        import threading

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        logging.info("Started metagraph sync thread")
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

        start_time = time()
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

        end_time = time()
        elapsed_time = end_time - start_time
        tokens_per_second = token_count / elapsed_time
        logging.info(f"Token generation rate: {tokens_per_second} tokens/second")
        logging.info(f"{res} | {token_count}")
    except Exception as e:
        logging.error(e)


load_dotenv()
TOKEN = os.getenv("HUB_SECRET_TOKEN")


async def safeParseAndCall(req: Request):
    data = await req.json()
    if data.get("api_key") != TOKEN and TOKEN is not None:
        return "", 401

    logging.info("Received organic request")
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
        logging.error(f"Failed due to: {e}")
        return "", 500


if __name__ == "__main__":

    dendrite = Dendrite()

    metagraph_controller = MetagraphController(netuid=4)
    metagraph_controller.start_sync_thread()

    app = FastAPI()
    app.router.add_api_route(
        "/api/chat/completions", safeParseAndCall, methods=["POST"]
    )
    uvicorn.run(
        app, host="0.0.0.0", loop="asyncio", port=int(os.getenv("PROXY_PORT", 8081))
    )
