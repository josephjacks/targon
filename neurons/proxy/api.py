import time
import uuid
import json
import logging
import asyncio
import numpy as np
import bittensor as bt
from fastapi import FastAPI, Request

from typing import Optional
from sse_starlette.sse import EventSourceResponse
from targon.protocol import Inference, InferenceSamplingParams


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


def select_highest_n_peers(n: int, metagraph=None, return_all=False):
    """
    Selects the highest incentive peers from the metagraph.

    Parameters:
        n (int): number of top peers to return.

    Returns:
        int: uid of the selected peer from unique highest IPs.
    """
    assert metagraph is not None, "metagraph is None"
    # Get the top n indices based on incentive
    indices = np.argsort(metagraph.incentive)[-n:][::-1]
    # Get the corresponding uids
    uids_with_highest_incentives = metagraph.uids[indices].tolist()


    if return_all:
        return uids_with_highest_incentives
    
    # get the axon of the uids
    axons = [metagraph.axons[uid] for uid in uids_with_highest_incentives]

    # get the ip from the axons
    ips = [axon.ip for axon in axons]

    # get the coldkey from the axons
    coldkeys = [axon.coldkey for axon in axons]

    # Filter out the uids and ips whose coldkeys are in the blacklist
    uids_with_highest_incentives, ips = zip(*[(uid, ip) for uid, ip, coldkey in zip(uids_with_highest_incentives, ips, coldkeys)])

    unique_ip_to_uid = {ip: uid for ip, uid in zip(ips, uids_with_highest_incentives)}
    uids = list(unique_ip_to_uid.values())
    return uids_with_highest_incentives


class API:
    def __init__(self, metagraph_controller: MetagraphController, wallet: bt.wallet):
        self.metagraph_controller = metagraph_controller
        self.wallet = wallet
        

    def _validate_metagraph(self):
        if self.metagraph_controller.metagraph is None:
            raise MetagraphNotSyncedException()
        
    async def generate(
            self,
            query: str,
            sources: list[str] = [],
            max_new_tokens: int = 12,
    ):
        uids = select_highest_n_peers(192, self.metagraph_controller.metagraph)
        top_k_axons = [self.metagraph_controller.metagraph.axons[uid] for uid in uids]

        inference_params = InferenceSamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

        search_result_synapse = Inference(
            query=query,
            sources=sources,
            inference_params=inference_params
        )
        
        results = [asyncio.create_task(self.dendrite(axons=[axon], synapse=search_result_synapse, timeout=60, streaming=True)) for axon in top_k_axons]

        # Set a global timeout (for example, 120 seconds)
        global_timeout = 120

        # try:
        while True:
            # Wait for the first task to finish
            done, pending = await asyncio.wait(results, timeout=global_timeout, return_when=asyncio.FIRST_COMPLETED)
            # If the first task finished, cancel the rest
            if len(done) > 0:
                for task in pending:
                    task.cancel()
                break
            # If the first task did not finish, raise an exception
            else:
                raise asyncio.TimeoutError()
        
        # Get the result from the first task
        result = done.pop().result()
        return result
        # except asyncio.TimeoutError:
        #     # If the first task did not finish within the global timeout, cancel all tasks
        #     for task in results:
        #         task.cancel()
        #     raise asyncio.TimeoutError()
        # except asyncio.CancelledError:
        #     # Handle cancelled tasks
        #     logging.error("Task was cancelled")
        #     raise



def random_uuid() -> str:
    """Generate a random UUID."""
    return str(uuid.uuid4())       

async def send_generate_request( query: str, sources: list[str] = [], max_new_tokens: int = 256 ):
    return await API(wallet=bt.wallet(name="opentensor_validator")).generate(
        query=query,
        sources=sources,
        max_new_tokens=max_new_tokens,
        )

async def stream_generate_request( query: str, sources: list[str] = [], max_new_tokens: int = 256 ):
    
    request = await send_generate_request(query, sources, max_new_tokens)
    print(request)
    # completion = ""
    # async for chunk in request:
    #     completion += chunk
    #     print(chunk)
    #     yield preprocess_answer(chunk, False, random_uuid())
    # yield preprocess_answer(completion, True, random_uuid())

def preprocess_answer(token: str, finished: bool, uuid=random_uuid()):
    if token == "<s>" or token == "</s>" or token == "<im_end>":
        token = ""

    params = {"type": "answer", "text": token, "finished": finished}

    return {
        "event": "new_message",
        "id": uuid,
        "retry": 1500,
        "data": json.dumps(params),
    }


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate")
async def search(request: Request):
    r = await request.json()
    query = r['query']
    return EventSourceResponse(
        stream_generate_request(query),
        media_type="text/event-stream",
    )

if __name__ == "__main__":
    import uvicorn
    bt.logging.on()
    bt.logging.set_debug(True)
    bt.logging.set_trace(True)
    bt.turn_console_on()
    uvicorn.run(app, host="0.0.0.0", port=8000)