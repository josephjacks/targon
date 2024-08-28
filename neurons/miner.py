import traceback
import time
from bittensor.subtensor import serve_extrinsic
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
import netaddr
import requests
from starlette.responses import StreamingResponse

from neurons.base import BaseNeuron, NeuronType
from targon.epistula import (
    epistula_v2,
)
from targon.utils import print_info
import uvicorn
import bittensor as bt

from bittensor.axon import FastAPIThreadedServer


class Miner(BaseNeuron):
    neuron_type = NeuronType.Miner
    fast_api: FastAPIThreadedServer

    def shutdown(self):
        if self.fast_api:
            self.fast_api.stop()

    def __init__(self, config=None):
        super().__init__(config)
        ## Typesafety
        assert self.config.netuid
        assert self.config.neuron
        assert self.config.logging

        ## BITTENSOR INITIALIZATION
        bt.logging.info(
            "\N{grinning face with smiling eyes}", "Successfully Initialized!"
        )

    async def inference(self, request: Request):
        bt.logging.info("\u2713", "Getting Inference request!")

        async def stream(req):
            assert self.config.neuron
            assert req.sampling_params
            stream = self.client.chat.completions.create(
                    **req
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token.encode("utf-8")
            bt.logging.info("\N{grinning face}", "Processed forward")
        req = await request.json()
        return StreamingResponse(stream(req))

    async def verify_request(self, request: Request):
        params = [request, self.wallet.hotkey.ss58_address, self.metagraph.hotkeys]
        version = request.headers.get("Epistula-Version")
        match version:
            case "2":
                await epistula_v2(*params)
            case _:
                raise HTTPException(status_code=400, detail=f"Unknown Epistula Version {version}")

    def run(self):
        assert self.config.netuid
        assert self.config.subtensor
        assert self.config.axon
        assert self.config.neuron

        # Check that miner is registered on the network.
        self.sync_metagraph()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        external_ip = self.config.axon.ip
        if not external_ip or external_ip == "[::]":
            try:
                external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
                netaddr.IPAddress(external_ip)
            except Exception:
                bt.logging.error("Failed to get external IP")

        bt.logging.info(
            f"Serving miner endpoint {external_ip}:{self.config.axon.port} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        serve_success = serve_extrinsic(
            subtensor=self.subtensor,
            wallet=self.wallet,
            ip=external_ip,
            port=self.config.axon.port,
            protocol=4,
            netuid=self.config.netuid,
        )
        if not serve_success:
            bt.logging.error("Failed to serve endpoint")
            return

        # Start  starts the miner's endpoint, making it active on the network.
        # change the config in the axon
        app = FastAPI()
        router = APIRouter()
        router.add_api_route(
            "/inference",
            self.inference,
            dependencies=[Depends(self.verify_request)],
            methods=["POST"],
        )
        app.include_router(router)
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.axon.port,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.info(f"Miner starting at block: {self.subtensor.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                # Print Logs for Miner
                print_info(
                    self.metagraph,
                    self.wallet.hotkey.ss58_address,
                    self.subtensor.block,
                )
                # Wait before checking again.
                time.sleep(12)

                # Sync metagraph if stale
                self.sync_metagraph()
        except Exception as e:
            bt.logging.error(str(e))
            bt.logging.error(traceback.format_exc())
        self.shutdown()


if __name__ == "__main__":
    try:
        miner = Miner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
