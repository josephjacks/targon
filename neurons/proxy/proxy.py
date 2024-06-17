from time import time
from sse_starlette.sse import EventSourceResponse
from fastapi import Request
import os
from bittensor import typing, logging, metagraph
from dotenv import load_dotenv
from bittensor.axon import FastAPI, uvicorn

from targon import protocol
from targon.verifier.inference import select_highest_n_peers


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
        metagraph = None # @TODO carro
        uid = select_highest_n_peers(1, metagraph)[0]
        res = ""
        async for token in await dendrite.forward( # @TODO carro
            metagraph.axons[uid],
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
    app = FastAPI()
    app.router.add_api_route(
        "/api/chat/completions", safeParseAndCall, methods=["POST"]
    )
    uvicorn.run(
        app, host="0.0.0.0", loop="asyncio", port=int(os.getenv("PROXY_PORT", 8081))
    )
