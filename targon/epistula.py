import json
import bittensor as bt
from hashlib import sha256
from uuid import uuid4
from math import ceil
from typing import Annotated, Any, Dict, List, Optional, Union

import time
from fastapi import HTTPException, Request
from substrateinterface import Keypair


def generate_header(
    hotkey: Keypair,
    body: Union[Dict[Any, Any], List[Any]],
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4)
    uuid = str(uuid4())
    headers = {
        "Epistula-Version": str(2),
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(
            f"{sha256(json.dumps(body).encode('utf-8')).hexdigest()}.{uuid}.{timestamp}.{signed_for or ''}"
        ).hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-1"] = (
            "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        )
    return headers


async def epistula_v2(request: Request, ss58, hotkeys):
    # We do this as early as possible so that now has a lesser chance
    # of causing a stale request
    now = round(time.time() * 1000)

    # We need to check the signature of the body as bytes
    # But use some specific fields from the body
    signed_by = request.headers.get("Epistula-Signed-By")
    signed_for = request.headers.get("Epistula-Signed-For")
    if signed_for != ss58:
        raise HTTPException(
            status_code=400, detail="Bad Request, message is not intended for self"
        )
    if signed_by not in hotkeys:
        raise HTTPException(status_code=401, detail="Signer not in metagraph")

    # If anything is returned here, we can throw
    body = await request.body()
    err = epistula_verify_v2(
        request.headers.get("Epistula-Request-Signature"),
        body,
        request.headers.get("Epistula-Timestamp"),
        request.headers.get("Epistula-Uuid"),
        signed_for,
        signed_by,
        now,
    )
    if err:
        bt.logging.error(err)
        raise HTTPException(status_code=400, detail=err)


def epistula_verify_v2(
    signature, body: bytes, timestamp, uuid, signed_for, signed_by, now
) -> Optional[Annotated[str, "Error Message"]]:
    if not isinstance(signature, str):
        return "Invalid Signature"
    timestamp = int(timestamp)
    if not isinstance(timestamp, int):
        return "Invalid Timestamp"
    if not isinstance(signed_by, str):
        return "Invalid Sender key"
    if not isinstance(signed_for, str):
        return "Invalid receiver key"
    if not isinstance(uuid, str):
        return "Invalid uuid"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    ALLOWED_DELTA_MS = 8000
    keypair = Keypair(ss58_address=signed_by)
    if timestamp + ALLOWED_DELTA_MS < now:
        return "Request is too stale"
    message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for}"
    verified = keypair.verify(message, signature)
    if not verified:
        return "Signature Mismatch"
    return None
