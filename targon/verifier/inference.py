# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import math
import torch
import random
import typing
import string
import asyncio
import plotext as plt
import bittensor as bt

from targon import protocol
from targon.utils.prompt import create_prompt
from targon.constants import CHALLENGE_FAILURE_REWARD
from targon.verifier.uids import get_random_uids
from targon.verifier.reward import hashing_function
from targon.verifier.state import EventSchema


# get highest incentive axons from metagraph
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
    indices = torch.topk(metagraph.incentive, n).indices

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
    # axons_with_highest_incentives = [metagraph.axons[uid] for uid in uids_with_highest_incentives]
    # unique_ip_to_uid = {ip: uid for ip, uid in zip(ips, uids_with_highest_incentives)}
    # uids = list(unique_ip_to_uid.values())
    return uids_with_highest_incentives



def _filter_verified_responses(uids, responses):
    """
    Filters out responses that have not been verified.

    Args:
    - uids (list): A list of user IDs.
    - responses (list): A list of tuples containing verification status and response.

    Returns:
    - tuple: Two tuples, one containing filtered user IDs and the other containing their corresponding responses.
    """
    not_none_responses = [
        (uid, response[0])
        for (uid, (verified, response)) in zip(uids, responses)
        if verified != None
    ]

    if len(not_none_responses) == 0:
        return (), ()

    uids, responses = zip(*not_none_responses)
    return uids, responses


def check_tokens(self, prover_output, ground_truth_output):
    # Tokenize the prover output and the ground truth output
    prover_tokenized = self.embedding_tokenizer(
        prover_output, return_tensors="pt", padding=True, truncation=True
    )
    ground_truth_tokenized = self.embedding_tokenizer(
        ground_truth_output, return_tensors="pt", padding=True, truncation=True
    )

    # Compare the list of tokens
    prover_tokens = prover_tokenized["input_ids"]
    ground_truth_tokens = ground_truth_tokenized["input_ids"]

    bt.logging.info(prover_tokens)
    bt.logging.info(ground_truth_tokens)

    # convert to list
    prover_tokens = prover_tokens[0].tolist()
    ground_truth_tokens = ground_truth_tokens[0].tolist()

    # make the tokenized outputs the same length, perferring the ground truth output length
    if len(prover_tokens) > len(ground_truth_tokens):
        prover_tokens = prover_tokens[: len(ground_truth_tokens)]
    elif len(prover_tokens) < len(ground_truth_tokens):
        return 0

    # Calculate the score from 0 to 1
    score = sum([1 for token in prover_tokens if token in ground_truth_tokens]) / len(
        prover_tokens
    )

    bt.logging.info(score)
    return score


def verify(self, prover_output, ground_truth_output, prover_ss58):
    """
    Verifies the prover's output against the ground truth output.

    Args:
    - self: Reference to the current instance of the class.
    - prover_output (str): The output provided by the prover.
    - ground_truth_output (str): The expected output.
    - prover_ss58 (str): The prover's SS58 address.

    Returns:
    - bool: True if the outputs match or if the embedding check passes, False otherwise.
    """
    prover_output_hash = hashing_function(prover_output)
    ground_truth_hash = hashing_function(ground_truth_output)

    if not prover_output_hash == ground_truth_hash:
        bt.logging.debug(
            f"Output hash {prover_output_hash} does not match ground truth hash {ground_truth_hash}"
        )

        # check how t

        # return asyncio.run(embedding_check( self, prover_output, ground_truth_output, prover_ss58 ))
        return check_tokens(self, prover_output, ground_truth_output)

    bt.logging.debug(
        f"Output hash {prover_output_hash} matches ground truth hash {ground_truth_hash}"
    )
    return True

async def api_chat_completions(
    self,
    prompt: str,
    sampling_params: protocol.InferenceSamplingParams,
) -> typing.Tuple[bool, protocol.Inference]:
    """
    Handles a inference sent to a prover and verifies the response.

    Returns:
    - Tuple[bool, protocol.Inference]: A tuple containing the verification result and the inference.
    """
    synapse = protocol.Inference(
        sources=[],
        query=prompt,
        sampling_params=sampling_params,
    )

    start_time = time.time()
    token_count = 0
    uid = select_highest_n_peers(1)[0]
    async for token in await self.dendrite(
        self.metagraph.axons[uid],
        synapse,
        deserialize=False,
        timeout=self.config.neuron.timeout,
        streaming=True,
    ):
        if isinstance(token, list):
            yield token[0]
        elif isinstance(token, str):
            yield token
        token_count += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = token_count / elapsed_time
    bt.logging.info(f"Token generation rate: {tokens_per_second} tokens/second")


async def handle_inference(
    self,
    uid: int,
    private_input: typing.Dict,
    ground_truth_output: str,
    sampling_params: protocol.InferenceSamplingParams,
) -> typing.Tuple[bool, protocol.Inference]:
    """
    Handles a inference sent to a prover and verifies the response.

    Parameters:
    - uid (int): The UID of the prover being inferenced.

    Returns:
    - Tuple[bool, protocol.Inference]: A tuple containing the verification result and the inference.
    """
    if not self.config.mock:
        synapse = protocol.Inference(
            sources=[private_input["sources"]],
            query=private_input["query"],
            sampling_params=sampling_params,
        )

        response_tokens = []

        try:
            start_time = time.time()
            token_count = 0
            async for token in await self.dendrite(
                self.metagraph.axons[uid],
                synapse,
                deserialize=False,
                timeout=self.config.neuron.timeout,
                streaming=True,
            ):
                if isinstance(token, list):
                    response_tokens.append(token[0])
                    token_count += 1
                elif isinstance(token, str):
                    response_tokens.append(token)
                    token_count += 1
                else:
                    output_synapse = token
            
            end_time = time.time()
            output = "".join(response_tokens)


            elapsed_time = end_time - start_time
            tokens_per_second = token_count / (elapsed_time if elapsed_time > 0 else 1000)
            bt.logging.info(f"Token generation rate: {tokens_per_second} tokens/second")

        except Exception as e:
            bt.logging.error(f"Error in handle_inference: {e}")
            return False, (synapse, uid, 0)

        # output_encoded = output.encode('utf-8')
        if output is not None:
            start_time = time.time()
            output_normalized = output.replace("\r\n", "\n")
            output_cleaned = " ".join(output_normalized.split())
            end_time = time.time()
            elapsed_time = end_time - start_time
            bt.logging.info(f"Output normalization rate: {elapsed_time} seconds")
            bt.logging.info(f"Output normalization rate: {tokens_per_second} tokens/second")

            bt.logging.debug("output", output_cleaned)
            verified = verify(
                self, output_cleaned, ground_truth_output, self.metagraph.hotkeys[uid]
            )

        else:
            verified = False

        output_dict = (output_synapse, uid, tokens_per_second)
        return verified, output_dict

    else:
        prompt = create_prompt(private_input)

        synapse = protocol.Inference(
            sources=[private_input["sources"]],
            query=private_input["query"],
            sampling_params=sampling_params,
        )

        response_tokens = []

        start_time = time.time()
        token_count = 0
        async for token in await self.client.text_generation(
            prompt,
            best_of=sampling_params.best_of,
            max_new_tokens=sampling_params.max_new_tokens,
            seed=sampling_params.seed,
            do_sample=sampling_params.do_sample,
            repetition_penalty=sampling_params.repetition_penalty,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
            truncate=sampling_params.truncate,
            typical_p=sampling_params.typical_p,
            watermark=sampling_params.watermark,
            details=False,
            stream=True,
        ):
            response_tokens.append(token)
            token_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = token_count / elapsed_time


        response = "".join(response_tokens)

        synapse.completion = response

        verified = verify(
            self, response, ground_truth_output, self.metagraph.hotkeys[uid]
        )

        output_dict = (synapse, uid, tokens_per_second)
        return verified, output_dict


async def inference_data(self):
    """
    Orchestrates the inference process, from fetching inference data to applying rewards based on the verification results.

    This function performs several key steps:
    1. Fetches inference data from a configured URL.
    2. Generates a ground truth output using the inference data.
    3. Selects a set of UIDs (user identifiers) to inference.
    4. Sends the inference to each selected UID and collects their responses.
    5. Verifies the responses against the ground truth output.
    6. Applies rewards or penalties based on the verification results.
    7. Updates the event schema with the results of the inference.

    The function handles both real and mock inferences, allowing for testing without actual data.

    Returns:
    - EventSchema: An object containing detailed information about the inference, including which UIDs were successful, the rewards applied, and other metadata.
    """

    def remove_indices_from_tensor(tensor, indices_to_remove):
        # Sort indices in descending order to avoid index out of range error
        sorted_indices = sorted(indices_to_remove, reverse=True)
        for index in sorted_indices:
            tensor = torch.cat([tensor[:index], tensor[index + 1 :]])
        return tensor

    # --- Create the event
    event = EventSchema(
        task_name="inference",
        successful=[],
        completion_times=[],
        task_status_messages=[],
        task_status_codes=[],
        block=self.subtensor.get_current_block(),
        uids=[],
        step_length=0.0,
        best_uid=-1,
        best_hotkey="",
        rewards=[],
        set_weights=None,
        moving_averaged_scores=None,
    )


    bt.logging.info("Generating challenge data")
    challenge_data = {
        "query": "".join(random.choice(string.ascii_letters) for _ in range(12)),
        "sources": "".join(random.choice(string.ascii_letters) for _ in range(12)),
    }
    prompt = create_prompt(challenge_data)

    bt.logging.info("prompt created")
    seed = random.randint(10000, 10000000)

    sampling_params = protocol.InferenceSamplingParams(seed=seed)

    ground_truth_tokens = []


    start_time = time.time()
    async for token in await self.client.text_generation(
        prompt,
        best_of=sampling_params.best_of,
        max_new_tokens=sampling_params.max_new_tokens,
        seed=sampling_params.seed,
        do_sample=sampling_params.do_sample,
        repetition_penalty=sampling_params.repetition_penalty,
        temperature=sampling_params.temperature,
        top_k=sampling_params.top_k,
        top_p=sampling_params.top_p,
        truncate=sampling_params.truncate,
        typical_p=sampling_params.typical_p,
        watermark=sampling_params.watermark,
        details=False,
        stream=True,
    ):
        ground_truth_tokens.append(token)
    


    ground_truth_output = "".join(ground_truth_tokens)

    # ground_truth_output_encoded = ground_truth_output.encode('utf-8')
    ground_truth_output_normalized = ground_truth_output.replace("\r\n", "\n")
    ground_truth_output_cleaned = " ".join(ground_truth_output_normalized.split())

    # --- Get the uids to query
    tasks = []
    # uids = await get_tiered_uids( self, k=self.config.neuron.sample_size )
    uids = get_random_uids(self, k=self.config.neuron.sample_size)

    bt.logging.debug(f"inference uids {uids}")
    
    responses = []
    for uid in uids:
        tasks.append(
            asyncio.create_task(
                handle_inference(
                    self,
                    uid,
                    challenge_data,
                    ground_truth_output_cleaned,
                    sampling_params,
                )
            )
        )
    responses = await asyncio.gather(*tasks)

    # Create a list of tuples (uid, tokens_per_second) for sorting
    uid_tokens_pairs = [(uid, tokens_per_second) for _, (_, uid, tokens_per_second) in responses]

    # Initialize or update moving averages dictionary
    if not hasattr(self, 'moving_averages'):
        self.moving_averages = {uid: 0 for uid, _ in uid_tokens_pairs}
    
    alpha = 0.1  # Smoothing factor for moving average, can be adjusted
    for uid, tokens_per_second in uid_tokens_pairs:
        if uid in self.moving_averages:
            self.moving_averages[uid] = alpha * tokens_per_second + (1 - alpha) * self.moving_averages[uid]
        else:
            self.moving_averages[uid] = tokens_per_second

    # Sort the list by tokens_per_second in descending order
    sorted_uid_tokens_pairs = sorted(uid_tokens_pairs, key=lambda x: x[1])

    # Extract tokens_per_second for plotting
    tokens_per_second_sorted = [tokens_per_second for _, tokens_per_second in sorted_uid_tokens_pairs]

<<<<<<< HEAD
    # Initialize or update moving averages for tokens statistics
    if not hasattr(self, 'moving_tokens_stats'):
        self.moving_tokens_stats = {
            'max_tokens': 0,
            'min_tokens': float('inf'),
            'range_tokens': 0,
            'avg_tokens': 0
        }

    current_max_tokens = max(tokens_per_second_sorted)
    current_min_tokens = min(tokens_per_second_sorted)
    current_range_tokens = current_max_tokens - current_min_tokens
    current_avg_tokens = sum(tokens_per_second_sorted) / len(tokens_per_second_sorted)

    # Update moving averages for tokens statistics
    self.moving_tokens_stats['max_tokens'] = alpha * current_max_tokens + (1 - alpha) * self.moving_tokens_stats['max_tokens']
    self.moving_tokens_stats['min_tokens'] = alpha * current_min_tokens + (1 - alpha) * self.moving_tokens_stats['min_tokens']
    self.moving_tokens_stats['range_tokens'] = alpha * current_range_tokens + (1 - alpha) * self.moving_tokens_stats['range_tokens']
    self.moving_tokens_stats['avg_tokens'] = alpha * current_avg_tokens + (1 - alpha) * self.moving_tokens_stats['avg_tokens']
=======
    y = plt.scatter(tokens_per_second_sorted, color='red')  # Reduced marker size for a smaller plot
    plt.title('Sorted Tokens per Second')
    plt.xlabel('UID Index (sorted)')
    plt.ylabel('Tokens per Second')
    plt.plotsize(100, 20)
    plt.show()
    plt.clf()  # Clear the plot after showing
>>>>>>> b27a27d (rewrite in progress)

    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(
        self.device
    )

<<<<<<< HEAD
    # Calculate rewards based on the difference between the highest and lowest tokens_per_second
    if self.moving_tokens_stats['range_tokens'] > 0:
        for i, (_, tokens_per_second) in enumerate(sorted_uid_tokens_pairs):
            normalized_difference = (tokens_per_second - self.moving_tokens_stats['avg_tokens']) / self.moving_tokens_stats['range_tokens']
            reward_multiplier = math.exp(normalized_difference * 10)  # Scale the difference to enhance reward disparity
            rewards[i] = reward_multiplier * tokens_per_second
    else:
        rewards.fill_(1)  # Avoid division by zero if all tokens_per_second are the same
=======
    remove_reward_idxs = []
    for i, (uid, tokens_per_second) in enumerate(sorted_uid_tokens_pairs):
        # Find the response associated with the current uid
        response = next(res for _, (res, res_uid, _) in responses if res_uid == uid)
        verified = next(ver for ver, (_, res_uid, _) in responses if res_uid == uid)

        bt.logging.trace(
            f"Inference iteration {i} uid {uid} response {str(response.completion if not self.config.mock else response)}"
        )

        hotkey = self.metagraph.hotkeys[uid]

    # Calculate mean, median, and mode of moving averages
    moving_averages_values = list(self.moving_averages.values())
    if moving_averages_values:
        mean_value = sum(moving_averages_values) / len(moving_averages_values)
        median_value = sorted(moving_averages_values)[len(moving_averages_values) // 2]
        mode_value = max(set(moving_averages_values), key=moving_averages_values.count)

        bt.logging.debug(f"Mean of moving averages: {mean_value}")
        bt.logging.debug(f"Median of moving averages: {median_value}")
        bt.logging.debug(f"Mode of moving averages: {mode_value}")
    else:
        bt.logging.info("No moving averages data available to calculate mean, median, and mode.")
    return event
>>>>>>> b27a27d (rewrite in progress)

    # Initialize or update moving average for rewards
    if not hasattr(self, 'moving_rewards'):
        self.moving_rewards = torch.zeros(len(256), dtype=torch.float32).to(self.device)

    for i, reward in enumerate(rewards):
        self.moving_rewards[i] = alpha * reward + (1 - alpha) * self.moving_rewards[i]

    # Print the highest UID and its corresponding tokens_per_second and reward score
    highest_uid, highest_tokens_per_second = sorted_uid_tokens_pairs[-1]
    highest_reward = rewards[0]
    print(f"Highest UID: {highest_uid}, Tokens/Second: {highest_tokens_per_second}, Reward: {highest_reward}")

    # Plot moving average of rewards
    y = plt.scatter(self.moving_rewards.numpy(), color='red')  # Reduced marker size for a smaller plot
    plt.title('Sorted Tokens per Second')
    plt.xlabel('UID (sorted)')
    plt.ylabel('Reward Score')
    plt.plotsize(100, 20)
    plt.show()
    plt.clf()  # Clear the plot after showing


