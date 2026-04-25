"""Corpus loading.

Two sources are provided:

- ``synthetic_corpus``: small, fast, with KNOWN topic changepoints. Useful
  as a smoke test and as a positive control for segmentation metrics.
- ``load_jsonl``: read conversations from a JSONL file with one
  ``{"messages": [...]}`` object per line. Compatible with ShareGPT
  exports and most chat-log dumps.
"""
import json
import random
from pathlib import Path


def load_jsonl(path, min_length=3):
    """Load conversations from a JSONL file.

    Each line is expected to be a JSON object with a ``"messages"`` (or
    ``"turns"``) key holding a list of strings or ``{"content": ...}``
    dicts. Returns a list of conversations (each a list of message
    strings). Conversations shorter than ``min_length`` are dropped.
    """
    convs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or obj.get("turns") or []
            texts = []
            for m in msgs:
                if isinstance(m, dict):
                    t = m.get("content") or m.get("text")
                else:
                    t = m
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            if len(texts) >= min_length:
                convs.append(texts)
    return convs


_TOPIC_TEMPLATES = {
    "weather": [
        "What's the weather like in {city} today?",
        "Is it going to rain in {city} this weekend?",
        "How hot does {city} get in the summer?",
        "Should I bring an umbrella to {city}?",
        "What's the temperature right now in {city}?",
        "Has {city} had unusual weather this year?",
    ],
    "cooking": [
        "How do I make a good {dish}?",
        "What goes well with {dish}?",
        "Can I substitute butter for oil in {dish}?",
        "What's a good cooking time for {dish}?",
        "How do I make {dish} less salty?",
        "Is {dish} a beginner-friendly recipe?",
    ],
    "programming": [
        "How do I sort a list in Python?",
        "What's the difference between a list and a tuple?",
        "How do I read a file in Python efficiently?",
        "What are decorators in Python?",
        "How do I install a package using pip?",
        "What's the right way to handle exceptions?",
    ],
    "travel": [
        "What's the best time of year to visit {city}?",
        "How do I get from the airport to downtown {city}?",
        "What are the must-see places in {city}?",
        "Is {city} safe for solo tourists?",
        "How much does a hotel cost per night in {city}?",
        "What's the food scene like in {city}?",
    ],
    "fitness": [
        "How often should I run per week?",
        "What's a good warmup before lifting?",
        "How do I prevent shin splints?",
        "Is it better to do cardio before or after weights?",
        "How long should a rest day be?",
        "What's a sustainable weekly mileage to start at?",
    ],
}

_CITIES = ["Tokyo", "Paris", "New York", "Cairo", "Sydney",
          "Cleveland", "Berlin", "Lisbon", "Bogota"]
_DISHES = ["risotto", "tacos", "curry", "pasta", "stir-fry",
          "lasagna", "ramen", "paella"]


def synthetic_corpus(n_conversations=20, seed=0,
                     min_segments=2, max_segments=3,
                     min_segment_length=3, max_segment_length=6):
    """Generate synthetic conversations with known topic structure.

    Each conversation consists of 2-3 randomly chosen topic segments
    concatenated. The boundaries between segments are returned alongside
    the conversations so that segmentation metrics can be evaluated
    against a known ground truth.

    Returns
    -------
    conversations : list of list of str
    changepoints  : list of list of int
        ``changepoints[i]`` are the indices of segment boundaries in
        ``conversations[i]`` (i.e., the first index of each segment after
        the first).
    """
    rng = random.Random(seed)

    def fill(template):
        return template.format(
            city=rng.choice(_CITIES),
            dish=rng.choice(_DISHES),
        )

    conversations = []
    changepoints = []
    topic_keys = list(_TOPIC_TEMPLATES.keys())

    for _ in range(n_conversations):
        n_segments = rng.randint(min_segments, max_segments)
        chosen_topics = rng.sample(topic_keys, n_segments)
        msgs = []
        cps = []
        for tk in chosen_topics:
            seg_len = rng.randint(min_segment_length, max_segment_length)
            seg = [fill(rng.choice(_TOPIC_TEMPLATES[tk])) for _ in range(seg_len)]
            if msgs:
                cps.append(len(msgs))
            msgs.extend(seg)
        conversations.append(msgs)
        changepoints.append(cps)

    return conversations, changepoints
