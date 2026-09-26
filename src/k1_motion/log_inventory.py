"""Decode recovered ROS MCAP payloads at their actual recorded interface layer."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def inventory_mcap(paths, output):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    result = {
        "source_kind": "archived_ros_commands_and_state",
        "raw_mocopi": False,
        "state_origin": "physical_vs_simulated_unverified",
        "bags": [],
    }
    for path in paths:
        topics = defaultdict(
            lambda: {
                "messages": 0,
                "decode_errors": 0,
                "nonfinite_messages": 0,
                "first_log_time_ns": None,
                "last_log_time_ns": None,
                "intervals": [],
            }
        )
        with Path(path).open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            # Iterate raw messages so one unsupported schema cannot hide other topics.
            factory = DecoderFactory()
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                row = topics[channel.topic]
                row["schema"] = schema.name if schema else None
                row["messages"] += 1
                if row["last_log_time_ns"] is not None:
                    row["intervals"].append((message.log_time - row["last_log_time_ns"]) * 1e-9)
                else:
                    row["first_log_time_ns"] = message.log_time
                row["last_log_time_ns"] = message.log_time
                try:
                    decoder = factory.decoder_for(channel.message_encoding, schema)
                    if decoder is None:
                        raise ValueError("No ROS2 decoder for schema")
                    decoded = decoder(message.data)
                    plain = _plain(decoded)
                    if isinstance(plain, dict) and isinstance(plain.get("data"), str):
                        try:
                            plain["decoded_json"] = json.loads(plain["data"])
                        except (ValueError, TypeError):
                            pass
                    if "sample" not in row:
                        row["sample"] = plain
                    numbers = list(_numbers(plain))
                    if numbers and not np.isfinite(numbers).all():
                        row["nonfinite_messages"] += 1
                except Exception as exc:
                    row["decode_errors"] += 1
                    row.setdefault("first_error", f"{type(exc).__name__}: {exc}")
        for row in topics.values():
            intervals = row.pop("intervals")
            row["publication_interval_p50_ms"] = float(np.median(intervals) * 1000) if intervals else None
            row["publication_interval_p99_ms"] = (
                float(np.percentile(intervals, 99) * 1000) if intervals else None
            )
        result["bags"].append({"path": str(path), "topics": dict(topics)})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "__slots__"):
        return {name.lstrip("_"): _plain(getattr(value, name)) for name in value.__slots__}
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items()}
    return str(value)


def _numbers(value):
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _numbers(child)
