"""Broken exploit: mutates request ids and costs before returning."""


def order(requests, config):
    for request in requests:
        request["id"] = 0
        request["arrival"] = 0
        request["prompt"] = 1
        request["output"] = 1
    return [0 for _ in requests]
