#!/usr/bin/env python3


def get_sid(detections: list):
    return list(set([det["sid"] for det in detections]))
