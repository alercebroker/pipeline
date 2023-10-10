#!/usr/bin/env python3


def test(docker_compose_file, pytestconfig):
    print(docker_compose_file)
    print(pytestconfig.rootdir)
    assert False
