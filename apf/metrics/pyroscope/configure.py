import pyroscope

def init_pyroscope(app_name: str, server: str):
    pyroscope.configure(application_name=app_name, server_address=server)