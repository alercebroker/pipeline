from prometheus_client import Enum, Summary


class PrometheusMetrics:
    consumed_messages: Summary
    processed_messages: Summary
    execution_time: Summary
    step_state: Enum

    def __init__(self):
        self.consumed_messages = Summary(
            "consumed_messages", "Number of messages consumed"
        )
        self.processed_messages = Summary(
            "processed_messages", "Number of messages processed"
        )
        self.execution_time = Summary("execution_time", "Execution time of the batch")
        self.step_state = Enum(
            "step_state",
            "Current proccessing state of the step",
            states=[
                "pre_consume",
                "pre_execute",
                "execute",
                "post_execute",
                "pre_produce",
                "produce",
                "post_produce",
                "tear_down",
            ],
        )

    def tear_down(self):
        self.consumed_messages.remove()
        self.processed_messages.remove()
        self.execution_time.remove()
        self.step_state.remove()
