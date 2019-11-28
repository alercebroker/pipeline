from apf.core.step import GenericStep

class {{step_name}}(GenericStep):
    """{{step_name}} Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    producer : GenericProducer
        Description of parameter `producer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    def __init__(self,consumer,producer = None, **step_args):
        super().__init__(consumer,producer, **step_args)

    def execute(self):
        while True:
            message = self.consumer.consume()
            ################################
            #   Here comes the Step Logic  #
            ################################
            print("Running ")

            if self.producer:
                self.producer.produce()
            self.consumer.commit(message)
