class GenericProducerTest():
    component = None
    params = {}

    def test_produce(self):
        comp = self.component(self.params)
        msj = comp.produce("test")
        self.assertIsInstance(msj, str)
