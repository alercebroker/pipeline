class GenericConsumerTest():
    component = None
    params = {}

    def test_consume(self):
        comp = self.component(self.params)
        for msj in comp.consume():
            self.assertIsInstance(msj, dict)
            comp.commit()
