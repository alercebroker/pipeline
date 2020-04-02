class GenericProducerTest():
    component = None
    params = {}

    def test_produce(self):
        comp = self.component(self.params)
        msj = comp.produce('test')
        with open(self.params["FILE_PATH"], 'r') as outfile:
            self.assertEqual(outfile.read(),'"test"')
            
