class GenericProducerTest():
    component = None
    params = {}

    def test_produce(self):
        comp = self.component(self.params)
        msj = comp.produce('test')
        with open(self.params["FILE_PATH"], 'r') as outfile:
            print(outfile)
            self.assertEquals(outfile.read(),'"test"')
            
