import pandas as pd

from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from cds_xmatch_client import XmatchClient

import pandas as pd
import logging


class XmatchStep(GenericStep):

	def __init__(self,consumer = None, config = None,level = logging.INFO,**step_args):
		super().__init__(consumer,config=config, level=level)

		self.xmatch_config = config['XMATCH_CONFIG']
		self.xmatch_client = XmatchClient()
		self.producer = KafkaProducer(config["PRODUCER_CONFIG"])

	def _extract_coordinates(self, message: dict ):
		record = {
			'oid' : message['objectId'],
			'ra' : message['candidate']['ra'],
			'dec': message['candidate']['dec']
		}
		return record

	def _format_result(self, input,result):

		messages = []
		#objects without xmatch
		without_result = input[ ~input['oid_in'].isin(result['oid_in']) ]['oid_in'].values
		for oid in without_result:
			message = { 'oid' : oid, 'result' : None }
			messages.append(message)

		#results from allwise
		for idx,row in result.iterrows():
			oid =  row['oid_in']
			columns = dict(row)
			del columns['oid_in']
			message = {
						'oid': oid,
						'result' :  {
										'allwise' : columns
						}
			}
			messages.append(message)

		return messages

	def _produce(self,messages):
		for message in messages:
			self.producer.produce(message)

	def execute(self,messages):
		array = []
		for m in messages:
			record = self._extract_coordinates(m)
			array.append(record)
	
		df = pd.DataFrame(array,columns=['oid','ra','dec'])

		#xmatch
		catalog = self.xmatch_config['CATALOG']
		catalog_alias = catalog['name']
		columns = catalog['columns']
		radius = 1
		selection = 'best'
		input_type  = 'pandas'
		output_type = 'pandas'

		result = self.xmatch_client.execute(
							df,
							input_type,
							catalog_alias,
							columns,
							selection,
							output_type,
							radius
		)

		messages = self._format_result(df,result)

		self._produce(messages)

