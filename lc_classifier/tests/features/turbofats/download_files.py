from alerce.core import Alerce

alerce = Alerce()

oid = 'ZTF18aaiopei'
detections = alerce.query_detections(oid, format='pandas', sort='mjd')
detections = detections[['fid', 'mjd', 'magpsf_corr', 'sigmapsf_corr']].copy()
detections.dropna(inplace=True)
detections.drop_duplicates(['fid', 'mjd'], inplace=True)
detections['oid'] = oid
detections.set_index('oid', inplace=True)
detections.sort_values('mjd', inplace=True)
detections = detections.iloc[:239].copy()

detections.to_pickle(f'{oid}_detections.pkl')
