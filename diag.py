import requests

r = requests.post('http://localhost:8000/predict', json={
    'smiles': 'OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl',
    'top_n': 5,
    'run_ndealk': True,
    'run_reduction': True,
})
data = r.json()

stats = data['pipeline_stats']
print('BioTransformer:', stats['biotransformer_count'])
print('SyGMa:         ', stats['sygma_count'])
print('DL:            ', stats['dl_count'])
print('SMARTCyp:      ', stats['smartcyp_count'])
print('Total after dedup:', stats['total_after_dedup'])
print()
for m in data['metabolites']:
    print('Rank', m['rank'], '|', m['transformation_type'], '|',
          'score='+str(m['ensemble_score']), '|', m['responsible_enzyme'])
