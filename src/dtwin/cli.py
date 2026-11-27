import argparse
import json
from pathlib import Path
from .pipeline import load_config,bronze_silver_gold,write_json
from .analysis import analyze

def main():
    parser=argparse.ArgumentParser(description='Longitudinal diabetes digital twin benchmark')
    parser.add_argument('command',choices=['run','check','report'])
    parser.add_argument('--env',choices=['dev','test','prod'],default='dev')
    args=parser.parse_args()
    cfg=load_config(args.env)
    if args.command=='run':
        dest,data,truth,qc=bronze_silver_gold(cfg)
        diag,methods=analyze(dest,data,truth,cfg['seed'],cfg['max_weight'])
        write_json(dest/'run_manifest.json',{'environment':args.env,'seed':cfg['seed'],'patients':cfg['n_patients'],
                     'method_count':len(methods),'quality_passed':qc['passed']})
        print('RUN COMPLETE:',args.env,'\n',methods.to_string(index=False))
        print('Diagnostics:',json.dumps({k:v for k,v in diag.items() if k!='warnings'},indent=2))
    else:
        dest=Path(cfg['artifact_dir'])
        if not (dest/'run_manifest.json').exists():raise SystemExit('No artifacts. Run pipeline first.')
        if args.command=='check':
            quality=json.loads((dest/'quality.json').read_text())
            assert quality['passed'] and quality['failures']==[]
            for filename in ['method_comparison.csv','diagnostics.json','twin_trajectories.csv','counterfactual_evaluation.csv']:
                assert (dest/filename).exists(),filename
            print('QUALITY CHECK PASSED:',args.env)
        else:print((dest/'diagnostics.json').read_text())

if __name__=='__main__':main()
