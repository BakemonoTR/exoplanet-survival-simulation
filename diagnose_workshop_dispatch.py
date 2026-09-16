"""Read-only blocker accounting; isolated fixed-seed short mission."""
import argparse, json, logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import rl_learning_experiment as experiment

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--ticks',type=int,default=5000)
args=parser.parse_args()
logging.disable(logging.CRITICAL)
experiment.vector_store._use_tfidf_fallback = True
counts=Counter(); examples=[]
original=experiment.SimulationEngine._run_tick

def audited(self):
    result=original(self)
    for cycle in self._manufacturing_cycles.values():
        if not cycle.get('output_ready') or cycle.get('unloading'): continue
        recipe=self._get_recipe(cycle['output'])
        prefix=cycle['machine_type']+':'
        counts[prefix+'waiting']+=1
        fit=[]; available=[]
        for a in self.agents:
            distance=max(abs(a.x-cycle['x']),abs(a.y-cycle['y']))
            if self._manufacturing_operator_is_fit(a,recipe,duty_ticks=1):
                fit.append(a)
                action=a.action.action_type
                counts[prefix+'fit_action:'+str(action)]+=1
                if distance<=2: counts[prefix+'fit_near_action:'+str(action)]+=1
                if a.action.ticks_remaining<=0 or action in (None,'','idle'):
                    available.append(a)
        counts[prefix+('no_fit_operator' if not fit else 'fit_but_busy' if not available else 'available_operator')]+=1
        owner=next((a for a in self.agents if a.id==cycle.get('route_operator_id')),None)
        if owner:
            counts[prefix+'reserved_owner_action:'+str(owner.action.action_type)]+=1
        if len(examples)<12 and fit and all(max(abs(a.x-cycle['x']),abs(a.y-cycle['y']))>2 for a in fit):
            examples.append({'tick':self.current_tick,'machine':cycle['machine_id'],'wait':self.current_tick-cycle.get('process_completed_tick',self.current_tick),'crew':[{'id':a.id,'position':[a.x,a.y],'action':a.action.action_type,'target':a.action.target,'energy':round(a.needs.energy,1)} for a in fit]})
    return result

experiment.SimulationEngine._run_tick=audited
experiment.run_episode(SimpleNamespace(output=args.output,seed=42,planet='kepler-442b',ticks=args.ticks,carry_tactical=False),'probe')
(args.output/'blockers.json').write_text(json.dumps({'counts':dict(counts),'examples':examples},indent=2,default=str),encoding='utf-8')
print(json.dumps(dict(counts)),flush=True)
