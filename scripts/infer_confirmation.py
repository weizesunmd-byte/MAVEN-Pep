"""Regenerate confirmation first answers from the released, verified checkpoints.
The original greedy decoding calculation is retained. Generation never reads gold.
"""
import argparse, hashlib, json, pathlib, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'data/confirmation'
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in pathlib.Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def infer(args):
 from verify_models import verify_paper_model
 adapter=args.old_adapter if args.model=='old' else args.candidate if args.model=='candidate' else None
 component={'old':'r1','candidate':'r6','base':None}[args.model]
 verify_paper_model(args.base,args.tokenizer,adapter,component)
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig,set_seed
 from peft import PeftModel
 cases=rows(BUNDLE/'public_inputs.jsonl')
 args.output_dir.mkdir(parents=True,exist_ok=True)
 out=args.output_dir/f'{args.model}.jsonl'
 if out.exists(): raise RuntimeError('Never overwrite first answers')
 tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True);tok.padding_side='left'
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 set_seed(20261010);torch.set_num_threads(8)
 model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True,device_map={'':f'cuda:{args.device}'})
 if adapter: model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
 model.eval()
 finite=[True]
 def check_finite(module,inputs,output):
  if hasattr(output,'logits'): finite[0]=finite[0] and bool(torch.isfinite(output.logits[:,-1,:]).all().item())
 hook=model.register_forward_hook(check_finite)
 cfg=json.loads((pathlib.Path(args.base)/'generation_config.json').read_text())
 eos=cfg.get('eos_token_id',tok.eos_token_id);eos=eos if isinstance(eos,list) else [eos]
 eos=sorted(set(eos+[tok.eos_token_id]))
 gen=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.0,encoder_repetition_penalty=1.0,length_penalty=1.0,min_length=0,min_new_tokens=0,no_repeat_ngram_size=0,early_stopping=False,temperature=1.0,top_p=1.0,top_k=50,max_new_tokens=480,eos_token_id=eos,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id,use_cache=True,return_dict_in_generate=False)
 provenance={'model':args.model,'base_config_sha256':sha(pathlib.Path(args.base)/'config.json'),'adapter_sha256':sha(pathlib.Path(adapter)/'adapter_model.safetensors') if adapter else None,'public_sha256':sha(BUNDLE/'public_inputs.jsonl'),'script_sha256':sha(__file__),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'peft':__import__('peft').__version__,'generation':gen.to_dict(),'device':torch.cuda.get_device_name(args.device),'gold_access':False}
 (args.output_dir/f'provenance_{args.model}.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
 with out.open('x',encoding='utf8') as stream:
  for start in range(0,len(cases),4):
   batch=cases[start:start+4]
   texts=[tok.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for c in batch]
   tokens=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False)
   lengths=tokens['attention_mask'].sum(1).tolist()
   if max(lengths)>2048: raise RuntimeError('Input exceeds frozen budget')
   tokens={k:v.to(f'cuda:{args.device}') for k,v in tokens.items()};width=tokens['input_ids'].shape[1]
   torch.cuda.synchronize();t=time.perf_counter()
   with torch.inference_mode(): output=model.generate(**tokens,generation_config=gen,use_model_defaults=False)
   torch.cuda.synchronize();elapsed=time.perf_counter()-t
   if not finite[0]: raise RuntimeError('Nonfinite generation logits; do not publish answers')
   for i,c in enumerate(batch):
    generated=output[i,width:].tolist();end=next((n+1 for n,x in enumerate(generated) if x in eos),len(generated));generated=generated[:end]
    rec={'model':args.model,'case_id':c['case_id'],'raw_output':tok.decode(generated,skip_special_tokens=True),'prompt_tokens':lengths[i],'generated_tokens':len(generated),'batch_elapsed_seconds':elapsed,'finish_reason':'eos' if generated and generated[-1] in eos else 'length','input_sha256':hashlib.sha256(json.dumps(c,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
    stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush()
   print(json.dumps({'model':args.model,'completed':start+len(batch),'total':48}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',choices=['base','old','candidate'],required=True)
 p.add_argument('--base',type=pathlib.Path,default=ROOT/'models/Qwen3-8B')
 p.add_argument('--old-adapter',type=pathlib.Path,default=ROOT/'models/r1')
 p.add_argument('--candidate',type=pathlib.Path,default=ROOT/'models/r6')
 p.add_argument('--tokenizer',type=pathlib.Path,default=ROOT/'models/tokenizer')
 p.add_argument('--device',type=int,default=0)
 p.add_argument('--output-dir',type=pathlib.Path,required=True)
 a=p.parse_args()
 if a.model=='candidate' and a.candidate is None:p.error('--candidate is required for candidate inference')
 infer(a)
