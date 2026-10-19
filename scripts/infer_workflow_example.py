"""One predeclared public offline-workflow request, separate from confirmation.
No gold, repair, retry, routing or reference fallback. Real completed manifest is
compatible with the existing workflow.verified_model_record identity checks.
"""
import argparse,hashlib,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=ROOT/'data/workflow_example'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--base',type=pathlib.Path,default=ROOT/'models/Qwen3-8B')
 parser.add_argument('--tokenizer',type=pathlib.Path,default=ROOT/'models/tokenizer')
 parser.add_argument('--candidate',type=pathlib.Path,default=ROOT/'models/r6')
 parser.add_argument('--device',type=int,default=0)
 parser.add_argument('--output-dir',type=pathlib.Path,required=True)
 args=parser.parse_args()
 BASE,OLD,CAND=args.base,args.tokenizer,args.candidate
 from verify_models import verify_paper_model
 verify_paper_model(BASE,OLD,CAND,'r6')
 args.output_dir.mkdir(parents=True,exist_ok=False)
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig,set_seed
 from peft import PeftModel
 case=json.loads((DATA/'public_case.json').read_text(encoding='utf8'))
 protocol=json.loads((DATA/'protocol.json').read_text(encoding='utf8'))
 assert sha(DATA/'public_case.json')==protocol['case_file_sha256'].lower()
 assert sha(CAND/'adapter_model.safetensors')==protocol['candidate_sha256'].lower()
 assert not (args.output_dir/'predictions.jsonl').exists()
 torch.set_num_threads(8);set_seed(20261010);device=f'cuda:{args.device}'
 tok=AutoTokenizer.from_pretrained(OLD,local_files_only=True);tok.padding_side='left'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 ids=tok.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 assert len(ids)<=2048
 model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True,device_map={'':device})
 model=PeftModel.from_pretrained(model,CAND,is_trainable=False);model.eval()
 x=torch.tensor([ids],device=device);mask=torch.ones_like(x)
 with torch.inference_mode():
  smoke=model(input_ids=x,attention_mask=mask,use_cache=False).logits[:,-1,:]
  assert bool(torch.isfinite(smoke).all().item());del smoke
 cfg=json.loads((BASE/'generation_config.json').read_text());eos=cfg.get('eos_token_id',tok.eos_token_id);eos=eos if isinstance(eos,list) else [eos];eos=sorted(set(eos+[tok.eos_token_id]))
 gen=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.0,encoder_repetition_penalty=1.0,length_penalty=1.0,min_length=0,min_new_tokens=0,no_repeat_ngram_size=0,early_stopping=False,temperature=1.0,top_p=1.0,top_k=50,max_new_tokens=480,eos_token_id=eos,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id,use_cache=True,return_dict_in_generate=False)
 spec={'model_kind':'single_model','base_revision':'b968826d9c46dd6066d109eabc6255188de91218','base_weight_sha256':json.loads((ROOT/'results/model_confirmation/model_identity.json').read_text())['base_weight_sha256'],'adapter_sha256':sha(CAND/'adapter_model.safetensors'),'adapter_config_sha256':sha(CAND/'adapter_config.json')}
 spec['model_fingerprint']=fp(spec)
 settings={'max_input_tokens':2048,'max_new_tokens':480,'enable_thinking':False,'generation':gen.to_dict(),'precision':'fp16','attention':'sdpa'}
 immutable={'source_type':'real_model','case_ids':[case['case_id']],'case_hashes':{case['case_id']:fp(case)},'input_sha256':sha(DATA/'public_case.json'),'models':{'R6_offline_single':spec},'settings':settings,'settings_fingerprint':fp(settings),'driver_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(DATA/'protocol.json'),'runtime':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'peft':__import__('peft').__version__},'gold_access':False,'reference_fallback':False}
 runfp=fp(immutable)
 torch.cuda.synchronize(device);t=time.perf_counter()
 with torch.inference_mode():out=model.generate(input_ids=x,attention_mask=mask,generation_config=gen,use_model_defaults=False)
 torch.cuda.synchronize(device);elapsed=time.perf_counter()-t
 ids=out[0,len(ids):].tolist();end=next((n+1 for n,v in enumerate(ids) if v in eos),len(ids));ids=ids[:end]
 record={'status':'completed','inference_backend':'transformers_real_model','model':'R6_offline_single','case_id':case['case_id'],'case_sha256':fp(case),'model_fingerprint':spec['model_fingerprint'],'run_fingerprint':runfp,'settings_fingerprint':fp(settings),'input_sha256':immutable['input_sha256'],'adapter_sha256':spec['adapter_sha256'],'raw_output':tok.decode(ids,skip_special_tokens=True),'generated_tokens':len(ids),'prompt_tokens':x.shape[1],'elapsed_seconds':elapsed,'finish_reason':'eos' if ids and ids[-1] in eos else 'length','forward_last_logits_finite':True}
 with (args.output_dir/'predictions.jsonl').open('x',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
 write(args.output_dir/'run_manifest.json',{**immutable,'status':'completed','run_fingerprint':runfp,'predictions_sha256':sha(args.output_dir/'predictions.jsonl')})
 print('completed_one_frozen_offline_request',flush=True)
if __name__=='__main__':main()
