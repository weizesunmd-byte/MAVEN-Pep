"""CPU real-tokenizer verification or finite fixed-budget R6 continuation.

Public portable worker for the original fixed-budget calculation. This worker
never evaluates benchmark questions. External assets are verified before use.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.metadata, json, math, os, random, sys, time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
def sha_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def save_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['validate','train'],required=True)
    ap.add_argument('--arm',choices=['A','B'],required=True)
    ap.add_argument('--data-dir',type=Path,required=True,help='Directory with exact domain_rows.jsonl, replay_rows.jsonl and slots.csv')
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--base-model',type=Path)
    ap.add_argument('--parent-adapter',type=Path)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    for name,expected in p['frozen_training_files_sha256'].items():
        if name=='replay_rows.jsonl' and args.arm=='A':continue
        candidate=args.data_dir/name
        if not candidate.is_file():raise FileNotFoundError(f'Required external training file: {candidate}')
        assert sha_file(candidate)==expected,('training file changed',name)
    for name,expected in p['tokenizer_files_sha256'].items():
        assert sha_file(args.tokenizer/name)==expected,('tokenizer mismatch',name)
    from maven_final.model_runtime import load_tokenizer,encode_training_example,model_training_loss
    from semantic_loss import semantic_reweight_example
    tokenizer=load_tokenizer(args.tokenizer)
    assert tokenizer.is_fast
    domains=read_rows(args.data_dir/'domain_rows.jsonl');replays=read_rows(args.data_dir/'replay_rows.jsonl') if args.arm=='B' else []
    with (args.data_dir/'slots.csv').open(newline='',encoding='utf-8') as f:slots=list(csv.DictReader(f))
    encoded=[];stats={'rows':0,'domain_rows':0,'replay_rows':0,'input_tokens':0,'supervised_tokens':0,'weighted_supervised_mass':0.0,'max_tokens':0,'all_prompt_masks_zero':True,'all_semantic_ids_labels_masks_unchanged':True}
    for slot in slots:
        is_replay=args.arm=='B' and int(slot['replay_index'])>=0
        raw=replays[int(slot['replay_index'])] if is_replay else domains[int(slot['domain_index'])]
        item=encode_training_example(raw,tokenizer,p['max_length'],p['action_loss_total_weight'])
        item,audit=semantic_reweight_example(raw,item,tokenizer,p)
        if is_replay:
            assert not audit['weighted']
            assert all(w==(0.0 if y==-100 else 1.0) for w,y in zip(item['loss_weights'],item['labels']))
        else:assert audit['weighted'] and audit['ids_labels_mask_unchanged']
        assert all(w==0.0 for w,y in zip(item['loss_weights'],item['labels']) if y==-100)
        stats['rows']+=1;stats['replay_rows' if is_replay else 'domain_rows']+=1
        stats['input_tokens']+=len(item['input_ids']);stats['supervised_tokens']+=sum(y!=-100 for y in item['labels'])
        stats['weighted_supervised_mass']+=sum(item['loss_weights']);stats['max_tokens']=max(stats['max_tokens'],len(item['input_ids']))
        encoded.append({k:item[k] for k in ('input_ids','attention_mask','labels','loss_weights')})
    assert stats['rows']==4096 and stats['replay_rows']==p['arms'][args.arm]['replay']
    assert stats['domain_rows']==p['arms'][args.arm]['domain']
    args.output_dir.mkdir(parents=True,exist_ok=False)
    stats.update(arm=args.arm,tokenizer='real_fast_offsets_assistant_only',versions={name:importlib.metadata.version(name) for name in ('transformers','tokenizers')})
    save_json(args.output_dir/'tokenizer_validation.json',stats)
    print('CPU_VALIDATED',json.dumps(stats),flush=True)
    if args.mode=='validate':return
    if args.base_model is None or args.parent_adapter is None:raise ValueError('Training requires base and parent')
    assert sha_file(args.parent_adapter/'adapter_model.safetensors')==p['parent_adapter_sha256']
    for name,expected in p['base_files_sha256'].items():
        assert sha_file(args.base_model/name)==expected,('base changed',name)
    import numpy as np
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(6);random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
    model=PeftModel.from_pretrained(model,args.parent_adapter,is_trainable=True,autocast_adapter_dtype=True)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
    trainable=[(n,t) for n,t in model.named_parameters() if t.requires_grad]
    assert trainable and all(t.dtype==torch.float32 for n,t in trainable)
    conf=model.peft_config['default']
    assert conf.r==16 and conf.lora_alpha==32 and conf.lora_dropout==.05
    assert set(conf.target_modules)==set(p['lora']['target_modules'])
    initial={n:t.detach().cpu().clone() for n,t in trainable}
    initial_hash=hashlib.sha256()
    for n,t in initial.items():initial_hash.update(n.encode());initial_hash.update(t.numpy().tobytes())
    optimizer=torch.optim.AdamW([t for n,t in trainable],lr=p['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    updates=[0]
    def count_step(optimizer,args,kwargs):updates[0]+=1
    optimizer.register_step_post_hook(count_step)
    environment={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','numpy')}
    environment.update(python=sys.version,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),trainable_parameters=sum(t.numel() for n,t in trainable),initial_trainable_sha256=initial_hash.hexdigest(),protocol_sha256=sha_file(HERE/'protocol.json'),worker_sha256=sha_file(Path(__file__)))
    save_json(args.output_dir/'environment.json',environment)
    try:
        with (args.output_dir/'optimization.csv').open('w',encoding='utf-8',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=['update','learning_rate','mean_micro_loss','gradient_norm','loss_scale','input_tokens','supervised_tokens','elapsed_seconds']);w.writeheader()
            for step in range(128):
                u=step+1
                # The zero endpoint is after the fixed budget, so all 128
                # observed optimizer updates use a strictly positive LR.
                factor=u/4 if u<=4 else .5*(1+math.cos(math.pi*(u-4)/125))
                lr=p['learning_rate']*factor
                for g in optimizer.param_groups:g['lr']=lr
                optimizer.zero_grad(set_to_none=True);losses=[];batch=encoded[step*32:(step+1)*32]
                for item in batch:
                    inputs={k:torch.tensor([v],dtype=torch.float32 if k=='loss_weights' else torch.long,device='cuda') for k,v in item.items()}
                    with torch.autocast('cuda',dtype=torch.float16):loss,_=model_training_loss(model,inputs)
                    if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss')
                    losses.append(float(loss.detach()));scaler.scale(loss/32).backward()
                scaler.unscale_(optimizer)
                if any(t.grad is not None and not bool(torch.isfinite(t.grad).all()) for n,t in trainable):raise RuntimeError('nonfinite gradients; no retry')
                gradient_norm=torch.nn.utils.clip_grad_norm_([t for n,t in trainable],1.0,error_if_nonfinite=True)
                before=updates[0];scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if updates[0]!=before+1 or scaler.get_scale()<scale:raise RuntimeError('GradScaler skipped update; failed budget')
                record=dict(update=updates[0],learning_rate=lr,mean_micro_loss=sum(losses)/32,gradient_norm=float(gradient_norm),loss_scale=scaler.get_scale(),input_tokens=sum(len(x['input_ids']) for x in batch),supervised_tokens=sum(sum(y!=-100 for y in x['labels']) for x in batch),elapsed_seconds=time.monotonic()-started)
                w.writerow(record);handle.flush();print('UPDATE',json.dumps(record),flush=True)
        assert updates[0]==128
        changed=0;max_delta=0.0;final_hash=hashlib.sha256()
        for n,t in trainable:
            final=t.detach().cpu();assert bool(torch.isfinite(final).all())
            delta=(final-initial[n]).abs();changed+=int((delta!=0).sum());max_delta=max(max_delta,float(delta.max()))
            final_hash.update(n.encode());final_hash.update(final.numpy().tobytes())
        assert changed>0 and final_hash.hexdigest()!=initial_hash.hexdigest()
        model.save_pretrained(args.output_dir/'adapter',safe_serialization=True)
        result=dict(status='completed_real_optimizer_updates',arm=args.arm,protocol_sha256=sha_file(HERE/'protocol.json'),parent_adapter_sha256=p['parent_adapter_sha256'],actual_optimizer_updates=updates[0],attempted_updates=128,skipped_updates=0,finite_gradient_steps=128,changed_trainable_elements=changed,max_absolute_parameter_delta=max_delta,initial_trainable_sha256=initial_hash.hexdigest(),final_trainable_sha256=final_hash.hexdigest(),adapter_sha256=sha_file(args.output_dir/'adapter/adapter_model.safetensors'),adapter_config_sha256=sha_file(args.output_dir/'adapter/adapter_config.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),wall_seconds=time.monotonic()-started,token_stats=stats)
        save_json(args.output_dir/'TRAINING_COMPLETE.json',result);print('COMPLETE',json.dumps(result),flush=True)
    except Exception as exc:
        save_json(args.output_dir/'TRAINING_FAILED.json',dict(status='failed_no_completed_adapter',reason=repr(exc),actual_optimizer_updates=updates[0],wall_seconds=time.monotonic()-started))
        raise

if __name__=='__main__':main()
