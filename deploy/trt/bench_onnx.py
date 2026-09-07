import sys, time, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,"/data4/src/shunsuke/MICCAI2026/RARE/submit/v001_seg_aux")
import model.predictor as P
P.DINOV3_REPO_PATH="/data4/src/shunsuke/RARE/dinov3"
res=Path("/data4/src/shunsuke/MICCAI2026/RARE/submit/v001_seg_aux/resources")
man=json.load(open(res/"manifest.json"))["members"]
spec=next(m for m in man if m["config"]=="resnext101_swsl_seg")
size=int(spec["image_size"]); dev="cuda"
model=P.build_member(spec,res)
sd=torch.load(res/spec["weights"], map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=not spec.get("partial",False)); model.to(dev).eval()

B=32
x=torch.randn(B,3,size,size,device=dev)
class Cls(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,v): return s.m.forward_cls(v)
wrap=Cls(model).eval()

def bench(fn, n=12, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n

with torch.no_grad():
    mh=wrap.half(); xh=x.half()
    ref=mh(xh).float().cpu().numpy()
    t_pt=bench(lambda: mh(xh))
print(f"  PyTorch fp16      {t_pt*1000:7.1f} ms / batch{B}")

onnx_p="/tmp/m.onnx"
with torch.no_grad():
    torch.onnx.export(wrap.float(), x.float(), onnx_p, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x":{0:"b"},"y":{0:"b"}}, opset_version=17)
import onnx
mo=onnx.load(onnx_p); mo.ir_version=10; onnx.save(mo,onnx_p)
import onnxruntime as ort
for prov in (["CUDAExecutionProvider"], ["TensorrtExecutionProvider","CUDAExecutionProvider"]):
    try:
        so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s=ort.InferenceSession(onnx_p, so, providers=prov)
        used=s.get_providers()
        if prov[0] not in used:
            print(f"  {prov[0]:24s} NOT USED -> fell back to {used}"); continue
        xn=x.float().cpu().numpy()
        f=lambda: s.run(None, {"x":xn})
        for _ in range(4): f()
        t=time.perf_counter()
        for _ in range(12): f()
        t_on=(time.perf_counter()-t)/12
        out=s.run(None,{"x":xn})[0].astype(np.float32)
        print(f"  {prov[0]:24s} {t_on*1000:7.1f} ms  ({t_pt/t_on:.2f}x vs PyTorch fp16)  "
              f"max|Δlogit| {np.abs(out.ravel()-ref.ravel()).max():.4f}  providers={used[:1]}")
    except Exception as e:
        print(f"  {prov[0]:24s} FAILED {type(e).__name__}: {str(e)[:110]}")
