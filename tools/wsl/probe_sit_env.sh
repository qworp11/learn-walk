#!/bin/bash
# G1WholeBodyTracking: условия termination, спавн, пороги — что режет s9_sit
source /root/hlc-venv/bin/activate
python - <<'EOF'
import inspect, re, dataclasses
from mujoco_playground._src.rl import env_registry
e = env_registry.get("G1WholeBodyTracking")
print("entry attrs:", [a for a in dir(e) if not a.startswith("_")])
mod = inspect.getmodule(e.env_builder)
print("module:", mod.__file__)
src = inspect.getsource(mod)
print("---- строки terminate/healthy/height/z/fall ----")
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r"terminat|healthy|height|z_range|fall|bad_ref|body_pos_w\[" , line, re.I):
        print(f"{i}: {line.rstrip()}")
print("---- конфиг-классы ----")
for name, obj in vars(mod).items():
    if dataclasses.is_dataclass(obj) and isinstance(obj, type):
        print(f"class {name}:")
        for f in dataclasses.fields(obj):
            if f.default is not dataclasses.MISSING:
                d = f.default
            elif f.default_factory is not dataclasses.MISSING:
                d = f.default_factory
            else:
                d = "REQ"
            print(f"  {f.name} = {d!r}")
EOF
