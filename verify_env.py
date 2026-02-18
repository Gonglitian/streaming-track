"""Verify the streaming-track conda environment meets all acceptance criteria."""
import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  [PASS] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print(f"Python: {sys.version}")
print()

results = []

# Acceptance criteria
results.append(check("torch.cuda.is_available()",
    lambda: (__import__('torch'), __import__('torch').cuda.is_available())[1]))

results.append(check("import mujoco",
    lambda: __import__('mujoco').__version__))

results.append(check("import smplx",
    lambda: (__import__('smplx'), "OK")[1]))

results.append(check("import ultralytics",
    lambda: __import__('ultralytics').__version__))

results.append(check("import mink",
    lambda: (__import__('mink'), "OK")[1]))

# Additional key packages
print()
print("Additional packages:")
results.append(check("pytorch3d",
    lambda: __import__('pytorch3d').__version__))
results.append(check("lightning",
    lambda: __import__('lightning').__version__))
results.append(check("timm",
    lambda: __import__('timm').__version__))
results.append(check("hydra",
    lambda: __import__('hydra').__version__))
results.append(check("qpsolvers",
    lambda: __import__('qpsolvers').__version__))
results.append(check("redis",
    lambda: __import__('redis').__version__))

print()
passed = sum(results)
total = len(results)
if all(results):
    print(f"ALL {total}/{total} CHECKS PASSED")
else:
    print(f"FAILED: {passed}/{total} checks passed")
    sys.exit(1)
