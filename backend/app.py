import traceback

try:
    from main import app
except Exception:
    print("========== MAIN.PY IMPORT FAILED ==========")
    traceback.print_exc()
    print("===========================================")
    raise
