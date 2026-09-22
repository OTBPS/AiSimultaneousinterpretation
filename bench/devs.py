import openvino as ov
core = ov.Core()
for d in core.available_devices:
    print("="*50)
    print("DEVICE:", d)
    for k in ["FULL_DEVICE_NAME","DEVICE_TYPE","GPU_EXECUTION_UNITS_COUNT","OPTIMIZATION_CAPABILITIES","DEVICE_ARCHITECTURE","GPU_DEVICE_TOTAL_MEM_SIZE","MAX_BATCH_SIZE"]:
        try: print(f"  {k}: {core.get_property(d,k)}")
        except Exception: pass
