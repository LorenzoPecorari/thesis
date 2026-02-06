from concurrent.futures import ThreadPoolExecutor, wait

def printer(arg1, arg2):
    print(f"Hello {arg1} - {arg2}")
    
with ThreadPoolExecutor(max_workers=3) as executor:
    executor.submit(printer, 1, 2)
    executor.submit(printer, 2, 2)
    executor.submit(printer, 3, 2)
    
