import os

directory = "."

old_substring = "999"

new_substring = "1000"

for filename in os.listdir(directory):
    if old_substring in filename:
        new_filename = filename.replace(old_substring, new_substring)
        
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Rinominato: {filename} -> {new_filename}")