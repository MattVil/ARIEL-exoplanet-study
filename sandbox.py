import os
import random

DATA_DIR = "I:\CNES\ml_data_challenge_database\ml_data_challenge_database"

f_names = []
with open(os.path.join(DATA_DIR, "noisy_train.txt"), 'r') as f:
  for line in f.readlines():
    f_names.append(line.replace('\n', ''))

random.shuffle(f_names)
num_files = len(f_names)
split = int(0.8*num_files)
f_names_train = f_names[:split]
f_names_valid = f_names[split:]

with open(os.path.join(DATA_DIR, "noisy_train_t.txt"), 'w') as f:
  for name in f_names_train:
    f.write(name+'\n')
    
with open(os.path.join(DATA_DIR, "noisy_train_v.txt"), 'w') as f:
  for name in f_names_valid:
    f.write(name+'\n')