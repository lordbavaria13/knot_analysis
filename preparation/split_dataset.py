import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def main():
    input_dir = Path('dataset_processed')
    output_dir = Path('dataset')

    classes = ['correct', 'wrong']
    split_ratio = 0.8 

    if not input_dir.exists():
        print(f"Fehler: Ordner '{input_dir}' existiert nicht!")
        print("Bitte stelle sicher, dass der Pfad korrekt ist.")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for cls in classes:
        src = input_dir / cls
        if not src.exists():
            print(f"Warnung: Quellordner '{src}' nicht gefunden. Überspringe...")
            continue

        files = list(src.glob('*.png')) + list(src.glob('*.jpg'))
        if len(files) == 0:
            print(f"Warnung: Keine Bilder in '{src}' gefunden.")
            continue


        series_dict = defaultdict(list) # group picture series
        for f in files:
            parts = f.name.split('_')
            if len(parts) >= 2:
                series_id = f"{parts[0]}_{parts[1]}"
            else:
                series_id = f.stem # Fallback
            series_dict[series_id].append(f)

        series_ids = list(series_dict.keys()) # no random shuffle! series have to be kept together for train/val split
        random.shuffle(series_ids)

        target_train_count = int(len(files) * split_ratio)
        
        train_files = []
        val_files = []

        for sid in series_ids:
            if len(train_files) < target_train_count:
                train_files.extend(series_dict[sid])
            else:
                val_files.extend(series_dict[sid])

        train_dst = output_dir / 'train' / cls
        val_dst = output_dir / 'val' / cls
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy(f, train_dst / f.name)
        for f in val_files:
            shutil.copy(f, val_dst / f.name)

        print(f"Klasse '{cls}': {len(train_files)} Trainingsbilder, {len(val_files)} Validierungsbilder "
              f"(aus insgesamt {len(series_ids)} Serien)")

    print("\nFertig! Der Ordner 'dataset' ist bereit.")
    print("Nächster Schritt: python train_resnet18.py")

if __name__ == '__main__':
    main()