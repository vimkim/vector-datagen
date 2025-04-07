gen-256dim-all: gen-256dim-75000rows gen-256dim-150000rows gen-256dim-300000rows

gen-256dim-300000rows:
    uv run main.py --seed 0 --rows 300000 --dim 256

gen-256dim-150000rows:
    uv run main.py --seed 0 --rows 150000 --dim 256

gen-256dim-75000rows:
    uv run main.py --seed 0 --rows 75000 --dim 256
