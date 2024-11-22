## Install

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install open-mpi
pip install mpi4py
CFLAGS="-Wno-implicit-function-declaration -Wno-deprecated-declarations" python setup.py install
```

## Run
```
python example_Si.py -f models/Si.json
python example_validate.py models/Si.json models/Si.db
python example_sparse.py models/Si.json
python example_sampling.py models/Si.json models/Si.db
python example_cell_size.py models/Si.json models/Si.db
```

## Todo
```
Instruction on GULP will be followed
```
