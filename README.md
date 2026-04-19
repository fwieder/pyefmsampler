##  Installation

### Install from GitHub (recommended)

You can install `pyefmsampler` directly using pip:
```
pip install git+https://github.com/fwieder/pyefmsampler.git
```

## Usage

```
from pyefmsampler.helpers import FluxCone,find_objective_index,unsplit_vector
from pyefmsampler.functions import sample_efms
import cobra
import numpy as np


if __name__ == "__main__":
    
    model_id = "e_coli_core"                            # Any model_id from the BiGG database is possible here
    model = FluxCone.from_bigg_id(model_id)             # Create Fluxcone-object that contains all relevant information
    

    objective_index = find_objective_index(model) # Determine the index of the optimisation target defined in the sbml file
    
    max_efms = 2000                                     # Choose how many EFMs are sampled
    
    sample = sample_efms(model,objective_index, max_efms = max_efms)
```
