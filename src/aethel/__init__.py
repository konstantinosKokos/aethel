import sys
from .frontend import ProofBank
from .mill import terms

# backwards compatibility with 1.0.0x
this_package = __package__
old_package = 'LassyExtraction'
sys.modules[old_package] = this_package
sys.modules[f'{old_package}.mill.terms'] = terms