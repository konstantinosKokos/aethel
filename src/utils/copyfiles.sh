#/bin/bash
rsync -avm --include='*.py' -f 'hide,! */' Transformer/ test/
