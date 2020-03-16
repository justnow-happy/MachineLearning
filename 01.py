conda create -n mlbook python=3.7 anaconda
activate mlbook
jupyter contrib nbextension install --user
conda install -n mlbook -c conda-forge tensorflow
conda install -n mlbook -c conda-forge jupyter_contrib_nbextensions
jupyter nbextension enable toc2/main
jupyter notebook
conda install python-graphviz