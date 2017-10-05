# use the miniconda3 image as a basis
# this one is based on debian (8.5, currently)
FROM continuumio/miniconda3

MAINTAINER ilastik Team <team@ilastik.org>

ENTRYPOINT [ “/bin/bash”, “-c” ]

RUN ["/bin/bash", "-c", "conda install --yes -c ilastik-forge -c conda-forge ilastik-dependencies-no-solvers"]
