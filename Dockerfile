FROM continuumio/anaconda3

RUN mkdir -p /morevis
COPY enviroment.yml /morevis/enviroment.yml

RUN conda init bash && \
    . /root/.bashrc && \
    conda env create -f /morevis/enviroment.yml && \
    conda activate morevis 

ADD app/ /morevis/app
ADD data/ /morevis/data
#WORKDIR /morevis

RUN echo 'conda activate morevis' >> /root/.bashrc

WORKDIR /morevis/app
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["python -m flask run -h 0.0.0.0"]
    
