FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN pip install --no-cache-dir torchmetrics tabulate colorama pytorch-msssim

WORKDIR /workspace

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/bin/bash"]