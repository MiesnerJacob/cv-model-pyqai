FROM huggingface/transformers-pytorch-gpu

COPY . .

RUN pip install fastapi uvicorn

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 5000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]