$Env:TRITON_VERSION = "25.01"  
docker run --gpus=all -it --rm `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v D:\WORKING\Triton\mnist_cnn:/models `
  nvcr.io/nvidia/tritonserver:$Env:TRITON_VERSION-py3 `
  tritonserver --model-repository=/models
