# About zipformer-streaming-ovep-python-static
This Python pipeline is to show how to run [Zipformer Streaming](https://huggingface.co/stdo/PengChengStarling/tree/main) ASR on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is forked from [Zipformer](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/onnx_pretrained-streaming.py) of [Icefall](https://github.com/k2-fsa/icefall/tree/master) project 

Audio samples ("```en.wav```", "```ja.wav```" and "```zh.wav```") are downloaded from [Hugging Face sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10/tree/main/test_wavs)


## Key features
* On-line (streaming) mode
* Support Chinese, English, Russian, Vietnamese, Japanese, Thai, Indonesian, and Arabic
* Models are converted to static (mainly for NPU)

# Quick Steps
## Download and convert models
Visit https://huggingface.co/stdo/PengChengStarling/tree/main, download the following files
```
decoder-epoch-75-avg-11-chunk-16-left-128.onnx
encoder-epoch-75-avg-11-chunk-16-left-128.onnx
joiner-epoch-75-avg-11-chunk-16-left-128.onnx
tokens.txt
```
Run the following command to install dependencies
```
pip install -r requirements.txt
```
Convert models from dynamic to static (required for NPU)
```
python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param N --dim_value 1 encoder-epoch-75-avg-11-chunk-16-left-128.onnx encoder-epoch-75-avg-11-chunk-16-left-128_static.onnx

python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param N --dim_value 1 decoder-epoch-75-avg-11-chunk-16-left-128.onnx decoder-epoch-75-avg-11-chunk-16-left-128_static.onnx

python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param N --dim_value 1 joiner-epoch-75-avg-11-chunk-16-left-128.onnx joiner-epoch-75-avg-11-chunk-16-left-128_static.onnx
```

## Run
Usage
```
usage: onnx_pretrained-streaming.py [-h] --encoder-model-filename ENCODER_MODEL_FILENAME
                                    --decoder-model-filename DECODER_MODEL_FILENAME
                                    --joiner-model-filename JOINER_MODEL_FILENAME [--tokens TOKENS] [--device DEVICE]
                                    [--loopback]
                                    [sound_file]

positional arguments:
  sound_file            The input sound file to transcribe. Supported formats are those supported by
                        torchaudio.load(). For example, wav and flac are supported. The sample rate has to be 16kHz.
                        (default: None)

options:
  -h, --help            show this help message and exit
  --encoder-model-filename ENCODER_MODEL_FILENAME
                        Path to the encoder onnx model. (default: None)
  --decoder-model-filename DECODER_MODEL_FILENAME
                        Path to the decoder onnx model. (default: None)
  --joiner-model-filename JOINER_MODEL_FILENAME
                        Path to the joiner onnx model. (default: None)
  --tokens TOKENS       Path to tokens.txt. (default: None)
  --device DEVICE       Execution device. Use 'CPU', 'GPU', 'NPU' for OpenVINO. If not specified, default
                        CPUExecutionProvider will be used. (default: None)
  --loopback            The sound file will be played synchronously while being processed. (default: False)
```
Example: Transcribe and audio file, run on GPU
```
python onnx_pretrained-streaming.py ^
--encoder-model-filename encoder-epoch-75-avg-11-chunk-16-left-128_static.onnx ^
--decoder-model-filename decoder-epoch-75-avg-11-chunk-16-left-128_static.onnx ^
--joiner-model-filename joiner-epoch-75-avg-11-chunk-16-left-128_static.onnx ^
--tokens tokens.txt ^
--device GPU ^
zh.wav
```
The ```--device``` can be "CPU", "GPU", and "NPU". If ```--device``` is not specified, CPUExecutionProvider will be used by default

:warning:[NOTE] The 1st time running on NPU will take long time (about 3 minutes) on model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been enabled for NPU to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be faster as model compilation has been skipped.
## Tested devices
The pipeline has been verified working on a ```Intel(R) Core(TM) Ultra 5 238V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 130V GPU (16GB), driver 32.0.101.8247 (10/22/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4621 (2/25/2026)```
### Result
| Sample             | CPU | GPU | NPU |
|--------------------|-----|-----|-----|
| English (en.wav)   | OK  | OK  | OK  |
| Japanese (ja.wav)  | OK  | OK  | OK  |
| Chinese (zh.wav)   | OK  | OK  | OK  |

### Sample log
```
(python313_venv) C:\GitHub\zipformer-streaming-ovep-python-static>python onnx_pretrained-streaming.py --encoder-model-filename encoder-epoch-75-avg-11-chunk-16-left-128_static.onnx --decoder-model-filename decoder-epoch-75-avg-11-chunk-16-left-128_static.onnx --joiner-model-filename joiner-epoch-75-avg-11-chunk-16-left-128_static.onnx --tokens tokens.txt --device GPU --loopback en.wav
2026-03-10 14:17:42,084 INFO [onnx_pretrained-streaming.py:517] {'encoder_model_filename': 'encoder-epoch-75-avg-11-chunk-16-left-128_static.onnx', 'decoder_model_filename': 'decoder-epoch-75-avg-11-chunk-16-left-128_static.onnx', 'joiner_model_filename': 'joiner-epoch-75-avg-11-chunk-16-left-128_static.onnx', 'tokens': 'tokens.txt', 'device': 'GPU', 'loopback': True, 'sound_file': 'en.wav'}
Device: OpenVINO EP with device = GPU
2026-03-10 14:17:49,437 INFO [onnx_pretrained-streaming.py:191] encoder_meta={'encoder_dims': '192,384,768,1024,768,384', 'version': '1', 'model_type': 'zipformer2', 'model_author': 'k2-fsa', 'comment': 'streaming zipformer2', 'decode_chunk_len': '32', 'num_encoder_layers': '2,2,4,5,4,2', 'T': '45', 'cnn_module_kernels': '31,31,15,15,15,31', 'left_context_len': '128,64,32,16,32,64', 'query_head_dims': '32,32,32,32,32,32', 'value_head_dims': '12,12,12,12,12,12', 'num_heads': '4,4,4,8,4,4'}
2026-03-10 14:17:49,437 INFO [onnx_pretrained-streaming.py:218] decode_chunk_len: 32
2026-03-10 14:17:49,437 INFO [onnx_pretrained-streaming.py:219] T: 45
2026-03-10 14:17:49,437 INFO [onnx_pretrained-streaming.py:220] num_encoder_layers: [2, 2, 4, 5, 4, 2]
2026-03-10 14:17:49,437 INFO [onnx_pretrained-streaming.py:221] encoder_dims: [192, 384, 768, 1024, 768, 384]
2026-03-10 14:17:49,438 INFO [onnx_pretrained-streaming.py:222] cnn_module_kernels: [31, 31, 15, 15, 15, 31]
2026-03-10 14:17:49,438 INFO [onnx_pretrained-streaming.py:223] left_context_len: [128, 64, 32, 16, 32, 64]
2026-03-10 14:17:49,438 INFO [onnx_pretrained-streaming.py:224] query_head_dims: [32, 32, 32, 32, 32, 32]
2026-03-10 14:17:49,438 INFO [onnx_pretrained-streaming.py:225] value_head_dims: [12, 12, 12, 12, 12, 12]
2026-03-10 14:17:49,438 INFO [onnx_pretrained-streaming.py:226] num_heads: [4, 4, 4, 8, 4, 4]
2026-03-10 14:17:49,607 INFO [onnx_pretrained-streaming.py:284] context_size: 2
2026-03-10 14:17:49,607 INFO [onnx_pretrained-streaming.py:285] vocab_size: 16000
2026-03-10 14:17:49,754 INFO [onnx_pretrained-streaming.py:298] joiner_dim: 512
2026-03-10 14:17:49,754 INFO [onnx_pretrained-streaming.py:530] Constructing Fbank computer
[output device]
{'name': 'Headset Earphone (Jabra UC VOIC', 'index': 5, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_low_input_latency': 0.09, 'default_low_output_latency': 0.09, 'default_high_input_latency': 0.18, 'default_high_output_latency': 0.18, 'default_samplerate': 44100.0}
2026-03-10 14:17:49,873 INFO [onnx_pretrained-streaming.py:562] Reading sound file: en.wav
C:\Python\python313_venv\Lib\site-packages\torchaudio\_backend\utils.py:213: UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec` under the hood. Some parameters like ``normalize``, ``format``, ``buffer_size``, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.
  warnings.warn(
2026-03-10 14:17:50,988 INFO [onnx_pretrained-streaming.py:619] Partial:
2026-03-10 14:17:51,041 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER
2026-03-10 14:17:52,001 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY
2026-03-10 14:17:52,057 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHT
2026-03-10 14:17:52,952 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL
2026-03-10 14:17:53,005 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE Y
2026-03-10 14:17:53,059 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW
2026-03-10 14:17:53,950 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS
2026-03-10 14:17:54,004 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD
2026-03-10 14:17:54,054 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP
2026-03-10 14:17:54,103 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE
2026-03-10 14:17:54,952 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE
2026-03-10 14:17:55,003 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE
2026-03-10 14:17:55,057 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUA
2026-03-10 14:17:55,950 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID
2026-03-10 14:17:56,003 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER
2026-03-10 14:17:56,056 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE
2026-03-10 14:17:56,950 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHE
2026-03-10 14:17:57,002 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHEL
2026-03-10 14:17:57,056 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
2026-03-10 14:17:57,057 INFO [onnx_pretrained-streaming.py:627] en.wav
2026-03-10 14:17:57,057 INFO [onnx_pretrained-streaming.py:628] AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
2026-03-10 14:17:57,057 INFO [onnx_pretrained-streaming.py:630] Decoding Done
```
[Full log](https://github.com/luke-lin-vmc/zipformer-streaming-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Known issues
1. If upgrading openvino to 2025.4.1, running on CPU and NPU shows below repeated words. This will be fixed in openvino 2026.0.0
```
2026-03-10 14:19:35,738 INFO [onnx_pretrained-streaming.py:619] Partial:
2026-03-10 14:19:35,776 INFO [onnx_pretrained-streaming.py:619] Partial: AFTER
2026-03-10 14:19:36,735 INFO [onnx_pretrained-streaming.py:619] Partial: AFTERAFTER AFTEREAREAR EARLY
2026-03-10 14:19:36,773 INFO [onnx_pretrained-streaming.py:619] Partial: AFTERAFTER AFTEREAREAR EARLY EARLY EARLY EARLY EARLY EARLY EARLY EARLY
```

2. If the following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
This would be caused by that both ```onnxruntime``` and ```onnxruntime-openvino``` are installed. Solution is to remove both of them then re-install ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
Or simply to re-install ```onnxruntime-openvino``` if you would like to keep ```onnxruntime```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
